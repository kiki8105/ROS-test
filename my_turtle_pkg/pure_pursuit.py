"""AMCL 위치로 지정 경유점을 추종하는 교육용 Pure Pursuit 노드.

이 노드는 장애물을 감지하거나 회피하지 않는다. 저장 지도와 AMCL은 현재 위치만
제공하며, 출력 속도 명령은 설정된 경유점을 향해 직접 계산된다.
"""
import math

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor, SetParametersResult
from rclpy.node import Node


def normalize_angle(angle):
    """각도를 -pi 이상 pi 미만 범위로 정규화한다."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rate_limit(current, target, limit, dt):
    """현재 값이 한 주기에 limit * dt보다 많이 변하지 않게 한다."""
    delta = max(-limit * dt, min(limit * dt, target - current))
    return current + delta


def pursuit_command(x, y, yaw, goal_x, goal_y, linear_velocity,
                    lookahead_distance, max_angular_velocity):
    """Pure Pursuit 공식으로 거리, 방향오차, 목표 선·각속도를 계산한다."""
    dx, dy = goal_x - x, goal_y - y
    distance = math.hypot(dx, dy)
    target_angle = math.atan2(dy, dx)
    alpha = normalize_angle(target_angle - yaw)
    angular = linear_velocity * (2.0 * math.sin(alpha)) / lookahead_distance
    angular = max(-max_angular_velocity, min(max_angular_velocity, angular))
    return distance, alpha, linear_velocity, angular


def floating_descriptor(description, minimum, maximum, step):
    return ParameterDescriptor(
        description=description,
        floating_point_range=[FloatingPointRange(
            from_value=float(minimum), to_value=float(maximum), step=float(step))])


class PurePursuit(Node):
    obstacle_avoidance_enabled = False
    immediate_stop_on_completion = False
    refresh_pose_timeout_on_enable = True

    def __init__(self, node_name='pure_pursuit_node'):
        super().__init__(node_name)

        self.declare_parameter(
            'lookahead_distance', 0.5,
            floating_descriptor(
                '전방 주시 거리 [m]. 높이면 곡선이 완만해지고 낮추면 목표점에 민감하게 선회합니다.',
                0.05, 5.0, 0.05))
        self.declare_parameter(
            'linear_velocity', 0.1,
            floating_descriptor(
                '목표 선속도 [m/s]. Waffle Pi 실내 Pure Pursuit 권장 시작값은 0.08~0.15입니다.',
                0.0, 0.26, 0.01))
        self.declare_parameter(
            'drive_enabled', False,
            ParameterDescriptor(
                description=(
                    'true일 때만 자동주행합니다. 초기 위치 지정 후 경로를 확인하고 '
                    '마지막에 켜세요. false로 바꾸면 즉시 정지합니다.')))
        self.declare_parameter(
            'max_angular_velocity', 1.0,
            floating_descriptor(
                '최대 각속도 [rad/s]. 높이면 급하게 돌고 낮추면 선회가 완만해집니다.',
                0.05, 2.0, 0.05))
        self.declare_parameter(
            'linear_acceleration_limit', 0.15,
            floating_descriptor(
                '선속도 가속·감속 한계 [m/s^2]. 낮추면 출발과 정지가 부드러워집니다.',
                0.01, 2.5, 0.01))
        self.declare_parameter(
            'angular_acceleration_limit', 0.8,
            floating_descriptor(
                '각속도 가속·감속 한계 [rad/s^2]. 낮추면 급선회와 방향 전환이 줄어듭니다.',
                0.05, 5.0, 0.05))
        self.declare_parameter(
            'goal_tolerance', 0.2,
            floating_descriptor(
                '경유점 도착으로 인정하는 거리 [m]. 높이면 쉽게 통과하고 낮추면 더 정확히 접근합니다.',
                0.02, 2.0, 0.01))
        self.declare_parameter(
            'pose_timeout', 0.0,
            floating_descriptor(
                '위치 입력이 이 시간보다 오래 갱신되지 않으면 정지합니다 [s]. '
                'v2는 AMCL, v3는 AMCL+odom을 사용하며 0이면 검사를 끕니다.',
                0.0, 120.0, 0.5))
        self.declare_parameter(
            'control_period', 0.1,
            ParameterDescriptor(
                description='제어 명령 발행 간격 [s]. 변경하려면 노드를 재시작해야 합니다.',
                read_only=True,
                floating_point_range=[FloatingPointRange(
                    from_value=0.02, to_value=1.0, step=0.01)]))
        self.declare_parameter(
            'waypoints_x', [6.27],
            ParameterDescriptor(
                description='순서대로 방문할 경유점 x 좌표 배열 [m]. YAML 수정 후 재시작합니다.',
                read_only=True))
        self.declare_parameter(
            'waypoints_y', [0.56],
            ParameterDescriptor(
                description='순서대로 방문할 경유점 y 좌표 배열 [m]. YAML 수정 후 재시작합니다.',
                read_only=True))
        self.declare_parameter(
            'cmd_vel_topic', '/cmd_vel',
            ParameterDescriptor(
                description='속도 명령을 발행할 Twist 토픽입니다. 변경하려면 재시작합니다.',
                read_only=True))
        self.declare_parameter(
            'pose_topic', '/amcl_pose',
            ParameterDescriptor(
                description='현재 지도 위치를 받을 AMCL pose 토픽입니다. 변경하려면 재시작합니다.',
                read_only=True))
        self.declare_parameter(
            'goal_topic', '/goal_pose',
            ParameterDescriptor(
                description=(
                    '조종기 지도에서 선택한 단일 목표 PoseStamped 토픽입니다. '
                    '변경하려면 재시작합니다.'),
                read_only=True))

        xs = [float(value) for value in self.get_parameter('waypoints_x').value]
        ys = [float(value) for value in self.get_parameter('waypoints_y').value]
        if not xs or len(xs) != len(ys) or not all(math.isfinite(v) for v in xs + ys):
            raise ValueError('waypoints_x와 waypoints_y는 길이가 같은 유한 좌표 배열이어야 합니다.')
        self.path = list(zip(xs, ys))
        period = float(self.get_parameter('control_period').value)
        cmd_topic = str(self.get_parameter('cmd_vel_topic').value)
        pose_topic = str(self.get_parameter('pose_topic').value)
        goal_topic = str(self.get_parameter('goal_topic').value)

        self.publisher = self.create_publisher(Twist, cmd_topic, 10)
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped, pose_topic, self.pose_callback, 10)
        self.goal_subscription = self.create_subscription(
            PoseStamped, goal_topic, self.goal_callback, 10)
        self.add_on_set_parameters_callback(self.validate_parameters)

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.current_waypoint_index = 0
        self.last_pose_time = None
        self.last_control_time = self.get_clock().now()
        self.is_localized = False
        self.completed = False
        self.last_drive_enabled = None
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info(
            f'Pure Pursuit 시작: {len(self.path)}개 경유점, 제어주기={period:.2f}s, '
            f'cmd={cmd_topic}, pose={pose_topic}, goal={goal_topic}')
        if self.obstacle_avoidance_enabled:
            self.get_logger().info('LiDAR 반응형 장애물 회피 기능이 활성화됐습니다.')
        else:
            self.get_logger().warn(
                '이 교육용 노드에는 장애물 감지·회피 기능이 없습니다. 경로가 비어 있는지 먼저 확인하세요.')
        self.get_logger().info(
            '조종기 지도에서 초기 위치를 지정한 뒤 drive_enabled=true로 바꾸면 주행합니다.')

    def validate_parameters(self, parameters):
        positive = {
            'lookahead_distance', 'max_angular_velocity',
            'linear_acceleration_limit', 'angular_acceleration_limit',
            'goal_tolerance',
        }
        for parameter in parameters:
            if parameter.name == 'linear_velocity' and float(parameter.value) < 0.0:
                return SetParametersResult(
                    successful=False, reason='linear_velocity는 0 이상이어야 합니다.')
            if parameter.name in positive and float(parameter.value) <= 0.0:
                return SetParametersResult(
                    successful=False, reason=f'{parameter.name}은 0보다 커야 합니다.')
            if parameter.name == 'pose_timeout' and float(parameter.value) < 0.0:
                return SetParametersResult(
                    successful=False, reason='pose_timeout은 0 이상이어야 합니다.')
        return SetParametersResult(successful=True)

    def pose_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.last_pose_time = self.get_clock().now()
        if not self.is_localized:
            self.get_logger().info(
                f'AMCL 위치 수신: x={self.current_x:.3f}, y={self.current_y:.3f}, '
                f'yaw={self.current_yaw:.3f}')
        self.is_localized = True

    def goal_callback(self, msg):
        """Replace the YAML path with one valid map goal from the controller."""
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        if not math.isfinite(x) or not math.isfinite(y):
            self.get_logger().error('유효하지 않은 /goal_pose 좌표를 무시합니다.')
            return
        self.path = [(x, y)]
        self.current_waypoint_index = 0
        self.completed = False
        self.get_logger().info(
            f'새 /goal_pose 적용: frame={msg.header.frame_id or "(없음)"}, '
            f'x={x:.3f}, y={y:.3f}')

    def publish_stop(self, message=None):
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.publisher.publish(Twist())
        if message:
            self.get_logger().info(message)

    def adjust_command_for_environment(
            self, desired_linear, desired_angular, distance, alpha,
            goal_x, goal_y, now):
        """환경 센서가 없는 v2에서는 Pure Pursuit 명령을 그대로 사용한다."""
        return desired_linear, desired_angular

    def control_loop(self):
        now = self.get_clock().now()
        dt = (now - self.last_control_time).nanoseconds / 1e9
        period = float(self.get_parameter('control_period').value)
        if not math.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            dt = period
        self.last_control_time = now

        drive_enabled = bool(self.get_parameter('drive_enabled').value)
        if drive_enabled != self.last_drive_enabled:
            if drive_enabled:
                self.last_control_time = now
                # AMCL은 정지 중 pose를 발행하지 않을 수 있으므로 재출발
                # 순간부터 timeout 유예를 새로 준다.
                if self.is_localized and self.refresh_pose_timeout_on_enable:
                    self.last_pose_time = now
                self.get_logger().info('자동주행 시작: drive_enabled=true')
            else:
                # 비활성 전환 시에만 0을 발행한다. 이후에는 수동 조종기의
                # /cmd_vel 명령과 계속 경쟁하지 않는다.
                self.publish_stop('자동주행 잠금: drive_enabled=false')
            self.last_drive_enabled = drive_enabled
        if not drive_enabled:
            return

        if not self.is_localized:
            self.publish_stop()
            self.get_logger().info('AMCL 초기 위치 대기 중...', throttle_duration_sec=3.0)
            return

        pose_timeout = float(self.get_parameter('pose_timeout').value)
        pose_age = (now - self.last_pose_time).nanoseconds / 1e9
        if pose_timeout > 0.0 and (
                not math.isfinite(pose_age)
                or pose_age < 0.0
                or pose_age > pose_timeout):
            self.publish_stop()
            self.get_logger().warn(
                f'위치 입력 시간이 유효하지 않거나 {pose_age:.1f}s 동안 '
                '갱신되지 않아 정지합니다.',
                throttle_duration_sec=2.0)
            return

        tolerance = float(self.get_parameter('goal_tolerance').value)
        while self.current_waypoint_index < len(self.path):
            goal_x, goal_y = self.path[self.current_waypoint_index]
            distance = math.hypot(goal_x - self.current_x, goal_y - self.current_y)
            if distance >= tolerance:
                break
            self.get_logger().info(
                f'경유점 {self.current_waypoint_index + 1}/{len(self.path)} 도착: '
                f'({goal_x:.3f}, {goal_y:.3f})')
            self.current_waypoint_index += 1

        if self.current_waypoint_index >= len(self.path):
            if not self.completed:
                self.completed = True
                if self.immediate_stop_on_completion:
                    self.get_logger().info('모든 경유점 완료: 즉시 정지합니다.')
                else:
                    self.get_logger().info(
                        '모든 경유점 완료: 가속도 한계에 맞춰 정지합니다.')
            if self.immediate_stop_on_completion:
                self.publish_stop()
                return
            linear_accel = float(self.get_parameter('linear_acceleration_limit').value)
            angular_accel = float(self.get_parameter('angular_acceleration_limit').value)
            self.current_linear = rate_limit(
                self.current_linear, 0.0, linear_accel, dt)
            self.current_angular = rate_limit(
                self.current_angular, 0.0, angular_accel, dt)
            if abs(self.current_linear) < 1e-4:
                self.current_linear = 0.0
            if abs(self.current_angular) < 1e-4:
                self.current_angular = 0.0
            command = Twist()
            command.linear.x = self.current_linear
            command.angular.z = self.current_angular
            self.publisher.publish(command)
            return

        goal_x, goal_y = self.path[self.current_waypoint_index]
        distance = math.hypot(goal_x - self.current_x, goal_y - self.current_y)
        linear_velocity = float(self.get_parameter('linear_velocity').value)
        lookahead = float(self.get_parameter('lookahead_distance').value)
        max_angular = float(self.get_parameter('max_angular_velocity').value)
        linear_accel = float(self.get_parameter('linear_acceleration_limit').value)
        # 허용 반경 직전까지 현재 가속도 한계로 정지할 수 있는 속도를 계산한다.
        # v = sqrt(2*a*d)를 쓰면 마지막 경유점에서 급정지하지 않는다.
        stopping_speed = math.sqrt(
            2.0 * linear_accel * max(0.0, distance - tolerance))
        target_linear = min(linear_velocity, stopping_speed)
        distance, alpha, desired_linear, desired_angular = pursuit_command(
            self.current_x, self.current_y, self.current_yaw,
            goal_x, goal_y, target_linear, lookahead, max_angular)

        # 목표가 옆이나 뒤에 있으면 sin(pi)=0인 Pure Pursuit 공식만으로는
        # 직진할 수 있으므로, 먼저 제자리에서 목표 방향으로 정렬한다.
        if linear_velocity <= 1e-6:
            desired_linear = 0.0
            desired_angular = 0.0
        elif abs(alpha) >= math.pi / 2.0:
            desired_linear = 0.0
            desired_angular = math.copysign(max_angular, alpha)

        desired_linear, desired_angular = self.adjust_command_for_environment(
            desired_linear, desired_angular, distance, alpha,
            goal_x, goal_y, now)

        angular_accel = float(self.get_parameter('angular_acceleration_limit').value)
        self.current_linear = rate_limit(
            self.current_linear, desired_linear, linear_accel, dt)
        self.current_angular = rate_limit(
            self.current_angular, desired_angular, angular_accel, dt)
        # max_angular_velocity는 동적 변경 직후에도 반드시 지키는 절대 한계다.
        self.current_angular = max(
            -max_angular, min(max_angular, self.current_angular))

        command = Twist()
        command.linear.x = self.current_linear
        command.angular.z = self.current_angular
        self.publisher.publish(command)
        self.get_logger().info(
            f'WP {self.current_waypoint_index + 1}/{len(self.path)} '
            f'd={distance:.2f}m alpha={alpha:.2f}rad '
            f'v={self.current_linear:.2f}m/s w={self.current_angular:.2f}rad/s',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_stop()
        except KeyboardInterrupt:
            pass
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()

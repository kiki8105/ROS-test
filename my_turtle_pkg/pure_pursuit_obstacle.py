"""Obstacle-aware Pure Pursuit practice node for TurtleBot3 Waffle Pi.

The controller follows map-frame waypoints, validates AMCL and LiDAR input,
and uses a small TURN -> BYPASS -> TRACK state machine for local avoidance.
It remains an educational local controller; Nav2 is preferable for general
path planning in cluttered environments.
"""

import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
    SetParametersResult,
)
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


# turtlebot3_waffle_pi.urdf 기준 base_footprint 좌표 [m].
WAFFLE_PI_LIDAR_X = -0.064
WAFFLE_PI_FRONT_X = 0.080


def normalize_angle(angle):
    """Return an angle in the [-pi, pi) interval."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value, limit):
    """Clamp a signed value to +/- limit."""
    return max(-float(limit), min(float(limit), float(value)))


def rate_limit(current, target, limit, dt):
    """Limit how far a command may change during one control period."""
    maximum_change = max(0.0, float(limit)) * max(0.0, float(dt))
    delta = max(-maximum_change, min(maximum_change, target - current))
    return current + delta


def floating_descriptor(description, minimum, maximum, step=0.01):
    """Create a ROS floating-point parameter descriptor."""
    return ParameterDescriptor(
        description=description,
        floating_point_range=[FloatingPointRange(
            from_value=float(minimum),
            to_value=float(maximum),
            step=float(step),
        )],
    )


def integer_descriptor(description, minimum, maximum, step=1):
    """Create a ROS integer parameter descriptor."""
    return ParameterDescriptor(
        description=description,
        integer_range=[IntegerRange(
            from_value=int(minimum),
            to_value=int(maximum),
            step=int(step),
        )],
    )


def scan_points(scan, lidar_x=WAFFLE_PI_LIDAR_X):
    """Convert valid finite LaserScan returns to base-frame x/y points.

    Positive infinity is a valid open-space reading and contributes to the
    usable sample count. NaN, negative infinity, and below-range values are
    invalid. This distinction prevents a failed sensor from looking clear.
    """
    if (
            scan is None
            or not scan.ranges
            or not math.isfinite(float(scan.angle_min))
            or not math.isfinite(float(scan.angle_increment))
            or abs(float(scan.angle_increment)) < 1e-12
            or not math.isfinite(float(scan.range_min))
            or not math.isfinite(float(scan.range_max))
            or float(scan.range_min) < 0.0
            or float(scan.range_max) <= float(scan.range_min)):
        return [], 0

    minimum = float(scan.range_min)
    maximum = float(scan.range_max)
    range_epsilon = max(1e-6, maximum * 1e-6)
    points = []
    usable = 0
    for index, raw in enumerate(scan.ranges):
        try:
            distance = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(distance) or distance < minimum - range_epsilon:
            continue
        if math.isinf(distance):
            if distance > 0.0:
                usable += 1
            continue
        if distance > maximum + range_epsilon:
            continue

        usable += 1
        angle = float(scan.angle_min) + index * float(scan.angle_increment)
        points.append((
            float(lidar_x) + distance * math.cos(angle),
            distance * math.sin(angle),
        ))
    return points, usable


def scan_sector_distance(scan, center, half_width, percentile=0.10):
    """Return (near-distance percentile, usable-count) for an angle sector."""
    if (
            scan is None
            or not scan.ranges
            or not math.isfinite(float(scan.angle_min))
            or not math.isfinite(float(scan.angle_increment))
            or abs(float(scan.angle_increment)) < 1e-12
            or not math.isfinite(float(scan.range_min))
            or not math.isfinite(float(scan.range_max))
            or float(scan.range_min) < 0.0
            or float(scan.range_max) <= float(scan.range_min)):
        return math.nan, 0

    minimum = float(scan.range_min)
    maximum = float(scan.range_max)
    range_epsilon = max(1e-6, maximum * 1e-6)
    values = []
    for index, raw in enumerate(scan.ranges):
        angle = float(scan.angle_min) + index * float(scan.angle_increment)
        if abs(normalize_angle(angle - center)) > half_width:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(value) or value < minimum - range_epsilon:
            continue
        if math.isinf(value):
            if value < 0.0:
                continue
            value = maximum
        elif value > maximum + range_epsilon:
            continue
        values.append(value)

    if not values:
        return math.nan, 0
    values.sort()
    percentile = max(0.0, min(1.0, float(percentile)))
    index = int((len(values) - 1) * percentile)
    return values[index], len(values)


def nth_smallest(values, count):
    """Return the count-th smallest value or infinity when hits are sparse."""
    ordered = sorted(values)
    return ordered[count - 1] if len(ordered) >= count else math.inf


def front_body_clearance(
        scan, half_width=0.18, min_hits=3,
        lidar_x=WAFFLE_PI_LIDAR_X,
        body_front_x=WAFFLE_PI_FRONT_X):
    """Return free distance from the body front inside its forward corridor."""
    points, _ = scan_points(scan, lidar_x)
    gaps = [
        max(0.0, x - body_front_x)
        for x, y in points
        if x > lidar_x and abs(y) < half_width
    ]
    return nth_smallest(gaps, max(1, int(min_hits)))


def minimum_radial_clearance(scan, lidar_x=WAFFLE_PI_LIDAR_X):
    """Return the nearest obstacle radius around the robot rotation centre."""
    points, _ = scan_points(scan, lidar_x)
    if not points:
        return math.inf
    return min(math.hypot(x, y) for x, y in points)


def select_lookahead_point(
        x, y, segment_start, path, waypoint_index, lookahead_distance):
    """Select a point lookahead_distance ahead along the waypoint polyline."""
    if waypoint_index >= len(path):
        return float(x), float(y)

    start_x, start_y = segment_start
    goal_x, goal_y = path[waypoint_index]
    vector_x = goal_x - start_x
    vector_y = goal_y - start_y
    length_squared = vector_x * vector_x + vector_y * vector_y
    if length_squared > 1e-12:
        raw_projection = (
            (x - start_x) * vector_x + (y - start_y) * vector_y
        ) / length_squared
        # 경유점을 크게 빗나간 채 지나친 경우에는 다음 선분을 미리
        # 바라보지 않고 현재 경유점으로 복귀한다.
        if raw_projection > 1.0:
            return goal_x, goal_y
        projection = max(0.0, raw_projection)
        cursor_x = start_x + projection * vector_x
        cursor_y = start_y + projection * vector_y
    else:
        cursor_x, cursor_y = goal_x, goal_y

    remaining = max(0.0, float(lookahead_distance))
    index = waypoint_index
    while index < len(path):
        goal_x, goal_y = path[index]
        delta_x = goal_x - cursor_x
        delta_y = goal_y - cursor_y
        segment_length = math.hypot(delta_x, delta_y)
        if segment_length > 1e-12 and remaining <= segment_length:
            ratio = remaining / segment_length
            return (
                cursor_x + ratio * delta_x,
                cursor_y + ratio * delta_y,
            )
        remaining = max(0.0, remaining - segment_length)
        cursor_x, cursor_y = goal_x, goal_y
        index += 1

    return path[-1]


class PurePursuitLogic(Node):
    """Follow map waypoints with guarded reactive obstacle avoidance."""

    def __init__(self, **kwargs):
        super().__init__('pure_pursuit_obstacle', **kwargs)

        self.declare_parameter(
            'drive_enabled', False,
            ParameterDescriptor(
                description='true일 때만 자동주행합니다. false 전환 시 즉시 정지합니다.'))
        self.declare_parameter(
            'scan_timeout', 0.5,
            floating_descriptor('LiDAR 미수신 정지 시간 [s].', 0.1, 2.0, 0.1))
        self.declare_parameter(
            'pose_timeout', 2.0,
            floating_descriptor('AMCL 위치 미수신 정지 시간 [s].', 0.1, 10.0, 0.1))
        self.declare_parameter(
            'linear_velocity', 0.08,
            floating_descriptor('정상 경로 추종 속도 [m/s].', 0.0, 0.22, 0.01))
        self.declare_parameter(
            'avoidance_angular_velocity', 0.23,
            floating_descriptor('장애물 앞 선회 속도 [rad/s].', 0.05, 1.0, 0.01))
        self.declare_parameter(
            'avoidance_linear_velocity', 0.05,
            floating_descriptor('장애물 옆 우회 속도 [m/s].', 0.01, 0.15, 0.01))
        self.declare_parameter(
            'obs_threshold', 0.40,
            floating_descriptor('차체 앞에서 회피를 시작할 거리 [m].', 0.15, 1.5, 0.01))
        self.declare_parameter(
            'lookahead_distance', 0.40,
            floating_descriptor('경로 위 전방 주시 거리 [m].', 0.10, 2.0, 0.01))
        self.declare_parameter(
            'max_angular_velocity', 0.60,
            floating_descriptor('모든 회전 명령의 절대 상한 [rad/s].', 0.10, 1.5, 0.01))
        self.declare_parameter(
            'goal_tolerance', 0.20,
            floating_descriptor('경유점 도착 허용 반경 [m].', 0.05, 0.50, 0.01))
        self.declare_parameter(
            'linear_acceleration_limit', 0.15,
            floating_descriptor('선형 가속·감속 한계 [m/s^2].', 0.01, 1.0, 0.01))
        self.declare_parameter(
            'angular_acceleration_limit', 0.80,
            floating_descriptor('각가속·감속 한계 [rad/s^2].', 0.05, 3.0, 0.05))
        self.declare_parameter(
            'avoidance_clear_margin', 0.10,
            floating_descriptor('회피 해제 거리 추가 여유 [m].', 0.02, 0.50, 0.01))
        self.declare_parameter(
            'avoidance_clear_hold_time', 0.80,
            floating_descriptor('전방이 연속으로 열려 있어야 하는 시간 [s].', 0.10, 2.0, 0.10))
        self.declare_parameter(
            'turn_direction_deadband', 0.05,
            floating_descriptor('좌우 거리 차이가 이 값 이하면 동률로 봅니다 [m].', 0.0, 0.50, 0.01))
        self.declare_parameter(
            'front_corridor_half_width', 0.18,
            floating_descriptor('차체 전방 검사 통로의 반폭 [m].', 0.14, 0.50, 0.01))
        self.declare_parameter(
            'scan_min_hits', 3,
            integer_descriptor('유효 스캔과 장애물에 필요한 최소 광선 수.', 1, 20))
        self.declare_parameter(
            'spin_clearance_radius', 0.27,
            floating_descriptor('제자리 선회에 필요한 차체 중심 반경 [m].', 0.24, 0.60, 0.01))
        self.declare_parameter(
            'minimum_turn_angle', 0.35,
            floating_descriptor('우회 전 반드시 선회할 최소 각도 [rad].', 0.10, 1.57, 0.01))
        self.declare_parameter(
            'side_clearance', 0.35,
            floating_descriptor('우회 중 장애물 측면 목표 간격 [m].', 0.15, 1.0, 0.01))
        self.declare_parameter(
            'wall_follow_gain', 0.80,
            floating_descriptor('우회 중 측면 거리 보정 게인.', 0.10, 5.0, 0.10))
        self.declare_parameter(
            'avoidance_min_duration', 1.0,
            floating_descriptor('경로 재진입 전 최소 회피 시간 [s].', 0.20, 10.0, 0.10))
        self.declare_parameter(
            'avoidance_max_duration', 15.0,
            floating_descriptor('회피 제한 시간 [s]. 초과하면 안전 정지합니다.', 2.0, 120.0, 1.0))
        self.declare_parameter(
            'avoidance_progress_distance', 0.08,
            floating_descriptor('재진입 전 목표에 가까워질 최소 거리 [m].', 0.0, 1.0, 0.01))
        self.declare_parameter(
            'rejoin_heading_tolerance', 0.60,
            floating_descriptor('경로 재진입 방향 오차 상한 [rad].', 0.10, 1.50, 0.01))
        self.declare_parameter(
            'heading_rotate_enter', 1.05,
            floating_descriptor('제자리 목표 정렬을 시작할 방향 오차 [rad].', 0.30, 2.80, 0.01))
        self.declare_parameter(
            'heading_rotate_exit', 0.70,
            floating_descriptor('제자리 목표 정렬을 끝낼 방향 오차 [rad].', 0.10, 2.50, 0.01))
        self.declare_parameter(
            'heading_gain', 1.0,
            floating_descriptor('목표 방향 정렬 비례 게인.', 0.10, 4.0, 0.10))
        self.declare_parameter(
            'control_period', 0.10,
            ParameterDescriptor(
                description='제어 주기 [s]. 변경하려면 재시작합니다.',
                read_only=True,
                floating_point_range=[FloatingPointRange(
                    from_value=0.02, to_value=1.0, step=0.01)]))

        self.add_on_set_parameters_callback(self.validate_parameters)
        initial_validation = self.validate_parameters([])
        if not initial_validation.successful:
            raise ValueError(
                f'Pure Pursuit 초기 파라미터 오류: {initial_validation.reason}')

        # 사용자가 작성한 map-frame 웨이포인트를 유지한다.
        self.path = [
            (1.65, 0.95),
            (1.64, 1.83),
            (-0.12, -0.04),
        ]
        if not self.path or not all(
                math.isfinite(value)
                for point in self.path for value in point):
            raise ValueError('경유점은 하나 이상의 유한한 x/y 좌표여야 합니다.')

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, 'amcl_pose', self.pose_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.is_localized = False
        self.last_pose_time = None

        self.latest_scan = None
        self.last_scan_time = None
        self.left_dist = math.inf
        self.front_dist = math.inf
        self.right_dist = math.inf

        self.current_waypoint_index = 0
        self.segment_start = None
        self.completed = False
        self.ever_started = False

        self.avoidance_state = 'TRACK'
        self.avoidance_active = False
        self.turn_direction = 0.0
        self.obstacle_clear_since = None
        self.avoidance_started = None
        self.avoidance_hit_distance = math.inf
        self.turn_start_yaw = 0.0

        self.heading_align_active = False
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.last_control_time = self.get_clock().now()
        self.last_drive_enabled = None

        period = float(self.get_parameter('control_period').value)
        self.timer = self.create_timer(period, self.control_loop)
        enabled = bool(self.get_parameter('drive_enabled').value)
        self.get_logger().info(
            f'Pure Pursuit 준비: {len(self.path)}개 경유점, '
            f'drive_enabled={str(enabled).lower()}, 제어주기={period:.2f}s')

    def validate_parameters(self, parameters):
        """Validate both startup values and live parameter updates."""
        limits = {
            'scan_timeout': (0.1, 2.0),
            'pose_timeout': (0.1, 10.0),
            'linear_velocity': (0.0, 0.22),
            'avoidance_angular_velocity': (0.05, 1.0),
            'avoidance_linear_velocity': (0.01, 0.15),
            'obs_threshold': (0.15, 1.5),
            'lookahead_distance': (0.10, 2.0),
            'max_angular_velocity': (0.10, 1.5),
            'goal_tolerance': (0.05, 0.50),
            'linear_acceleration_limit': (0.01, 1.0),
            'angular_acceleration_limit': (0.05, 3.0),
            'avoidance_clear_margin': (0.02, 0.50),
            'avoidance_clear_hold_time': (0.10, 2.0),
            'turn_direction_deadband': (0.0, 0.50),
            'front_corridor_half_width': (0.14, 0.50),
            'spin_clearance_radius': (0.24, 0.60),
            'minimum_turn_angle': (0.10, 1.57),
            'side_clearance': (0.15, 1.0),
            'wall_follow_gain': (0.10, 5.0),
            'avoidance_min_duration': (0.20, 10.0),
            'avoidance_max_duration': (2.0, 120.0),
            'avoidance_progress_distance': (0.0, 1.0),
            'rejoin_heading_tolerance': (0.10, 1.50),
            'heading_rotate_enter': (0.30, 2.80),
            'heading_rotate_exit': (0.10, 2.50),
            'heading_gain': (0.10, 4.0),
            'control_period': (0.02, 1.0),
        }
        values = {
            name: self.get_parameter(name).value
            for name in limits
        }
        values['scan_min_hits'] = self.get_parameter('scan_min_hits').value
        for parameter in parameters:
            if parameter.name in values:
                values[parameter.name] = parameter.value

        for name, (minimum, maximum) in limits.items():
            try:
                value = float(values[name])
            except (TypeError, ValueError):
                return SetParametersResult(
                    successful=False, reason=f'{name}에는 숫자가 필요합니다.')
            if not math.isfinite(value) or not minimum <= value <= maximum:
                return SetParametersResult(
                    successful=False,
                    reason=f'{name} 허용 범위는 {minimum}~{maximum}입니다.')

        try:
            scan_min_hits = int(values['scan_min_hits'])
        except (TypeError, ValueError):
            return SetParametersResult(
                successful=False, reason='scan_min_hits에는 정수가 필요합니다.')
        if not 1 <= scan_min_hits <= 20:
            return SetParametersResult(
                successful=False, reason='scan_min_hits 허용 범위는 1~20입니다.')
        if float(values['avoidance_angular_velocity']) > float(
                values['max_angular_velocity']):
            return SetParametersResult(
                successful=False,
                reason='avoidance_angular_velocity는 max_angular_velocity 이하여야 합니다.')
        if float(values['heading_rotate_exit']) >= float(
                values['heading_rotate_enter']):
            return SetParametersResult(
                successful=False,
                reason='heading_rotate_exit는 heading_rotate_enter보다 작아야 합니다.')
        if float(values['avoidance_min_duration']) >= float(
                values['avoidance_max_duration']):
            return SetParametersResult(
                successful=False,
                reason='avoidance_min_duration은 avoidance_max_duration보다 작아야 합니다.')
        return SetParametersResult(successful=True)

    def pose_callback(self, msg):
        """Accept only finite, normalizable AMCL poses."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        values = (
            float(position.x), float(position.y),
            float(orientation.x), float(orientation.y),
            float(orientation.z), float(orientation.w),
        )
        norm = math.sqrt(sum(value * value for value in values[2:]))
        if not all(math.isfinite(value) for value in values) or norm < 1e-9:
            self.get_logger().error(
                '유효하지 않은 AMCL pose를 무시합니다.',
                throttle_duration_sec=2.0)
            return

        qx, qy, qz, qw = (value / norm for value in values[2:])
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        if not math.isfinite(yaw):
            return

        self.current_x = values[0]
        self.current_y = values[1]
        self.current_yaw = yaw
        self.is_localized = True
        self.last_pose_time = self.get_clock().now()

    def scan_callback(self, msg):
        """Store the newest scan; validity is checked before every command."""
        self.latest_scan = msg
        self.last_scan_time = self.get_clock().now()

    def scan_metrics(self, alpha):
        """Return angle-aware LiDAR metrics, or None for an invalid scan."""
        scan = self.latest_scan
        minimum_hits = int(self.get_parameter('scan_min_hits').value)
        _, usable = scan_points(scan)
        if usable < max(12, minimum_hits * 4):
            return None

        sectors = {
            'front_sector': (0.0, math.radians(40.0), 0.05),
            'front_left': (math.radians(45.0), math.radians(25.0), 0.15),
            'front_right': (math.radians(-45.0), math.radians(25.0), 0.15),
            'left': (math.pi / 2.0, math.radians(30.0), 0.15),
            'right': (-math.pi / 2.0, math.radians(30.0), 0.15),
            'goal': (alpha, math.radians(12.0), 0.05),
        }
        distances = {}
        for name, (center, width, percentile) in sectors.items():
            distance, count = scan_sector_distance(
                scan, center, width, percentile)
            if name != 'goal' and count < minimum_hits:
                return None
            distances[name] = distance
            distances[f'{name}_count'] = count

        half_width = float(
            self.get_parameter('front_corridor_half_width').value)
        distances['front'] = front_body_clearance(
            scan, half_width, minimum_hits)
        distances['radial'] = minimum_radial_clearance(scan)
        return distances

    def publish_stop(self, message=None):
        """Publish an immediate zero command and reset the command history."""
        self.current_linear = 0.0
        self.current_angular = 0.0
        self.publisher_.publish(Twist())
        if message:
            self.get_logger().info(message)

    def publish_motion(
            self, desired_linear, desired_angular, dt,
            immediate_linear_stop=False):
        """Apply acceleration and absolute angular limits before publishing."""
        linear_limit = float(
            self.get_parameter('linear_acceleration_limit').value)
        angular_limit = float(
            self.get_parameter('angular_acceleration_limit').value)
        max_angular = float(
            self.get_parameter('max_angular_velocity').value)

        if immediate_linear_stop and desired_linear <= 0.0:
            self.current_linear = 0.0
        else:
            self.current_linear = rate_limit(
                self.current_linear, desired_linear, linear_limit, dt)
        self.current_angular = rate_limit(
            self.current_angular, desired_angular, angular_limit, dt)
        self.current_angular = clamp(self.current_angular, max_angular)

        command = Twist()
        command.linear.x = self.current_linear
        command.angular.z = self.current_angular
        self.publisher_.publish(command)

    def reset_avoidance_state(self, state='TRACK'):
        """Clear all state that belongs to a previous obstacle."""
        self.avoidance_state = state
        self.avoidance_active = state in ('TURN', 'BYPASS')
        self.turn_direction = 0.0
        self.obstacle_clear_since = None
        self.avoidance_started = None
        self.avoidance_hit_distance = math.inf
        self.turn_start_yaw = self.current_yaw

    def set_avoidance_state(self, state, reason):
        """Change avoidance state and keep the legacy active flag in sync."""
        if state != self.avoidance_state:
            self.avoidance_state = state
            self.avoidance_active = state in ('TURN', 'BYPASS')
            self.get_logger().info(f'회피 상태 -> {state}: {reason}')

    def advance_waypoints(self):
        """Advance all reached or safely passed waypoints in one control tick."""
        tolerance = float(self.get_parameter('goal_tolerance').value)
        while self.current_waypoint_index < len(self.path):
            goal_x, goal_y = self.path[self.current_waypoint_index]
            distance = math.hypot(
                goal_x - self.current_x,
                goal_y - self.current_y,
            )
            reached = distance <= tolerance
            if not reached and self.segment_start is not None:
                start_x, start_y = self.segment_start
                vector_x = goal_x - start_x
                vector_y = goal_y - start_y
                length_squared = vector_x * vector_x + vector_y * vector_y
                if length_squared > 1e-12:
                    progress = (
                        (self.current_x - start_x) * vector_x
                        + (self.current_y - start_y) * vector_y
                    ) / length_squared
                    cross_track = abs(
                        vector_x * (self.current_y - start_y)
                        - vector_y * (self.current_x - start_x)
                    ) / math.sqrt(length_squared)
                    reached = (
                        progress >= 1.0
                        and cross_track <= max(0.40, 2.0 * tolerance)
                    )
            if not reached:
                break

            self.get_logger().info(
                f'경유점 {self.current_waypoint_index + 1}/{len(self.path)} 도착: '
                f'({goal_x:.2f}, {goal_y:.2f})')
            self.segment_start = (goal_x, goal_y)
            self.current_waypoint_index += 1

        if self.current_waypoint_index >= len(self.path):
            if not self.completed:
                self.completed = True
                self.reset_avoidance_state()
                self.get_logger().info('모든 경유점 완료: 정지합니다.')
            return True
        return False

    def path_geometry(self):
        """Return current waypoint distance and real polyline look-ahead error."""
        goal_x, goal_y = self.path[self.current_waypoint_index]
        goal_distance = math.hypot(
            goal_x - self.current_x,
            goal_y - self.current_y,
        )
        lookahead = float(self.get_parameter('lookahead_distance').value)
        target_x, target_y = select_lookahead_point(
            self.current_x,
            self.current_y,
            self.segment_start,
            self.path,
            self.current_waypoint_index,
            lookahead,
        )
        delta_x = target_x - self.current_x
        delta_y = target_y - self.current_y
        target_distance = math.hypot(delta_x, delta_y)
        alpha = normalize_angle(
            math.atan2(delta_y, delta_x) - self.current_yaw)
        return goal_distance, target_distance, alpha

    def path_command(self, goal_distance, target_distance, alpha):
        """Calculate a smoothed Pure Pursuit command for the polyline target."""
        linear_velocity = float(self.get_parameter('linear_velocity').value)
        max_angular = float(
            self.get_parameter('max_angular_velocity').value)
        enter_angle = float(
            self.get_parameter('heading_rotate_enter').value)
        exit_angle = float(
            self.get_parameter('heading_rotate_exit').value)

        if self.heading_align_active:
            if abs(alpha) <= exit_angle:
                self.heading_align_active = False
        elif abs(alpha) >= enter_angle:
            self.heading_align_active = True

        if linear_velocity <= 1e-9:
            return 0.0, 0.0, True
        if self.heading_align_active:
            gain = float(self.get_parameter('heading_gain').value)
            return 0.0, clamp(gain * alpha, max_angular), True

        tolerance = float(self.get_parameter('goal_tolerance').value)
        acceleration = float(
            self.get_parameter('linear_acceleration_limit').value)
        stopping_speed = math.sqrt(
            2.0 * acceleration * max(0.0, goal_distance - tolerance))
        heading_scale = max(0.0, math.cos(alpha))
        desired_linear = min(linear_velocity, stopping_speed) * heading_scale
        if target_distance <= 1e-6:
            desired_angular = 0.0
        else:
            curvature = 2.0 * math.sin(alpha) / target_distance
            desired_angular = clamp(
                desired_linear * curvature,
                max_angular,
            )
        return desired_linear, desired_angular, False

    def avoidance_elapsed(self, now):
        if self.avoidance_started is None:
            return 0.0
        return max(
            0.0,
            (now - self.avoidance_started).nanoseconds / 1e9,
        )

    def start_avoidance(self, metrics, goal_distance, alpha, now):
        """Latch one safe turn direction when entering avoidance."""
        spin_radius = float(
            self.get_parameter('spin_clearance_radius').value)
        if metrics['radial'] <= spin_radius:
            self.reset_avoidance_state('BLOCKED')
            self.get_logger().error(
                f'제자리 선회 공간 부족 ({metrics["radial"]:.2f}m): 안전 정지합니다.')
            return False

        left_score = min(metrics['front_left'], metrics['left'])
        right_score = min(metrics['front_right'], metrics['right'])
        difference = left_score - right_score
        deadband = float(
            self.get_parameter('turn_direction_deadband').value)
        if difference > deadband:
            self.turn_direction = 1.0
        elif difference < -deadband:
            self.turn_direction = -1.0
        else:
            self.turn_direction = 1.0 if alpha >= 0.0 else -1.0

        self.avoidance_started = now
        self.avoidance_hit_distance = goal_distance
        self.turn_start_yaw = self.current_yaw
        self.obstacle_clear_since = None
        direction = 'LEFT' if self.turn_direction > 0.0 else 'RIGHT'
        self.set_avoidance_state(
            'TURN', f'전방 {metrics["front"]:.2f}m, {direction} 방향 고정')
        return True

    def turn_command(self, metrics, now, dt):
        """Turn in the latched direction until the forward corridor is clear."""
        if metrics['radial'] <= float(
                self.get_parameter('spin_clearance_radius').value):
            self.set_avoidance_state('BLOCKED', '선회 반경 내 장애물')
            self.publish_stop()
            return
        if self.avoidance_elapsed(now) > float(
                self.get_parameter('avoidance_max_duration').value):
            self.set_avoidance_state('BLOCKED', '회피 제한 시간 초과')
            self.publish_stop()
            return

        clear_distance = (
            float(self.get_parameter('obs_threshold').value)
            + float(self.get_parameter('avoidance_clear_margin').value)
        )
        turned = abs(normalize_angle(
            self.current_yaw - self.turn_start_yaw))
        minimum_turn = float(
            self.get_parameter('minimum_turn_angle').value)
        if metrics['front'] >= clear_distance and turned >= minimum_turn:
            if self.obstacle_clear_since is None:
                self.obstacle_clear_since = self.last_scan_time
            clear_duration = (
                self.last_scan_time - self.obstacle_clear_since
            ).nanoseconds / 1e9
            if clear_duration >= float(
                    self.get_parameter('avoidance_clear_hold_time').value):
                self.obstacle_clear_since = None
                self.set_avoidance_state('BYPASS', '전방 통로 연속 확인')
                self.bypass_command(metrics, now, dt)
                return
        else:
            self.obstacle_clear_since = None

        angular = float(
            self.get_parameter('avoidance_angular_velocity').value)
        self.publish_motion(
            0.0,
            self.turn_direction * angular,
            dt,
            immediate_linear_stop=True,
        )

    def bypass_command(self, metrics, now, dt):
        """Move past the obstacle while maintaining side clearance."""
        if self.avoidance_elapsed(now) > float(
                self.get_parameter('avoidance_max_duration').value):
            self.set_avoidance_state('BLOCKED', '우회 제한 시간 초과')
            self.publish_stop()
            return

        stop_distance = float(self.get_parameter('obs_threshold').value)
        if metrics['front'] <= stop_distance:
            self.turn_start_yaw = self.current_yaw
            self.obstacle_clear_since = None
            self.set_avoidance_state('TURN', '우회 중 전방 재차단')
            self.turn_command(metrics, now, dt)
            return

        side_distance = (
            metrics['right'] if self.turn_direction > 0.0
            else metrics['left']
        )
        target_clearance = float(
            self.get_parameter('side_clearance').value)
        bounded_side = min(side_distance, 2.0 * target_clearance)
        gain = float(self.get_parameter('wall_follow_gain').value)
        angular = (
            -self.turn_direction
            * gain
            * (bounded_side - target_clearance)
        )
        angular = clamp(
            angular,
            float(self.get_parameter('max_angular_velocity').value),
        )
        linear = float(
            self.get_parameter('avoidance_linear_velocity').value)
        self.publish_motion(linear, angular, dt)

    def can_rejoin(self, metrics, goal_distance, alpha, now):
        """Return true only after measurable bypass progress and a clear goal ray."""
        if self.avoidance_elapsed(now) < float(
                self.get_parameter('avoidance_min_duration').value):
            return False
        progress = self.avoidance_hit_distance - goal_distance
        if progress < float(
                self.get_parameter('avoidance_progress_distance').value):
            return False
        if abs(alpha) > float(
                self.get_parameter('rejoin_heading_tolerance').value):
            return False
        clear_distance = (
            float(self.get_parameter('obs_threshold').value)
            + float(self.get_parameter('avoidance_clear_margin').value)
        )
        return (
            metrics['front'] > clear_distance
            and metrics['goal_count'] >= int(
                self.get_parameter('scan_min_hits').value)
            and metrics['goal'] > clear_distance
        )

    def control_loop(self):
        """Validate state, then publish one safe command for this control tick."""
        now = self.get_clock().now()
        dt = (now - self.last_control_time).nanoseconds / 1e9
        period = float(self.get_parameter('control_period').value)
        if not math.isfinite(dt) or dt <= 0.0 or dt > 1.0:
            dt = period
        self.last_control_time = now

        enabled = bool(self.get_parameter('drive_enabled').value)
        if enabled != self.last_drive_enabled:
            if enabled:
                self.get_logger().info('자동주행 시작: drive_enabled=true')
            else:
                self.reset_avoidance_state()
                self.heading_align_active = False
                self.publish_stop('자동주행 잠금: drive_enabled=false')
            self.last_drive_enabled = enabled
        if not enabled:
            return

        if not self.is_localized or self.last_pose_time is None:
            self.publish_stop()
            self.get_logger().warn(
                'AMCL 위치를 기다리는 중입니다.', throttle_duration_sec=2.0)
            return
        pose_age = (now - self.last_pose_time).nanoseconds / 1e9
        if (
                not math.isfinite(pose_age)
                or pose_age < 0.0
                or pose_age > float(self.get_parameter('pose_timeout').value)):
            self.reset_avoidance_state('WAIT_POSE')
            self.publish_stop()
            self.get_logger().error(
                f'AMCL 위치가 {pose_age:.1f}s 동안 갱신되지 않아 정지합니다.',
                throttle_duration_sec=2.0)
            return

        if self.segment_start is None:
            self.segment_start = (self.current_x, self.current_y)
            self.ever_started = True
        if self.advance_waypoints():
            self.publish_stop()
            return

        goal_distance, target_distance, alpha = self.path_geometry()

        if self.latest_scan is None or self.last_scan_time is None:
            self.reset_avoidance_state('WAIT_SCAN')
            self.publish_stop()
            self.get_logger().warn(
                'LiDAR 데이터를 기다리는 중입니다.', throttle_duration_sec=2.0)
            return
        scan_age = (now - self.last_scan_time).nanoseconds / 1e9
        if (
                not math.isfinite(scan_age)
                or scan_age < 0.0
                or scan_age > float(self.get_parameter('scan_timeout').value)):
            self.reset_avoidance_state('WAIT_SCAN')
            self.publish_stop()
            self.get_logger().error(
                f'LiDAR가 {scan_age:.1f}s 동안 갱신되지 않아 정지합니다.',
                throttle_duration_sec=2.0)
            return

        metrics = self.scan_metrics(alpha)
        if metrics is None:
            self.reset_avoidance_state('WAIT_SCAN')
            self.publish_stop()
            self.get_logger().error(
                '유효한 LiDAR 표본이 부족해 정지합니다.',
                throttle_duration_sec=2.0)
            return
        self.front_dist = metrics['front']
        self.left_dist = metrics['left']
        self.right_dist = metrics['right']
        if self.avoidance_state in ('WAIT_SCAN', 'WAIT_POSE'):
            self.reset_avoidance_state()
            self.get_logger().info('센서 입력 회복: TRACK 재개')

        if self.avoidance_state == 'BLOCKED':
            self.publish_stop()
            return
        if self.avoidance_state == 'TURN':
            self.turn_command(metrics, now, dt)
            return
        if self.avoidance_state == 'BYPASS':
            if self.can_rejoin(metrics, goal_distance, alpha, now):
                self.reset_avoidance_state()
                self.get_logger().info('우회 진전 확인: TRACK 복귀')
            else:
                self.bypass_command(metrics, now, dt)
                return

        if metrics['front'] <= float(
                self.get_parameter('obs_threshold').value):
            if self.start_avoidance(
                    metrics, goal_distance, alpha, now):
                self.turn_command(metrics, now, dt)
            else:
                self.publish_stop()
            return

        desired_linear, desired_angular, rotate_only = self.path_command(
            goal_distance, target_distance, alpha)
        self.publish_motion(
            desired_linear,
            desired_angular,
            dt,
            immediate_linear_stop=rotate_only,
        )
        self.get_logger().info(
            f'WP {self.current_waypoint_index + 1}/{len(self.path)} '
            f'd={goal_distance:.2f}m alpha={alpha:.2f}rad '
            f'v={self.current_linear:.2f}m/s w={self.current_angular:.2f}rad/s '
            f'state={self.avoidance_state}',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitLogic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_stop()
        except (KeyboardInterrupt, RuntimeError):
            pass
        try:
            node.destroy_node()
        except (KeyboardInterrupt, RuntimeError):
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass


if __name__ == '__main__':
    main()

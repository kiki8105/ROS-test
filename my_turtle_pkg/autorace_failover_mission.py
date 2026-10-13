"""GPS outbound 임무와 GPS 장애 시 breadcrumb 자동 복귀 상태기계."""

import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float32, String


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_yaw(quaternion):
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z))


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def polyline_lengths(points):
    lengths = [0.0]
    for start, end in zip(points, points[1:]):
        lengths.append(lengths[-1] + distance(start, end))
    return lengths


def route_progress(points, cumulative, waypoint_index, position):
    """현재 segment 투영을 사용한 0~1 거리 진행률."""
    if len(points) < 2 or cumulative[-1] <= 0.0:
        return 0.0
    segment = min(max(waypoint_index, 0), len(points) - 2)
    start, end = points[segment], points[segment + 1]
    vx, vy = end[0] - start[0], end[1] - start[1]
    norm = vx * vx + vy * vy
    if norm <= 1e-12:
        fraction = 1.0
    else:
        fraction = max(0.0, min(
            1.0,
            ((position[0] - start[0]) * vx + (position[1] - start[1]) * vy) / norm))
    travelled = cumulative[segment] + fraction * math.sqrt(norm)
    return max(0.0, min(1.0, travelled / cumulative[-1]))


class AutoraceFailoverMission(Node):
    TERMINAL_STATES = ('HOME_HOLD', 'OUTBOUND_COMPLETE', 'ABORT')

    def __init__(self):
        super().__init__('autorace_failover_mission')
        defaults = {
            'mission_enabled': False,
            'simulate_imu_failure': False,
            'simulate_vision_failure': False,
            'simulate_slam_failure': False,
            'simulate_odom_failure': False,
            'pose_topic': '/hybrid_pose',
            'odom_topic': '/odom',
            'imu_topic': '/imu',
            'scan_topic': '/scan',
            'cmd_vel_topic': '/cmd_vel/mission',
            'linear_speed': 0.10,
            'return_speed': 0.07,
            'max_angular_speed': 0.55,
            'heading_gain': 1.5,
            'vision_gain': 0.25,
            'goal_tolerance': 0.14,
            'breadcrumb_tolerance': 0.13,
            'home_tolerance': 0.20,
            'breadcrumb_spacing': 0.10,
            'gps_loss_hold': 0.5,
            'pose_timeout': 0.7,
            'odom_timeout': 0.5,
            'imu_timeout': 0.5,
            'vision_timeout': 0.7,
            'slam_timeout': 0.7,
            'scan_timeout': 0.7,
            'brake_min_duration': 0.8,
            'brake_hold_time': 0.4,
            'brake_speed_threshold': 0.02,
            'fallback_acquire_timeout': 5.0,
            'front_emergency_distance': 0.16,
            'front_stop_distance': 0.30,
            'front_slow_distance': 0.60,
            'avoid_turn_speed': 0.38,
            'control_period': 0.10,
            'waypoints_x': [1.307],
            'waypoints_y': [-1.754],
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        xs = [float(value) for value in self.get_parameter('waypoints_x').value]
        ys = [float(value) for value in self.get_parameter('waypoints_y').value]
        if not xs or len(xs) != len(ys):
            raise ValueError('waypoints_x/y는 길이가 같은 비어 있지 않은 배열이어야 합니다.')
        self.waypoints = list(zip(xs, ys))

        self.state = 'IDLE'
        self.state_started = self.now_seconds()
        self.last_enabled = False
        self.failure_latched = False
        self.gps_bad_since = None
        self.gps_fault_mode = 'unknown'
        self.position_source = 'NO_POSITION'
        self.pose = None
        self.pose_time = None
        self.odom = None
        self.odom_speed = 0.0
        self.odom_time = None
        self.imu_yaw = None
        self.imu_time = None
        self.vision_valid = False
        self.vision_error = 0.0
        self.vision_time = None
        self.slam_healthy = False
        self.slam_time = None
        self.scan_time = None
        self.front_distance = math.inf
        self.left_clearance = math.inf
        self.right_clearance = math.inf
        self.home_map = None
        self.home_odom = None
        self.route_points = []
        self.route_cumulative = []
        self.waypoint_index = 0
        self.progress = 0.0
        self.breadcrumbs = []
        self.return_index = -1
        self.brake_stopped_since = None

        topic = lambda name: str(self.get_parameter(name).value)
        self.cmd_pub = self.create_publisher(Twist, topic('cmd_vel_topic'), 10)
        self.status_pub = self.create_publisher(String, '/mission/status', 10)
        self.sensor_pub = self.create_publisher(String, '/mission/sensor_status', 10)
        self.progress_pub = self.create_publisher(Float32, '/mission/progress', 10)
        self.create_subscription(
            PoseWithCovarianceStamped, topic('pose_topic'), self.on_pose, 10)
        self.create_subscription(
            Odometry, topic('odom_topic'), self.on_odom, qos_profile_sensor_data)
        self.create_subscription(
            Imu, topic('imu_topic'), self.on_imu, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, topic('scan_topic'), self.on_scan, qos_profile_sensor_data)
        self.create_subscription(
            Bool, '/vision/line_valid', self.on_vision_valid, 10)
        self.create_subscription(
            Float32, '/vision/lane_error', self.on_vision_error, 10)
        self.create_subscription(
            Bool, '/slam/healthy', self.on_slam, 10)
        self.create_subscription(
            String, '/navigation/position_source', self.on_position_source, 10)
        self.create_subscription(
            String, '/gps/fault_mode', self.on_gps_fault_mode, 10)
        self.create_timer(
            float(self.get_parameter('control_period').value), self.control)
        self.get_logger().info('Autorace GPS failover mission 준비: mission_enabled=false')

    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_pose(self, msg):
        pose = msg.pose.pose
        self.pose = (float(pose.position.x), float(pose.position.y),
                     quaternion_yaw(pose.orientation))
        self.pose_time = self.now_seconds()

    def on_odom(self, msg):
        pose = msg.pose.pose
        self.odom = (float(pose.position.x), float(pose.position.y),
                     quaternion_yaw(pose.orientation))
        self.odom_speed = max(
            abs(float(msg.twist.twist.linear.x)),
            0.15 * abs(float(msg.twist.twist.angular.z)))
        self.odom_time = self.now_seconds()

    def on_imu(self, msg):
        self.imu_yaw = quaternion_yaw(msg.orientation)
        self.imu_time = self.now_seconds()

    def on_scan(self, msg):
        front, left, right = [], [], []
        for index, value in enumerate(msg.ranges):
            if not math.isfinite(value) or value < msg.range_min or value > msg.range_max:
                continue
            angle = normalize_angle(msg.angle_min + index * msg.angle_increment)
            if abs(angle) <= math.radians(25.0):
                front.append(value)
            elif math.radians(25.0) < angle <= math.radians(100.0):
                left.append(value)
            elif -math.radians(100.0) <= angle < -math.radians(25.0):
                right.append(value)
        self.front_distance = min(front) if front else math.inf
        self.left_clearance = sum(left) / len(left) if left else math.inf
        self.right_clearance = sum(right) / len(right) if right else math.inf
        self.scan_time = self.now_seconds()

    def on_vision_valid(self, msg):
        self.vision_valid = bool(msg.data)
        self.vision_time = self.now_seconds()

    def on_vision_error(self, msg):
        self.vision_error = float(msg.data)

    def on_slam(self, msg):
        self.slam_healthy = bool(msg.data)
        self.slam_time = self.now_seconds()

    def on_position_source(self, msg):
        self.position_source = msg.data

    def on_gps_fault_mode(self, msg):
        self.gps_fault_mode = msg.data

    def fresh(self, stamp, timeout):
        return stamp is not None and 0.0 <= self.now_seconds() - stamp <= timeout

    def health(self):
        gps = (not self.failure_latched and self.gps_fault_mode == 'normal' and
               self.position_source.startswith('GPS_PRIMARY') and
               self.fresh(self.pose_time, float(self.get_parameter('pose_timeout').value)))
        odom = (not bool(self.get_parameter('simulate_odom_failure').value) and
                self.fresh(self.odom_time, float(self.get_parameter('odom_timeout').value)))
        imu = (not bool(self.get_parameter('simulate_imu_failure').value) and
               self.fresh(self.imu_time, float(self.get_parameter('imu_timeout').value)))
        vision = (not bool(self.get_parameter('simulate_vision_failure').value) and
                  self.vision_valid and self.fresh(
                      self.vision_time, float(self.get_parameter('vision_timeout').value)))
        slam = (not bool(self.get_parameter('simulate_slam_failure').value) and
                self.slam_healthy and self.fresh(
                    self.slam_time, float(self.get_parameter('slam_timeout').value)))
        scan = self.fresh(self.scan_time, float(self.get_parameter('scan_timeout').value))
        return gps, odom, imu, vision, slam, scan

    def transition(self, state, reason):
        if state == self.state:
            return
        self.state = state
        self.state_started = self.now_seconds()
        self.brake_stopped_since = None
        self.get_logger().warn(f'임무 상태 -> {state}: {reason}')

    def stop(self):
        self.cmd_pub.publish(Twist())

    def reset_for_start(self):
        self.failure_latched = False
        self.gps_bad_since = None
        self.home_map = None
        self.home_odom = None
        self.route_points = []
        self.route_cumulative = []
        self.waypoint_index = 0
        self.progress = 0.0
        self.breadcrumbs = []
        self.return_index = -1
        self.transition('ARMING', '센서 준비 및 출발점 기록')

    def latch_home(self):
        self.home_map = self.pose
        self.home_odom = self.odom
        self.route_points = [(self.pose[0], self.pose[1])] + self.waypoints
        self.route_cumulative = polyline_lengths(self.route_points)
        self.breadcrumbs = [{
            'map': (self.pose[0], self.pose[1]),
            'odom': (self.odom[0], self.odom[1]),
        }]

    def record_breadcrumb(self):
        if self.pose is None or self.odom is None or not self.breadcrumbs:
            return
        current = (self.odom[0], self.odom[1])
        if distance(current, self.breadcrumbs[-1]['odom']) >= float(
                self.get_parameter('breadcrumb_spacing').value):
            self.breadcrumbs.append({
                'map': (self.pose[0], self.pose[1]),
                'odom': current,
            })

    def command_to(self, position, yaw, target, speed, use_vision=False):
        dx, dy = target[0] - position[0], target[1] - position[1]
        dist = math.hypot(dx, dy)
        alpha = normalize_angle(math.atan2(dy, dx) - yaw)
        max_w = float(self.get_parameter('max_angular_speed').value)
        angular = float(self.get_parameter('heading_gain').value) * alpha
        if use_vision:
            angular -= float(self.get_parameter('vision_gain').value) * self.vision_error
        angular = max(-max_w, min(max_w, angular))
        linear = speed * max(0.15, 1.0 - min(abs(alpha), 1.0))
        if abs(alpha) > 1.0:
            linear = 0.0
        return linear, angular, dist

    def apply_obstacle_safety(self, linear, angular, scan_ok):
        if not scan_ok:
            return 0.0, 0.0, 'scan_stale'
        emergency = float(self.get_parameter('front_emergency_distance').value)
        stop = float(self.get_parameter('front_stop_distance').value)
        slow = float(self.get_parameter('front_slow_distance').value)
        if self.front_distance <= emergency:
            return 0.0, 0.0, 'emergency_obstacle'
        if self.front_distance <= stop:
            direction = 1.0 if self.left_clearance >= self.right_clearance else -1.0
            return 0.0, direction * float(
                self.get_parameter('avoid_turn_speed').value), 'avoid_turn'
        if self.front_distance < slow:
            scale = max(0.15, (self.front_distance - stop) / max(slow - stop, 1e-3))
            return linear * scale, angular, 'obstacle_slow'
        return linear, angular, 'clear'

    def publish_command(self, linear, angular):
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)

    def publish_report(self, health, detail=''):
        gps, odom, imu, vision, slam, scan = health
        self.progress_pub.publish(Float32(data=float(self.progress)))
        self.sensor_pub.publish(String(data=(
            f'gps={gps}({self.gps_fault_mode}) odom={odom} imu={imu} '
            f'vision={vision} slam={slam} scan={scan} '
            f'source={self.position_source}')))
        home_distance = math.nan
        if self.home_odom is not None and self.odom is not None:
            home_distance = distance(self.home_odom[:2], self.odom[:2])
        self.status_pub.publish(String(data=(
            f'state={self.state} progress={self.progress:.3f} '
            f'wp={self.waypoint_index}/{len(self.waypoints)} '
            f'home_distance={home_distance:.2f}m breadcrumbs={len(self.breadcrumbs)} '
            f'{detail}')))

    def control(self):
        enabled = bool(self.get_parameter('mission_enabled').value)
        if enabled and not self.last_enabled:
            self.reset_for_start()
        elif not enabled and self.last_enabled:
            self.transition('IDLE', 'mission_enabled=false')
            self.stop()
        self.last_enabled = enabled
        health = self.health()
        gps_ok, odom_ok, imu_ok, vision_ok, slam_ok, scan_ok = health
        detail = ''

        if not enabled:
            self.stop()
            self.publish_report(health, '명령 대기')
            return

        if self.state == 'ARMING':
            self.stop()
            if gps_ok and odom_ok and self.pose is not None and self.odom is not None and scan_ok:
                self.latch_home()
                self.transition('OUTBOUND_GPS', 'GPS 정상, 출발점과 odom 기록 완료')
            else:
                detail = 'GPS/odom/pose/scan 준비 대기'

        elif self.state == 'OUTBOUND_GPS':
            if self.gps_fault_mode != 'normal':
                self.failure_latched = True
                self.transition('FAILSAFE_BRAKE', f'수동 GPS 고장 주입: {self.gps_fault_mode}')
                self.stop()
            elif not gps_ok:
                if self.gps_bad_since is None:
                    self.gps_bad_since = self.now_seconds()
                elif self.now_seconds() - self.gps_bad_since >= float(
                        self.get_parameter('gps_loss_hold').value):
                    self.failure_latched = True
                    self.transition('FAILSAFE_BRAKE', 'GPS timeout/status/covariance 이상')
                    self.stop()
            else:
                self.gps_bad_since = None
                self.record_breadcrumb()
                if self.waypoint_index >= len(self.waypoints):
                    self.progress = 1.0
                    self.transition('OUTBOUND_COMPLETE', '전체 GPS 웨이포인트 완료')
                    self.stop()
                else:
                    target = self.waypoints[self.waypoint_index]
                    linear, angular, dist = self.command_to(
                        self.pose[:2], self.pose[2], target,
                        float(self.get_parameter('linear_speed').value),
                        use_vision=vision_ok)
                    if dist <= float(self.get_parameter('goal_tolerance').value):
                        self.waypoint_index += 1
                    segment_index = min(self.waypoint_index, len(self.route_points) - 2)
                    self.progress = route_progress(
                        self.route_points, self.route_cumulative,
                        segment_index, self.pose[:2])
                    linear, angular, detail = self.apply_obstacle_safety(
                        linear, angular, scan_ok)
                    self.publish_command(linear, angular)

        elif self.state == 'FAILSAFE_BRAKE':
            self.stop()
            elapsed = self.now_seconds() - self.state_started
            if odom_ok and self.odom_speed <= float(
                    self.get_parameter('brake_speed_threshold').value):
                if self.brake_stopped_since is None:
                    self.brake_stopped_since = self.now_seconds()
            else:
                self.brake_stopped_since = None
            held = (self.brake_stopped_since is not None and
                    self.now_seconds() - self.brake_stopped_since >= float(
                        self.get_parameter('brake_hold_time').value))
            if elapsed >= float(self.get_parameter('brake_min_duration').value) and held:
                self.transition('FALLBACK_ACQUIRE', '차량 정지 확인, 대체 센서 확인')
            detail = f'비상제동 elapsed={elapsed:.1f}s'

        elif self.state == 'FALLBACK_ACQUIRE':
            self.stop()
            # 병진 위치는 odom/SLAM, 방향은 IMU/SLAM, 차선은 vision으로 보조한다.
            fallback_ready = odom_ok and (imu_ok or slam_ok) and (vision_ok or slam_ok)
            if fallback_ready and len(self.breadcrumbs) >= 2:
                self.return_index = len(self.breadcrumbs) - 2
                self.transition('RETURN_HOME', 'IMU/vision/SLAM/odom 대체 운용 확보')
            elif self.now_seconds() - self.state_started > float(
                    self.get_parameter('fallback_acquire_timeout').value):
                self.transition('ABORT', '대체 센서 조합을 확보하지 못함')
            detail = '필수: odom && (imu||slam) && (vision||slam)'

        elif self.state == 'RETURN_HOME':
            if not odom_ok:
                self.transition('ABORT', '복귀 중 odom 유실')
                self.stop()
            elif not (imu_ok or slam_ok) or not (vision_ok or slam_ok):
                self.transition('ABORT', '복귀 중 방향/차선/SLAM 조합 유실')
                self.stop()
            elif self.return_index < 0:
                self.transition('HOME_HOLD', 'breadcrumb 역추종 완료')
                self.stop()
            else:
                breadcrumb = self.breadcrumbs[self.return_index]
                if slam_ok and self.pose is not None and self.fresh(
                        self.pose_time, float(self.get_parameter('pose_timeout').value)):
                    position, yaw, target, source = (
                        self.pose[:2], self.pose[2], breadcrumb['map'], 'slam_map')
                else:
                    position, yaw, target, source = (
                        self.odom[:2], self.odom[2], breadcrumb['odom'], 'odom')
                linear, angular, dist = self.command_to(
                    position, yaw, target,
                    float(self.get_parameter('return_speed').value),
                    use_vision=vision_ok)
                tolerance_name = (
                    'home_tolerance' if self.return_index == 0
                    else 'breadcrumb_tolerance')
                if dist <= float(self.get_parameter(tolerance_name).value):
                    if self.return_index == 0:
                        self.return_index = -1
                        self.transition('HOME_HOLD', '출발점 허용 반경 진입')
                        self.stop()
                        detail = f'home 도착 control={source}'
                    else:
                        self.return_index -= 1
                if self.state == 'RETURN_HOME':
                    linear, angular, obstacle = self.apply_obstacle_safety(
                        linear, angular, scan_ok)
                    self.publish_command(linear, angular)
                    detail = (f'return={self.return_index + 1}/{len(self.breadcrumbs)} '
                              f'control={source} obstacle={obstacle}')

        elif self.state in self.TERMINAL_STATES:
            self.stop()
            detail = 'mission_enabled=false 후 true로 재시작'

        self.publish_report(health, detail)


def main(args=None):
    rclpy.init(args=args)
    node = AutoraceFailoverMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

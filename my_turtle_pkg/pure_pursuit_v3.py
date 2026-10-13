"""Pure Pursuit v3: waypoint tracking with reactive 360-degree LiDAR avoidance.

This is an educational local controller. It keeps the v2 Pure Pursuit command
when the route is clear, and temporarily switches to a TURN/BYPASS state
machine when a LaserScan obstacle occupies the Waffle Pi's forward corridor.
"""
import math
from collections import deque

import rclpy
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import (
    IntegerRange,
    ParameterDescriptor,
    SetParametersResult,
)
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from my_turtle_pkg.pure_pursuit import (
    PurePursuit,
    floating_descriptor,
    normalize_angle,
)


# turtlebot3_waffle_pi.urdf 기준 base_footprint 좌표 [m].
# LaserScan 원점은 회전 중심보다 뒤에 있고 차체 앞은 회전 중심보다 앞에 있다.
WAFFLE_PI_LIDAR_X = -0.064
WAFFLE_PI_FRONT_X = 0.080


def integer_descriptor(description, minimum, maximum, step=1):
    """Create an integer ROS parameter descriptor used by the controller UI."""
    return ParameterDescriptor(
        description=description,
        integer_range=[IntegerRange(
            from_value=int(minimum), to_value=int(maximum), step=int(step))])


def scan_sector_distance(scan, center, half_width, percentile=0.0):
    """Return a near-distance percentile inside one angular scan sector.

    Positive infinity is a valid no-return and is treated as range_max. NaN and
    values below range_min are ignored. The percentile lets the caller reject
    one-ray noise while still reacting to a real object.
    """
    if scan is None or not scan.ranges or scan.angle_increment == 0.0:
        return 0.0
    maximum = (
        float(scan.range_max)
        if math.isfinite(scan.range_max) and scan.range_max > 0.0
        else 10.0
    )
    minimum = max(0.0, float(scan.range_min))
    values = []
    for index, raw in enumerate(scan.ranges):
        angle = scan.angle_min + index * scan.angle_increment
        if abs(normalize_angle(angle - center)) > half_width:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(value) or value < minimum:
            continue
        if math.isinf(value):
            if value < 0.0:
                continue
            value = maximum
        elif value > maximum:
            value = maximum
        values.append(value)
    if not values:
        return maximum
    values.sort()
    percentile = max(0.0, min(1.0, float(percentile)))
    return values[int((len(values) - 1) * percentile)]


def scan_points(scan, lidar_x=WAFFLE_PI_LIDAR_X):
    """Convert finite scan returns to base_footprint x/y points.

    The second return value counts usable samples, including positive infinity
    (a valid open-space reading). Therefore an all-Inf scan is open, whereas an
    all-NaN scan is invalid and must stop the robot.
    """
    if scan is None or not scan.ranges or scan.angle_increment == 0.0:
        return [], 0
    minimum = max(0.0, float(scan.range_min))
    maximum = (
        float(scan.range_max)
        if math.isfinite(scan.range_max) and scan.range_max > minimum
        else math.inf
    )
    points = []
    usable = 0
    for index, raw in enumerate(scan.ranges):
        try:
            distance = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(distance) or distance < minimum:
            continue
        if math.isinf(distance) or distance > maximum:
            if distance > 0.0:
                usable += 1
            continue
        usable += 1
        angle = scan.angle_min + index * scan.angle_increment
        points.append((
            lidar_x + distance * math.cos(angle),
            distance * math.sin(angle),
        ))
    return points, usable


def nth_smallest(values, count):
    """Return the count-th smallest value, or infinity if hits are insufficient."""
    ordered = sorted(values)
    return ordered[count - 1] if len(ordered) >= count else math.inf


def front_body_clearance(
        scan, half_width=0.18, min_hits=3,
        lidar_x=WAFFLE_PI_LIDAR_X, body_front_x=WAFFLE_PI_FRONT_X):
    """Free distance in the rectangular corridor directly ahead of the body.

    Door posts and corridor walls outside +/- half_width do not block the
    robot. Requiring several returns rejects a single noisy ray.
    """
    points, _ = scan_points(scan, lidar_x)
    gaps = [
        x - body_front_x
        for x, y in points
        if x > body_front_x and abs(y) < half_width
    ]
    return nth_smallest(gaps, max(1, int(min_hits)))


def minimum_radial_clearance(scan, lidar_x=WAFFLE_PI_LIDAR_X):
    """Return the closest obstacle radius around the base rotation centre."""
    points, _ = scan_points(scan, lidar_x)
    if not points:
        return math.inf
    return min(math.hypot(x, y) for x, y in points)


def choose_turn_direction(left_clearance, right_clearance, previous=1.0):
    """Choose +1 for left or -1 for right, retaining direction on near ties."""
    if abs(left_clearance - right_clearance) < 0.05:
        return 1.0 if previous >= 0.0 else -1.0
    return 1.0 if left_clearance > right_clearance else -1.0


def quaternion_yaw(quaternion):
    """Return planar yaw from a geometry_msgs quaternion."""
    return math.atan2(
        2.0 * (
            quaternion.w * quaternion.z
            + quaternion.x * quaternion.y),
        1.0 - 2.0 * (
            quaternion.y * quaternion.y
            + quaternion.z * quaternion.z))


def stamp_nanoseconds(stamp):
    """Return a positive ROS message timestamp, or None when it is unset."""
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else None


def propagate_odom_pose(map_anchor, odom_anchor, odom_current):
    """Apply an odom-frame SE(2) delta to an AMCL map-frame anchor."""
    map_x, map_y, map_yaw = map_anchor
    odom_x, odom_y, odom_yaw = odom_anchor
    current_x, current_y, current_yaw = odom_current

    delta_x = current_x - odom_x
    delta_y = current_y - odom_y
    frame_yaw = normalize_angle(map_yaw - odom_yaw)
    cosine = math.cos(frame_yaw)
    sine = math.sin(frame_yaw)
    return (
        map_x + cosine * delta_x - sine * delta_y,
        map_y + sine * delta_x + cosine * delta_y,
        normalize_angle(
            map_yaw + normalize_angle(current_yaw - odom_yaw)),
    )


class PurePursuitV3(PurePursuit):
    """Track map waypoints and make a local detour around LiDAR obstacles."""

    obstacle_avoidance_enabled = True
    immediate_stop_on_completion = True
    # V3 has a continuous odom heartbeat, so toggling drive must never make an
    # old localization sample look fresh.
    refresh_pose_timeout_on_enable = False

    def __init__(self):
        super().__init__('pure_pursuit_v3_node')

        self.declare_parameter(
            'global_frame', 'map',
            ParameterDescriptor(
                description=(
                    '조종기 /goal_pose가 사용해야 하는 전역 좌표 프레임입니다. '
                    '다른 프레임의 목표는 좌표 변환 없이 사용할 수 없으므로 거부합니다.'),
                read_only=True))
        global_frame = str(
            self.get_parameter('global_frame').value).strip().lstrip('/')
        if not global_frame:
            raise ValueError('global_frame은 비어 있지 않아야 합니다.')

        self.latest_odom_pose = None
        self.map_anchor_pose = None
        self.odom_anchor_pose = None
        self.last_odom_time = None
        self.last_amcl_time = None
        self.latest_odom_stamp_ns = None
        self.odom_history = deque(maxlen=200)

        self.declare_parameter(
            'scan_topic', '/scan',
            ParameterDescriptor(
                description='장애물 회피에 사용할 LaserScan 토픽입니다. 변경 후 재시작합니다.',
                read_only=True))
        self.declare_parameter(
            'odom_topic', '/odom',
            ParameterDescriptor(
                description=(
                    'AMCL 갱신 사이의 상대이동을 받을 Odometry 토픽입니다. '
                    '변경 후 재시작합니다.'),
                read_only=True))
        self.declare_parameter(
            'odom_anchor_max_age', 0.20,
            floating_descriptor(
                'AMCL 시각과 odom 기준 표본 사이의 최대 허용 시간 [s].',
                0.02, 2.0, 0.01))
        self.declare_parameter(
            'odom_reanchor_timeout', 1.00,
            floating_descriptor(
                'odom 수신이 이 시간 이상 끊긴 뒤 재개되면 기준점을 다시 잡습니다 [s].',
                0.10, 10.0, 0.10))
        self.declare_parameter(
            'odom_jump_distance', 0.50,
            floating_descriptor(
                '한 odom 표본 사이에서 리셋으로 판정할 위치 점프 [m].',
                0.10, 5.0, 0.05))
        self.declare_parameter(
            'odom_jump_angle', 1.57,
            floating_descriptor(
                '한 odom 표본 사이에서 리셋으로 판정할 방향 점프 [rad].',
                0.20, 3.15, 0.01))
        self.declare_parameter(
            'scan_timeout', 0.60,
            floating_descriptor('LiDAR 미수신 시 정지할 시간 [s].', 0.1, 10.0, 0.1))
        self.declare_parameter(
            'emergency_stop_distance', 0.12,
            floating_descriptor(
                '차체 앞 충돌 여유가 이 값 이하면 즉시 정지합니다 [m].',
                0.05, 1.0, 0.01))
        self.declare_parameter(
            'front_stop_distance', 0.30,
            floating_descriptor(
                '차체 앞 통로가 이 값 이하면 회피 선회를 시작합니다 [m].',
                0.10, 2.0, 0.01))
        self.declare_parameter(
            'front_clear_distance', 0.45,
            floating_descriptor(
                '선회 후 전방이 다시 열렸다고 판단하는 거리 [m].',
                0.15, 3.0, 0.01))
        self.declare_parameter(
            'front_slow_distance', 0.60,
            floating_descriptor(
                '차체 앞 통로가 이 값보다 좁으면 접근 속도를 줄입니다 [m].',
                0.20, 4.0, 0.05))
        self.declare_parameter(
            'front_corridor_half_width', 0.18,
            floating_descriptor(
                '전방 장애물 검사 통로의 중심선 기준 반폭 [m].',
                0.10, 0.50, 0.01))
        self.declare_parameter(
            'scan_min_hits', 3,
            integer_descriptor(
                '장애물로 인정할 LiDAR 최소 검출 수입니다.', 1, 20))
        self.declare_parameter(
            'spin_clearance_radius', 0.27,
            floating_descriptor(
                '제자리 선회가 안전한지 검사할 차체 중심 반경 [m].',
                0.18, 0.60, 0.01))
        self.declare_parameter(
            'side_clearance', 0.32,
            floating_descriptor(
                '장애물 측면을 따라갈 LiDAR 기준 목표 간격 [m].',
                0.15, 1.5, 0.01))
        self.declare_parameter(
            'avoidance_linear_velocity', 0.06,
            floating_descriptor(
                '장애물 우회 중 전진 속도 [m/s].', 0.01, 0.20, 0.01))
        self.declare_parameter(
            'avoidance_angular_velocity', 0.45,
            floating_descriptor(
                '막힌 전방에서 사용하는 회전 속도 [rad/s].',
                0.10, 1.5, 0.05))
        self.declare_parameter(
            'minimum_turn_angle', 0.52,
            floating_descriptor(
                '우회 전진 전 반드시 선회할 최소 각도 [rad].',
                0.10, 1.57, 0.01))
        self.declare_parameter(
            'clear_hold_time', 0.40,
            floating_descriptor(
                '전방이 연속으로 열려 있어야 하는 시간 [s].',
                0.10, 3.0, 0.10))
        self.declare_parameter(
            'wall_follow_gain', 1.20,
            floating_descriptor(
                '측면 간격 오차에 대한 회전 보정 게인입니다.',
                0.1, 8.0, 0.1))
        self.declare_parameter(
            'obstacle_bias_gain', 0.60,
            floating_descriptor(
                '정상 주행 중 가까운 장애물 반대쪽 보정 게인입니다.',
                0.0, 3.0, 0.1))
        self.declare_parameter(
            'avoidance_min_duration', 1.50,
            floating_descriptor(
                '경로 재진입 전 유지할 최소 회피 시간 [s].',
                0.2, 10.0, 0.1))
        self.declare_parameter(
            'avoidance_max_duration', 20.0,
            floating_descriptor(
                '진전이 없을 때 회피 복구를 시작하는 시간 [s].',
                3.0, 120.0, 1.0))
        self.declare_parameter(
            'avoidance_retry_limit', 1,
            integer_descriptor(
                '시간 초과 후 반대 방향 회피를 시도할 횟수입니다.', 0, 5))
        self.declare_parameter(
            'rejoin_heading_tolerance', 0.55,
            floating_descriptor(
                '목표 경로로 복귀할 수 있는 방향 오차 [rad].',
                0.1, 1.5, 0.05))
        self.declare_parameter(
            'avoidance_progress_distance', 0.12,
            floating_descriptor(
                '경로 재진입 전 목표에 더 가까워져야 하는 거리 [m].',
                0.0, 1.0, 0.01))

        self.add_on_set_parameters_callback(self.validate_avoidance_parameters)
        validation = self.validate_avoidance_parameters([])
        if not validation.successful:
            raise ValueError(
                f'Pure Pursuit v3 파라미터 오류: {validation.reason}')
        self.latest_scan = None
        self.last_scan_time = None
        self.avoidance_state = 'TRACK'
        self.turn_direction = 1.0
        self.avoidance_started = None
        self.avoidance_hit_distance = math.inf
        self.turn_start_yaw = 0.0
        self.clear_since = None
        self.avoidance_retries = 0
        self.v3_drive_enabled = None

        scan_topic = str(self.get_parameter('scan_topic').value)
        odom_topic = str(self.get_parameter('odom_topic').value)
        self.scan_subscription = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, qos_profile_sensor_data)
        self.odom_subscription = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, qos_profile_sensor_data)
        self.get_logger().info(
            f'Pure Pursuit v3 장애물 회피 준비: scan={scan_topic}, '
            f'odom={odom_topic}, '
            'drive_enabled=true 전까지 정지')

    def goal_callback(self, msg):
        """Accept a new goal only while stopped and in the configured frame."""
        expected_frame = str(
            self.get_parameter('global_frame').value).strip().lstrip('/')
        received_frame = str(msg.header.frame_id).strip().lstrip('/')

        if not received_frame or received_frame != expected_frame:
            self.get_logger().warn(
                '목표 프레임이 올바르지 않아 /goal_pose를 무시합니다: '
                f'received={received_frame or "(없음)"}, '
                f'expected={expected_frame}')
            return

        if bool(self.get_parameter('drive_enabled').value):
            self.get_logger().warn(
                '주행 중에는 목표를 변경할 수 없습니다. '
                'drive_enabled=false로 정지한 뒤 다시 전송하세요.')
            return

        super().goal_callback(msg)

    def odom_anchor_for_amcl(self, amcl_stamp_ns, now):
        """Find the odom sample closest to an AMCL measurement timestamp."""
        maximum_age = float(
            self.get_parameter('odom_anchor_max_age').value)
        if amcl_stamp_ns is not None and self.odom_history:
            stamp_ns, pose = min(
                self.odom_history,
                key=lambda sample: abs(sample[0] - amcl_stamp_ns),
            )
            if abs(stamp_ns - amcl_stamp_ns) <= maximum_age * 1e9:
                return pose
        if amcl_stamp_ns is not None:
            return None

        # Some test or custom publishers leave header.stamp unset. Only use
        # the latest odom in that case when it was received very recently.
        if self.latest_odom_pose is None or self.last_odom_time is None:
            return None
        receipt_age = (now - self.last_odom_time).nanoseconds / 1e9
        if 0.0 <= receipt_age <= maximum_age:
            return self.latest_odom_pose
        return None

    def pose_callback(self, msg):
        """Use AMCL as the absolute map anchor and reset odom drift."""
        super().pose_callback(msg)
        now = self.get_clock().now()
        amcl_pose = (
            self.current_x,
            self.current_y,
            self.current_yaw,
        )
        self.map_anchor_pose = amcl_pose
        self.odom_anchor_pose = self.odom_anchor_for_amcl(
            stamp_nanoseconds(msg.header.stamp), now)
        if self.odom_anchor_pose is not None and self.latest_odom_pose is not None:
            # AMCL may describe a slightly older scan. Bring that absolute
            # correction forward to the latest buffered odom immediately.
            self.current_x, self.current_y, self.current_yaw = (
                propagate_odom_pose(
                    self.map_anchor_pose,
                    self.odom_anchor_pose,
                    self.latest_odom_pose,
                )
            )
        self.last_amcl_time = now
        self.last_pose_time = now

    def odom_callback(self, msg):
        """Propagate the map pose continuously using odometry deltas."""
        position = msg.pose.pose.position
        odom_pose = (
            float(position.x),
            float(position.y),
            quaternion_yaw(msg.pose.pose.orientation),
        )
        if not all(math.isfinite(value) for value in odom_pose):
            self.get_logger().warn(
                '유효하지 않은 /odom pose를 무시합니다.',
                throttle_duration_sec=2.0)
            return

        now = self.get_clock().now()
        stamp_ns = stamp_nanoseconds(msg.header.stamp)
        discontinuity_reasons = []
        receipt_age = None
        if self.last_odom_time is not None:
            receipt_age = (now - self.last_odom_time).nanoseconds / 1e9
            reanchor_timeout = float(
                self.get_parameter('odom_reanchor_timeout').value)
            if receipt_age < 0.0:
                discontinuity_reasons.append('시뮬레이션 시간 역행')
            elif receipt_age > reanchor_timeout:
                discontinuity_reasons.append(
                    f'odom {receipt_age:.1f}s 수신 중단')

        if (
                stamp_ns is not None
                and self.latest_odom_stamp_ns is not None
                and stamp_ns < self.latest_odom_stamp_ns):
            discontinuity_reasons.append('odom 타임스탬프 역행')

        if self.latest_odom_pose is not None:
            delta_distance = math.hypot(
                odom_pose[0] - self.latest_odom_pose[0],
                odom_pose[1] - self.latest_odom_pose[1],
            )
            delta_angle = abs(normalize_angle(
                odom_pose[2] - self.latest_odom_pose[2]))
            if delta_distance > float(
                    self.get_parameter('odom_jump_distance').value):
                discontinuity_reasons.append(
                    f'odom 위치 {delta_distance:.2f}m 점프')
            if delta_angle > float(
                    self.get_parameter('odom_jump_angle').value):
                discontinuity_reasons.append(
                    f'odom 방향 {delta_angle:.2f}rad 점프')

        if discontinuity_reasons:
            self.odom_history.clear()
        if stamp_ns is not None:
            self.odom_history.append((stamp_ns, odom_pose))
        self.latest_odom_pose = odom_pose
        self.latest_odom_stamp_ns = stamp_ns
        self.last_odom_time = now

        if not self.is_localized or self.map_anchor_pose is None:
            return

        if discontinuity_reasons:
            # Preserve the current map pose and establish a new odom origin.
            # This prevents a Gazebo reset or driver restart from teleporting
            # the estimated map pose.
            self.map_anchor_pose = (
                self.current_x,
                self.current_y,
                self.current_yaw,
            )
            self.odom_anchor_pose = odom_pose
            self.last_pose_time = now
            self.get_logger().warn(
                'odom 기준점 재설정: ' + ', '.join(discontinuity_reasons),
                throttle_duration_sec=1.0)
            return

        if self.odom_anchor_pose is None:
            # AMCL이 odom보다 먼저 들어오거나 동기화 가능한 odom 표본이
            # 없었던 경우, 첫 새 odom을 같은 시점의 기준으로 잡는다.
            self.odom_anchor_pose = odom_pose
            self.map_anchor_pose = (
                self.current_x,
                self.current_y,
                self.current_yaw,
            )
        else:
            self.current_x, self.current_y, self.current_yaw = (
                propagate_odom_pose(
                    self.map_anchor_pose,
                    self.odom_anchor_pose,
                    odom_pose,
                )
            )
        self.last_pose_time = self.last_odom_time

    def validate_avoidance_parameters(self, parameters):
        positive = {
            'scan_timeout', 'emergency_stop_distance', 'front_stop_distance',
            'front_clear_distance', 'front_slow_distance',
            'front_corridor_half_width', 'spin_clearance_radius',
            'side_clearance', 'avoidance_linear_velocity',
            'avoidance_angular_velocity', 'minimum_turn_angle',
            'clear_hold_time', 'wall_follow_gain', 'avoidance_min_duration',
            'avoidance_max_duration', 'rejoin_heading_tolerance',
            'odom_anchor_max_age', 'odom_reanchor_timeout',
            'odom_jump_distance', 'odom_jump_angle',
        }
        values = {
            name: float(self.get_parameter(name).value)
            for name in positive
        }
        for name in ('obstacle_bias_gain', 'avoidance_progress_distance'):
            values[name] = float(self.get_parameter(name).value)
        values['scan_min_hits'] = int(
            self.get_parameter('scan_min_hits').value)
        values['avoidance_retry_limit'] = int(
            self.get_parameter('avoidance_retry_limit').value)
        values['max_angular_velocity'] = float(
            self.get_parameter('max_angular_velocity').value)
        for parameter in parameters:
            if parameter.name in values:
                values[parameter.name] = parameter.value

        if any(float(values[name]) <= 0.0 for name in positive):
            return SetParametersResult(
                successful=False,
                reason='장애물 회피 거리/속도/시간 파라미터는 0보다 커야 합니다.')
        if (float(values['obstacle_bias_gain']) < 0.0
                or float(values['avoidance_progress_distance']) < 0.0):
            return SetParametersResult(
                successful=False,
                reason='회피 게인과 진행 거리는 0 이상이어야 합니다.')
        if (int(values['scan_min_hits']) < 1
                or int(values['avoidance_retry_limit']) < 0):
            return SetParametersResult(
                successful=False,
                reason='scan_min_hits는 1 이상, retry_limit은 0 이상이어야 합니다.')
        if not (
                float(values['emergency_stop_distance'])
                < float(values['front_stop_distance'])
                < float(values['front_clear_distance'])
                < float(values['front_slow_distance'])):
            return SetParametersResult(
                successful=False,
                reason=(
                    'emergency_stop_distance < front_stop_distance < '
                    'front_clear_distance < front_slow_distance여야 합니다.'))
        if (float(values['avoidance_min_duration'])
                >= float(values['avoidance_max_duration'])):
            return SetParametersResult(
                successful=False,
                reason=(
                    'avoidance_min_duration은 '
                    'avoidance_max_duration보다 작아야 합니다.'))
        if (float(values['avoidance_angular_velocity'])
                > float(values['max_angular_velocity'])):
            return SetParametersResult(
                successful=False,
                reason=(
                    'avoidance_angular_velocity는 '
                    'max_angular_velocity 이하여야 합니다.'))
        return SetParametersResult(successful=True)

    def scan_callback(self, msg):
        self.latest_scan = msg
        self.last_scan_time = self.get_clock().now()

    def reset_avoidance(self):
        self.avoidance_state = 'TRACK'
        self.avoidance_started = None
        self.avoidance_hit_distance = math.inf
        self.turn_start_yaw = self.current_yaw
        self.clear_since = None
        self.avoidance_retries = 0

    def control_loop(self):
        # BLOCKED는 안전 정지 상태다. false -> true 토글 시 새 회피 시도로 초기화한다.
        enabled = bool(self.get_parameter('drive_enabled').value)
        if enabled != self.v3_drive_enabled:
            self.reset_avoidance()
            self.v3_drive_enabled = enabled
        super().control_loop()

    def set_avoidance_state(self, state, reason):
        if state != self.avoidance_state:
            self.avoidance_state = state
            self.get_logger().info(f'회피 상태 -> {state}: {reason}')

    def hard_stop(self):
        """Bypass comfort rate limits for a sensor/safety stop."""
        self.current_linear = 0.0
        self.current_angular = 0.0
        return 0.0, 0.0

    def scan_distances(self, alpha):
        scan = self.latest_scan
        half_width = float(
            self.get_parameter('front_corridor_half_width').value)
        min_hits = int(self.get_parameter('scan_min_hits').value)
        return {
            'front': front_body_clearance(scan, half_width, min_hits),
            'front_left': scan_sector_distance(
                scan, math.radians(45), math.radians(25), 0.15),
            'front_right': scan_sector_distance(
                scan, math.radians(-45), math.radians(25), 0.15),
            'left': scan_sector_distance(
                scan, math.pi / 2.0, math.radians(30), 0.15),
            'right': scan_sector_distance(
                scan, -math.pi / 2.0, math.radians(30), 0.15),
            'goal': scan_sector_distance(
                scan, alpha, math.radians(12), 0.05),
            'radial': minimum_radial_clearance(scan),
        }

    def start_avoidance(self, distances, goal_distance, alpha, now, reason):
        spin_radius = float(
            self.get_parameter('spin_clearance_radius').value)
        if distances['radial'] <= spin_radius:
            self.set_avoidance_state(
                'BLOCKED',
                f'{reason}, 제자리 선회 반경 {distances["radial"]:.2f}m 부족')
            self.get_logger().error(
                '차체 둘레 공간이 부족해 안전 정지합니다. 장애물을 옮긴 뒤 '
                'drive_enabled를 false→true로 다시 켜세요.')
            return False

        left_score = min(distances['front_left'], distances['left'])
        right_score = min(distances['front_right'], distances['right'])
        if abs(left_score - right_score) < 0.05 and abs(alpha) > 0.10:
            self.turn_direction = 1.0 if alpha > 0.0 else -1.0
        else:
            self.turn_direction = choose_turn_direction(
                left_score, right_score, self.turn_direction)
        self.avoidance_started = now
        self.avoidance_hit_distance = goal_distance
        self.turn_start_yaw = self.current_yaw
        self.clear_since = None
        direction = '좌측' if self.turn_direction > 0.0 else '우측'
        self.set_avoidance_state('TURN', f'{reason}, {direction} 회피')
        return True

    def avoidance_elapsed(self, now):
        if self.avoidance_started is None:
            return 0.0
        return max(0.0, (now - self.avoidance_started).nanoseconds / 1e9)

    @staticmethod
    def elapsed_since(then, now):
        if then is None:
            return 0.0
        return max(0.0, (now - then).nanoseconds / 1e9)

    @staticmethod
    def clamp(value, limit):
        return max(-limit, min(limit, value))

    def scan_is_usable(self):
        _, usable = scan_points(self.latest_scan)
        minimum = max(3, int(self.get_parameter('scan_min_hits').value))
        return usable >= minimum

    def begin_turn_again(self, now, reason):
        self.turn_start_yaw = self.current_yaw
        self.clear_since = None
        self.set_avoidance_state('TURN', reason)

    def adjust_command_for_environment(
            self, desired_linear, desired_angular, distance, alpha,
            goal_x, goal_y, now):
        del goal_x, goal_y
        timeout = float(self.get_parameter('scan_timeout').value)
        if self.last_scan_time is None:
            self.set_avoidance_state('WAIT_SCAN', 'LiDAR 첫 데이터 대기')
            return self.hard_stop()

        scan_age = (now - self.last_scan_time).nanoseconds / 1e9
        if scan_age < 0.0 or scan_age > timeout:
            self.set_avoidance_state(
                'WAIT_SCAN', f'LiDAR {scan_age:.1f}s 지연')
            self.get_logger().warn(
                f'LiDAR가 {scan_age:.1f}s 동안 갱신되지 않아 즉시 정지합니다.',
                throttle_duration_sec=2.0)
            return self.hard_stop()
        if not self.scan_is_usable():
            self.set_avoidance_state('WAIT_SCAN', '유효한 LiDAR 표본 부족')
            self.get_logger().warn(
                'LaserScan 값이 유효하지 않아 즉시 정지합니다.',
                throttle_duration_sec=2.0)
            return self.hard_stop()
        if self.avoidance_state == 'WAIT_SCAN':
            self.reset_avoidance()
            self.get_logger().info('LiDAR 수신 회복: TRACK 재개')

        if self.avoidance_state == 'BLOCKED':
            return self.hard_stop()

        distances = self.scan_distances(alpha)
        emergency = float(
            self.get_parameter('emergency_stop_distance').value)
        configured_stop = float(
            self.get_parameter('front_stop_distance').value)
        clear_distance = float(
            self.get_parameter('front_clear_distance').value)
        configured_slow = float(
            self.get_parameter('front_slow_distance').value)
        clearance = float(self.get_parameter('side_clearance').value)
        avoid_linear = float(
            self.get_parameter('avoidance_linear_velocity').value)
        avoid_angular = float(
            self.get_parameter('avoidance_angular_velocity').value)
        max_angular = float(
            self.get_parameter('max_angular_velocity').value)
        linear_accel = float(
            self.get_parameter('linear_acceleration_limit').value)

        # 현재 속도에서 멈출 수 있는 거리(v²/2a)에 0.3초 반응 거리와
        # 최소 차체 여유를 더한다. 속도를 높여도 stop 값이 자동 보강된다.
        braking_distance = (
            self.current_linear * self.current_linear
            / max(0.02, 2.0 * linear_accel)
        )
        dynamic_stop = (
            emergency + braking_distance + abs(self.current_linear) * 0.30
        )
        stop_distance = max(configured_stop, dynamic_stop)
        slow_distance = max(configured_slow, stop_distance + 0.15)
        elapsed = self.avoidance_elapsed(now)
        spin_radius = float(
            self.get_parameter('spin_clearance_radius').value)

        if distances['front'] <= emergency:
            if distances['radial'] <= spin_radius:
                self.set_avoidance_state(
                    'BLOCKED', '긴급 장애물로 제자리 선회 공간 부족')
                self.get_logger().error(
                    '충돌 임박 장애물 때문에 회전도 안전하지 않아 즉시 정지합니다.')
                return self.hard_stop()
            if self.avoidance_state == 'TRACK':
                self.start_avoidance(
                    distances, distance, alpha, now,
                    f'긴급 장애물 {distances["front"]:.2f}m')
            # 긴급 상황에서는 선형 가속도 제한을 우회해 전진을 즉시 끊는다.
            self.current_linear = 0.0
            if self.avoidance_state == 'BLOCKED':
                return self.hard_stop()
            return 0.0, self.turn_direction * avoid_angular

        if self.avoidance_state == 'TRACK':
            # 목표가 옆/뒤라 v2가 제자리 정렬을 요구할 때도 360도 차체
            # 회전 반경을 먼저 검사한다.
            if (abs(desired_linear) < 1e-6
                    and abs(desired_angular) > 1e-6
                    and distances['radial'] <= spin_radius):
                self.set_avoidance_state(
                    'BLOCKED', '목표 방향 정렬을 위한 선회 공간 부족')
                return self.hard_stop()
            if distances['front'] <= stop_distance:
                started = self.start_avoidance(
                    distances, distance, alpha, now,
                    f'전방 장애물 {distances["front"]:.2f}m')
                self.current_linear = 0.0
                if not started:
                    return self.hard_stop()
                return 0.0, self.turn_direction * avoid_angular

            speed_scale = 1.0
            if distances['front'] < slow_distance:
                span = max(0.01, slow_distance - stop_distance)
                speed_scale = max(
                    0.20,
                    min(
                        1.0,
                        (distances['front'] - stop_distance) / span,
                    ),
                )
            left_pressure = max(
                0.0,
                (slow_distance - distances['front_left']) / slow_distance,
            )
            right_pressure = max(
                0.0,
                (slow_distance - distances['front_right']) / slow_distance,
            )
            bias = float(
                self.get_parameter('obstacle_bias_gain').value)
            desired_angular += bias * (right_pressure - left_pressure)
            return (
                desired_linear * speed_scale,
                self.clamp(desired_angular, max_angular),
            )

        max_duration = float(
            self.get_parameter('avoidance_max_duration').value)
        if elapsed > max_duration:
            retry_limit = int(
                self.get_parameter('avoidance_retry_limit').value)
            if self.avoidance_retries >= retry_limit:
                self.set_avoidance_state(
                    'BLOCKED', '회피 제한 시간과 재시도 횟수 초과')
                self.get_logger().error(
                    '장애물을 우회하지 못해 안전 정지합니다. 경로를 확인한 뒤 '
                    'drive_enabled를 false→true로 다시 켜세요.')
                return self.hard_stop()
            if distances['radial'] <= float(
                    self.get_parameter('spin_clearance_radius').value):
                self.set_avoidance_state('BLOCKED', '반대 선회 공간 부족')
                return self.hard_stop()
            self.avoidance_retries += 1
            self.turn_direction *= -1.0
            self.avoidance_started = now
            self.avoidance_hit_distance = distance
            self.begin_turn_again(
                now, '회피 진전 없음, 반대 방향 재시도')
            self.get_logger().warn(
                f'회피 제한 시간 초과: 반대 방향 재시도 '
                f'{self.avoidance_retries}/{retry_limit}')
            self.current_linear = 0.0
            return 0.0, self.turn_direction * avoid_angular

        if self.avoidance_state == 'TURN':
            if distances['radial'] <= spin_radius:
                self.set_avoidance_state(
                    'BLOCKED', '선회 중 차체 회전 반경에 장애물 진입')
                return self.hard_stop()
            turn_side = (
                distances['front_left']
                if self.turn_direction > 0.0
                else distances['front_right']
            )
            turned = abs(normalize_angle(
                self.current_yaw - self.turn_start_yaw))
            minimum_turn = float(
                self.get_parameter('minimum_turn_angle').value)
            clear_now = (
                turned >= minimum_turn
                and distances['front'] > clear_distance
                and turn_side > clearance
            )
            if clear_now:
                if self.clear_since is None:
                    self.clear_since = now
                hold = float(
                    self.get_parameter('clear_hold_time').value)
                if self.elapsed_since(self.clear_since, now) >= hold:
                    self.clear_since = None
                    self.set_avoidance_state(
                        'BYPASS',
                        '최소 선회 완료 및 전방 공간 연속 확인')
            else:
                self.clear_since = None
            self.current_linear = 0.0
            return 0.0, self.turn_direction * avoid_angular

        # BYPASS: 선택한 방향을 고정하고 반대편 장애물 측면을 따라간다.
        min_duration = float(
            self.get_parameter('avoidance_min_duration').value)
        progress = float(
            self.get_parameter('avoidance_progress_distance').value)
        heading_tolerance = float(
            self.get_parameter('rejoin_heading_tolerance').value)
        progressed = distance <= self.avoidance_hit_distance - progress
        goal_is_clear = distances['goal'] > clear_distance
        if (elapsed >= min_duration and progressed and goal_is_clear
                and abs(alpha) <= heading_tolerance):
            self.set_avoidance_state(
                'TRACK', '목표 방향이 다시 열리고 접근 거리 확보')
            self.avoidance_started = None
            self.avoidance_retries = 0
            return desired_linear, desired_angular

        if distances['front'] <= stop_distance:
            if distances['radial'] <= float(
                    self.get_parameter('spin_clearance_radius').value):
                self.set_avoidance_state(
                    'BLOCKED', '우회 중 선회 공간 부족')
                return self.hard_stop()
            self.current_linear = 0.0
            self.begin_turn_again(
                now, '측면 추종 중 전방 재차단')
            return 0.0, self.turn_direction * avoid_angular

        if self.turn_direction > 0.0:
            wall_distance = min(
                distances['front_right'], distances['right'])
        else:
            wall_distance = min(
                distances['front_left'], distances['left'])

        # 선회 직후 장애물이 아직 대각선에 있어도 추종한다. 대각선과
        # 측면이 모두 열렸다면 존재하지 않는 벽 쪽으로 포화 선회하지 않는다.
        wall_visible_distance = max(clearance * 2.5, configured_slow)
        if wall_distance < wall_visible_distance:
            wall_error = clearance - wall_distance
            wall_gain = float(
                self.get_parameter('wall_follow_gain').value)
            wall_turn = self.clamp(
                self.turn_direction * wall_gain * wall_error,
                avoid_angular * 0.65)
            goal_pull = 0.25 * alpha
        else:
            wall_turn = 0.0
            goal_pull = 0.50 * alpha
        angular = self.clamp(
            wall_turn + goal_pull, avoid_angular)
        span = max(0.01, slow_distance - stop_distance)
        speed_scale = max(
            0.20,
            min(1.0, (distances['front'] - stop_distance) / span),
        )
        return avoid_linear * speed_scale, angular


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitV3()
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

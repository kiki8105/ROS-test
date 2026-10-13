"""GPS 우선, AMCL/odom/IMU 대체 위치를 내보내는 교육용 위치원 선택기."""

import math

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, NavSatFix, NavSatStatus
from std_msgs.msg import String


EARTH_RADIUS_M = 6_378_137.0


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def yaw_quaternion(yaw):
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


def gps_to_local(latitude, longitude, datum_latitude, datum_longitude):
    """작은 시험장 범위에서 WGS84 위경도를 ENU East/North 미터로 변환한다."""
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    lat0 = math.radians(datum_latitude)
    lon0 = math.radians(datum_longitude)
    east = EARTH_RADIUS_M * (lon - lon0) * math.cos(lat0)
    north = EARTH_RADIUS_M * (lat - lat0)
    return east, north


def transform_local(east, north, offset_x, offset_y, yaw_offset):
    c = math.cos(yaw_offset)
    s = math.sin(yaw_offset)
    return offset_x + c * east - s * north, offset_y + s * east + c * north


def estimate_se2_transform(source_pose, target_pose):
    """source 좌표계 pose를 target 좌표계 pose로 옮기는 SE(2)를 계산한다."""
    yaw = normalize_angle(target_pose[2] - source_pose[2])
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotated_x = c * source_pose[0] - s * source_pose[1]
    rotated_y = s * source_pose[0] + c * source_pose[1]
    return (
        target_pose[0] - rotated_x,
        target_pose[1] - rotated_y,
        yaw,
    )


def apply_se2_transform(pose, transform):
    """SE(2) transform을 pose의 위치와 yaw에 적용한다."""
    tx, ty, yaw = transform
    c = math.cos(yaw)
    s = math.sin(yaw)
    return (
        tx + c * pose[0] - s * pose[1],
        ty + s * pose[0] + c * pose[1],
        normalize_angle(pose[2] + yaw),
    )


def slam_pose_in_map(slam_pose, slam_to_map=None):
    """정렬 정보가 있으면 SLAM pose를 GPS map으로 변환하고 없으면 그대로 쓴다."""
    pose = tuple(slam_pose[:3])
    if slam_to_map is None:
        return pose
    return apply_se2_transform(pose, slam_to_map)


def propagate_pose(anchor_pose, anchor_odom, current_odom, imu_delta=None):
    """마지막 절대 위치에서 odom 상대 이동을 map 좌표로 누적한다."""
    map_x, map_y, map_yaw = anchor_pose
    odom_x, odom_y, odom_yaw = anchor_odom
    current_x, current_y, current_yaw = current_odom
    frame_yaw = normalize_angle(map_yaw - odom_yaw)
    dx = current_x - odom_x
    dy = current_y - odom_y
    x = map_x + math.cos(frame_yaw) * dx - math.sin(frame_yaw) * dy
    y = map_y + math.sin(frame_yaw) * dx + math.cos(frame_yaw) * dy
    yaw = normalize_angle(
        map_yaw + (imu_delta if imu_delta is not None
                   else normalize_angle(current_yaw - odom_yaw)))
    return x, y, yaw


class HybridLocalizer(Node):
    def __init__(self):
        super().__init__('hybrid_localizer')
        defaults = {
            'fix_topic': '/fix',
            'imu_topic': '/imu',
            'odom_topic': '/odom',
            'amcl_topic': '/amcl_pose',
            'camera_topic': '/camera/image_raw',
            'output_topic': '/hybrid_pose',
            'frame_id': 'map',
            'datum_latitude': 37.5665,
            'datum_longitude': 126.9780,
            'map_offset_x': 0.0,
            'map_offset_y': 0.0,
            'map_yaw_offset': 0.0,
            'gps_timeout': 1.0,
            'imu_timeout': 0.5,
            'slam_timeout': 0.8,
            'max_gps_stddev_m': 1.0,
            'publish_rate': 20.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.fix = None
        self.fix_time = None
        self.imu_yaw = None
        self.imu_time = None
        self.odom = None
        self.amcl = None
        self.amcl_time = None
        self.camera_time = None
        self.anchor_pose = None
        self.anchor_odom = None
        self.anchor_imu_yaw = None
        self.slam_to_map = None
        self.fix_sequence = 0
        self.aligned_fix_sequence = -1
        self.last_fix_measurement_time = None
        self.last_source = ''

        topic = lambda name: str(self.get_parameter(name).value)
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, topic('output_topic'), 10)
        self.source_pub = self.create_publisher(String, '/navigation/position_source', 10)
        self.create_subscription(
            NavSatFix, topic('fix_topic'), self.fix_callback, qos_profile_sensor_data)
        self.create_subscription(
            Imu, topic('imu_topic'), self.imu_callback, qos_profile_sensor_data)
        self.create_subscription(
            Odometry, topic('odom_topic'), self.odom_callback, qos_profile_sensor_data)
        self.create_subscription(
            PoseWithCovarianceStamped, topic('amcl_topic'), self.amcl_callback, 10)
        self.create_subscription(
            Image, topic('camera_topic'), self.camera_callback, qos_profile_sensor_data)
        rate = max(float(self.get_parameter('publish_rate').value), 1.0)
        self.create_timer(1.0 / rate, self.publish_pose)
        self.get_logger().info(
            'Hybrid localization 준비: GPS 우선, GPS 실패 시 AMCL/odom/IMU 대체')

    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def fix_callback(self, msg):
        self.fix = msg
        stamp = msg.header.stamp
        measurement_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        # freeze 모드는 같은 timestamp의 측정치를 반복하므로 새 GPS 표본으로
        # 취급하지 않는다. timestamp가 없는 실제 드라이버는 수신마다 갱신한다.
        if (measurement_time <= 0.0 or
                measurement_time != self.last_fix_measurement_time):
            self.fix_sequence += 1
            self.last_fix_measurement_time = (
                measurement_time if measurement_time > 0.0 else None)
        self.fix_time = measurement_time if measurement_time > 0.0 else self.now_seconds()
        self.update_slam_alignment()

    def imu_callback(self, msg):
        values = (msg.orientation.x, msg.orientation.y,
                  msg.orientation.z, msg.orientation.w)
        if all(math.isfinite(value) for value in values):
            self.imu_yaw = quaternion_yaw(msg.orientation)
            self.imu_time = self.now_seconds()

    def odom_callback(self, msg):
        pose = msg.pose.pose
        values = (pose.position.x, pose.position.y,
                  pose.orientation.x, pose.orientation.y,
                  pose.orientation.z, pose.orientation.w)
        if all(math.isfinite(value) for value in values):
            self.odom = (
                float(pose.position.x), float(pose.position.y),
                quaternion_yaw(pose.orientation))

    def amcl_callback(self, msg):
        pose = msg.pose.pose
        self.amcl = (
            float(pose.position.x), float(pose.position.y),
            quaternion_yaw(pose.orientation), list(msg.pose.covariance))
        self.amcl_time = self.now_seconds()
        self.update_slam_alignment()

    def camera_callback(self, _msg):
        self.camera_time = self.now_seconds()

    def gps_healthy(self):
        if self.fix is None or self.fix_time is None:
            return False
        if self.now_seconds() - self.fix_time > float(self.get_parameter('gps_timeout').value):
            return False
        if self.fix.status.status < NavSatStatus.STATUS_FIX:
            return False
        if not math.isfinite(self.fix.latitude) or not math.isfinite(self.fix.longitude):
            return False
        variance = max(float(self.fix.position_covariance[0]),
                       float(self.fix.position_covariance[4]), 0.0)
        return math.sqrt(variance) <= float(
            self.get_parameter('max_gps_stddev_m').value)

    def imu_healthy(self):
        return (self.imu_time is not None and
                self.now_seconds() - self.imu_time <=
                float(self.get_parameter('imu_timeout').value))

    def slam_healthy(self):
        return (self.amcl is not None and self.amcl_time is not None and
                self.now_seconds() - self.amcl_time <=
                float(self.get_parameter('slam_timeout').value))

    def update_slam_alignment(self):
        """새 GPS 측정과 가장 최근 SLAM pose 사이의 map 정렬을 갱신한다."""
        if (self.amcl is None or self.fix_sequence == self.aligned_fix_sequence or
                not self.gps_healthy()):
            return
        self.slam_to_map = estimate_se2_transform(
            self.amcl[:3], self.gps_pose())
        self.aligned_fix_sequence = self.fix_sequence

    def set_anchor(self, pose):
        self.anchor_pose = tuple(pose)
        self.anchor_odom = self.odom
        self.anchor_imu_yaw = self.imu_yaw

    def gps_pose(self):
        east, north = gps_to_local(
            self.fix.latitude, self.fix.longitude,
            float(self.get_parameter('datum_latitude').value),
            float(self.get_parameter('datum_longitude').value))
        x, y = transform_local(
            east, north,
            float(self.get_parameter('map_offset_x').value),
            float(self.get_parameter('map_offset_y').value),
            float(self.get_parameter('map_yaw_offset').value))
        if self.imu_yaw is not None:
            yaw = normalize_angle(
                self.imu_yaw + float(self.get_parameter('map_yaw_offset').value))
        elif self.odom is not None:
            yaw = self.odom[2]
        else:
            yaw = 0.0
        return x, y, yaw

    def fallback_pose(self):
        if self.slam_healthy():
            pose = slam_pose_in_map(self.amcl[:3], self.slam_to_map)
            self.set_anchor(pose)
            return pose
        if self.anchor_pose is None:
            return None
        if self.odom is None or self.anchor_odom is None:
            return self.anchor_pose
        imu_delta = None
        if self.imu_healthy() and self.anchor_imu_yaw is not None:
            imu_delta = normalize_angle(self.imu_yaw - self.anchor_imu_yaw)
        return propagate_pose(
            self.anchor_pose, self.anchor_odom, self.odom, imu_delta)

    def publish_pose(self):
        if self.gps_healthy():
            source = 'GPS_PRIMARY'
            pose = self.gps_pose()
            self.update_slam_alignment()
            self.set_anchor(pose)
            variance = max(float(self.fix.position_covariance[0]), 0.0004)
        else:
            source = 'IMU_ODOM_SLAM_FALLBACK'
            pose = self.fallback_pose()
            variance = 0.25
        if pose is None:
            source = 'NO_POSITION'
            if source != self.last_source:
                self.get_logger().error('GPS와 대체 위치가 모두 없어 위치를 발행할 수 없습니다.')
                self.last_source = source
            return

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter('frame_id').value)
        msg.pose.pose.position.x = pose[0]
        msg.pose.pose.position.y = pose[1]
        qx, qy, qz, qw = yaw_quaternion(pose[2])
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance[0] = variance
        msg.pose.covariance[7] = variance
        msg.pose.covariance[35] = 0.05 if self.imu_healthy() else 0.25
        self.pose_pub.publish(msg)

        camera_ok = (self.camera_time is not None and
                     self.now_seconds() - self.camera_time < 1.0)
        status = f'{source};imu={self.imu_healthy()};vision={camera_ok}'
        self.source_pub.publish(String(data=status))
        if source != self.last_source:
            self.get_logger().warn(f'위치원 전환 -> {status}')
            self.last_source = source


def main(args=None):
    rclpy.init(args=args)
    node = HybridLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

import random

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from my_turtle_pkg.generate_lds02_dataset import (
    AVAILABLE_PATTERNS,
    generate_single_scan,
)


class LidarMock(Node):
    def __init__(self):
        super().__init__('lidar_mock')

        self.publisher = self.create_publisher(
            LaserScan,
            '/scan_mock',
            qos_profile_sensor_data,
        )

        # 2초마다 publish_scan() 실행
        self.timer = self.create_timer(2.0, self.publish_scan)

    def publish_scan(self):
        pattern = random.choice(AVAILABLE_PATTERNS)
        scan = generate_single_scan(pattern)

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_scan'

        msg.angle_min = scan['angle_min']
        msg.angle_max = scan['angle_max']
        msg.angle_increment = scan['angle_increment']

        msg.range_min = scan['range_min']
        msg.range_max = scan['range_max']

        # 모의 스캔은 한 번에 생성하며, 발행 간격은 2초
        msg.time_increment = 0.0
        msg.scan_time = 2.0

        msg.ranges = [float(v) for v in scan['ranges']]
        msg.intensities = [float(v) for v in scan['intensities']]

        self.publisher.publish(msg)
        self.get_logger().info(
            f'발행: {pattern}, 거리 데이터 {len(msg.ranges)}개'
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarMock()

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
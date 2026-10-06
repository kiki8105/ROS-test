import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanDrive(Node):
    def __init__(self):
        super().__init__('scan_drive')
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.declare_parameter('safe_distance', 1.0)
        self.declare_parameter('right_wall_min_clearance', 0.5)
        self.declare_parameter('left_wall_min_clearance', 0.5)
        self.declare_parameter('wall_avoidance_gain', 1.0)
        self.declare_parameter('max_wall_avoidance_turn', 0.35)

    @staticmethod
    def sector_distances(scan, center_deg, half_width_deg):
        center = math.radians(center_deg)
        half_width = math.radians(half_width_deg)
        distances = []

        for index, distance in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            difference = math.atan2(math.sin(angle - center), math.cos(angle - center))
            if abs(difference) <= half_width and math.isfinite(distance):
                if scan.range_min <= distance <= scan.range_max:
                    distances.append(distance)

        return distances or [scan.range_max]

    @classmethod
    def sector_mean(cls, scan, center_deg, half_width_deg):
        distances = cls.sector_distances(scan, center_deg, half_width_deg)
        return sum(distances) / len(distances)

    def scan_callback(self, scan):
        front = min(self.sector_distances(scan, 0, 10))
        left = self.sector_mean(scan, 90, 10)
        right = self.sector_mean(scan, -90, 10)
        left_nearest = min(self.sector_distances(scan, 90, 10))
        right_nearest = min(self.sector_distances(scan, -90, 10))
        safe_distance = self.get_parameter('safe_distance').value
        left_wall_min_clearance = self.get_parameter('left_wall_min_clearance').value
        right_wall_min_clearance = self.get_parameter('right_wall_min_clearance').value
        wall_avoidance_gain = self.get_parameter('wall_avoidance_gain').value
        max_wall_avoidance_turn = self.get_parameter('max_wall_avoidance_turn').value

        command = Twist()
        if front <= safe_distance:
            command.angular.z = 0.5 if left > right else -0.5
            action = 'turn_left' if left > right else 'turn_right'
        else:
            command.linear.x = 0.1
            left_too_close = left_nearest <= left_wall_min_clearance
            right_too_close = right_nearest <= right_wall_min_clearance

            if left_too_close and right_too_close:
                turn_left = left > right
                clearance_error = (
                    right_wall_min_clearance - right_nearest
                    if turn_left else left_wall_min_clearance - left_nearest
                )
            elif right_too_close:
                turn_left = True
                clearance_error = right_wall_min_clearance - right_nearest
            elif left_too_close:
                turn_left = False
                clearance_error = left_wall_min_clearance - left_nearest
            else:
                turn_left = None

            if turn_left is not None:
                turn_rate = min(
                    wall_avoidance_gain * clearance_error,
                    max_wall_avoidance_turn,
                )
                command.angular.z = turn_rate if turn_left else -turn_rate
                action = 'avoid_left_wall' if turn_left else 'avoid_right_wall'
            else:
                action = 'go_forward'

        self.cmd_pub.publish(command)
        self.get_logger().info(
            f'front={front:.2f} left={left:.2f} right={right:.2f} action={action}',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = ScanDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
"""임무·센서·odom 정보를 한 터미널에 1 Hz로 표시한다."""

import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float32, String


class MissionConsole(Node):
    def __init__(self):
        super().__init__('mission_console')
        self.status = '대기 중'
        self.sensors = '센서 대기 중'
        self.progress = 0.0
        self.odom = None
        self.last_print = 0.0
        self.create_subscription(String, '/mission/status', self.on_status, 10)
        self.create_subscription(String, '/mission/sensor_status', self.on_sensors, 10)
        self.create_subscription(Float32, '/mission/progress', self.on_progress, 10)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_timer(0.2, self.print_status)

    def on_status(self, msg):
        self.status = msg.data

    def on_sensors(self, msg):
        self.sensors = msg.data

    def on_progress(self, msg):
        self.progress = float(msg.data)

    def on_odom(self, msg):
        pose = msg.pose.pose
        twist = msg.twist.twist
        self.odom = (
            pose.position.x, pose.position.y,
            math.hypot(twist.linear.x, twist.linear.y), twist.angular.z)

    def print_status(self):
        now = time.monotonic()
        if now - self.last_print < 1.0:
            return
        self.last_print = now
        odom_text = 'odom=WAIT'
        if self.odom is not None:
            odom_text = (
                f'odom=({self.odom[0]:+.2f},{self.odom[1]:+.2f}) '
                f'v={self.odom[2]:.2f} w={self.odom[3]:+.2f}')
        print(
            f'\n[{self.progress * 100:5.1f}%] {self.status}\n'
            f'{self.sensors}\n{odom_text}', flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = MissionConsole()
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

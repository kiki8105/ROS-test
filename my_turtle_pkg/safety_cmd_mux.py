"""자동 임무와 수동 조종 중 하나만 최종 /cmd_vel로 전달한다."""

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import String


class SafetyCmdMux(Node):
    def __init__(self):
        super().__init__('safety_cmd_mux')
        defaults = {
            'mission_topic': '/cmd_vel/mission',
            'manual_topic': '/cmd_vel/manual',
            'output_topic': '/cmd_vel',
            'manual_override': False,
            'emergency_stop': False,
            'command_timeout': 0.5,
            'publish_rate': 20.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.mission_cmd = Twist()
        self.manual_cmd = Twist()
        self.mission_time = None
        self.manual_time = None
        self.last_source = ''
        topic = lambda name: str(self.get_parameter(name).value)
        self.pub = self.create_publisher(Twist, topic('output_topic'), 10)
        self.source_pub = self.create_publisher(String, '/navigation/drive_source', 10)
        self.create_subscription(Twist, topic('mission_topic'), self.on_mission, 10)
        self.create_subscription(Twist, topic('manual_topic'), self.on_manual, 10)
        rate = max(float(self.get_parameter('publish_rate').value), 1.0)
        self.create_timer(
            1.0 / rate, self.control,
            clock=Clock(clock_type=ClockType.STEADY_TIME))

    def on_mission(self, msg):
        self.mission_cmd = msg
        self.mission_time = time.monotonic()

    def on_manual(self, msg):
        self.manual_cmd = msg
        self.manual_time = time.monotonic()

    def control(self):
        now = time.monotonic()
        timeout = float(self.get_parameter('command_timeout').value)
        if bool(self.get_parameter('emergency_stop').value):
            source, command = 'emergency_stop', Twist()
        elif bool(self.get_parameter('manual_override').value):
            if self.manual_time is not None and now - self.manual_time <= timeout:
                source, command = 'manual', self.manual_cmd
            else:
                source, command = 'manual_timeout', Twist()
        elif self.mission_time is not None and now - self.mission_time <= timeout:
            source, command = 'mission', self.mission_cmd
        else:
            source, command = 'mission_timeout', Twist()
        self.pub.publish(command)
        self.source_pub.publish(String(data=source))
        if source != self.last_source:
            self.get_logger().warn(f'최종 속도 명령원 -> {source}')
            self.last_source = source


def main(args=None):
    rclpy.init(args=args)
    node = SafetyCmdMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

"""GPS 경로 명령과 비전 차선 명령 중 안전한 쪽만 차량에 전달한다."""

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from std_msgs.msg import Bool, String


def choose_source(operation_mode, gps_healthy, vision_valid, gps_ready, vision_ready):
    if operation_mode == 'indoor':
        return 'vision' if vision_valid and vision_ready else 'stop'
    if gps_healthy and gps_ready:
        return 'gps'
    return 'vision' if vision_valid and vision_ready else 'stop'


def approach(current, target, max_step):
    return current + max(-max_step, min(max_step, target - current))


class DriveModeManager(Node):
    def __init__(self):
        super().__init__('drive_mode_manager')
        defaults = {
            'operation_mode': 'outdoor',  # outdoor | indoor | auto
            'gps_cmd_topic': '/cmd_vel/gps',
            'vision_cmd_topic': '/cmd_vel/vision',
            'output_topic': '/cmd_vel',
            'source_topic': '/navigation/position_source',
            'vision_valid_topic': '/vision/line_valid',
            'command_timeout': 0.5,
            'gps_recovery_hold': 3.0,
            'max_linear_speed': 0.20,
            'max_angular_speed': 0.70,
            'max_linear_accel': 0.25,
            'max_angular_accel': 1.0,
            'publish_rate': 20.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.operation_mode = str(self.get_parameter('operation_mode').value)
        if self.operation_mode not in ('outdoor', 'indoor', 'auto'):
            raise ValueError('operation_mode은 outdoor, indoor, auto 중 하나여야 합니다.')

        self.gps_cmd = Twist()
        self.vision_cmd = Twist()
        self.gps_cmd_time = None
        self.vision_cmd_time = None
        self.gps_reported_healthy = False
        self.gps_healthy_since = None
        self.vision_valid = False
        self.current_source = ''
        self.out_v = 0.0
        self.out_w = 0.0
        self.last_tick = time.monotonic()

        topic = lambda name: str(self.get_parameter(name).value)
        self.pub = self.create_publisher(Twist, topic('output_topic'), 10)
        self.mode_pub = self.create_publisher(String, '/navigation/drive_source', 10)
        self.create_subscription(Twist, topic('gps_cmd_topic'), self.gps_cmd_callback, 10)
        self.create_subscription(Twist, topic('vision_cmd_topic'), self.vision_cmd_callback, 10)
        self.create_subscription(String, topic('source_topic'), self.source_callback, 10)
        self.create_subscription(Bool, topic('vision_valid_topic'), self.vision_callback, 10)
        rate = max(float(self.get_parameter('publish_rate').value), 1.0)
        self.create_timer(
            1.0 / rate, self.control,
            clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f'주행 선택기 준비: mode={self.operation_mode}, GPS 실패 시 vision 전환')

    def gps_cmd_callback(self, msg):
        self.gps_cmd = msg
        self.gps_cmd_time = time.monotonic()

    def vision_cmd_callback(self, msg):
        self.vision_cmd = msg
        self.vision_cmd_time = time.monotonic()

    def source_callback(self, msg):
        healthy = msg.data.startswith('GPS_PRIMARY')
        now = time.monotonic()
        if healthy and not self.gps_reported_healthy:
            self.gps_healthy_since = now
        if not healthy:
            self.gps_healthy_since = None
        self.gps_reported_healthy = healthy

    def vision_callback(self, msg):
        self.vision_valid = bool(msg.data)

    def gps_allowed(self, now):
        if not self.gps_reported_healthy or self.gps_healthy_since is None:
            return False
        return now - self.gps_healthy_since >= float(
            self.get_parameter('gps_recovery_hold').value)

    def control(self):
        now = time.monotonic()
        timeout = float(self.get_parameter('command_timeout').value)
        gps_ready = self.gps_cmd_time is not None and now - self.gps_cmd_time <= timeout
        vision_ready = (self.vision_cmd_time is not None and
                        now - self.vision_cmd_time <= timeout)
        self.operation_mode = str(self.get_parameter('operation_mode').value)
        source = choose_source(
            self.operation_mode, self.gps_allowed(now), self.vision_valid,
            gps_ready, vision_ready)
        target = self.gps_cmd if source == 'gps' else (
            self.vision_cmd if source == 'vision' else Twist())

        dt = min(0.1, max(0.001, now - self.last_tick))
        self.last_tick = now
        max_v = float(self.get_parameter('max_linear_speed').value)
        max_w = float(self.get_parameter('max_angular_speed').value)
        target_v = max(-max_v, min(max_v, float(target.linear.x)))
        target_w = max(-max_w, min(max_w, float(target.angular.z)))
        # 정지 판단은 즉시 적용하고, 정상적인 원천 전환은 가속도 제한을 적용한다.
        if source == 'stop':
            self.out_v = self.out_w = 0.0
        else:
            self.out_v = approach(
                self.out_v, target_v,
                float(self.get_parameter('max_linear_accel').value) * dt)
            self.out_w = approach(
                self.out_w, target_w,
                float(self.get_parameter('max_angular_accel').value) * dt)
        command = Twist()
        command.linear.x = self.out_v
        command.angular.z = self.out_w
        self.pub.publish(command)
        self.mode_pub.publish(String(data=source))
        if source != self.current_source:
            self.get_logger().warn(f'주행 명령원 전환 -> {source}')
            self.current_source = source


def main(args=None):
    rclpy.init(args=args)
    node = DriveModeManager()
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

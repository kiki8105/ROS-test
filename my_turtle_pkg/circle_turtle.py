#!/usr/bin/env python3
"""Publish a circular-motion command; send stop commands before normal exit."""
import signal
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist


class CircleTurtle(Node):
    def __init__(self):
        super().__init__('circle_turtle_node')
        self.declare_parameter('linear_speed', 0.2)   # m/s; 0.0이면 제자리 회전
        self.declare_parameter('angular_speed', 0.3)  # rad/s; 양수는 좌회전
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.get_logger().info('🐢 로봇 회전 노드가 시작되었습니다!')

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = float(self.get_parameter('linear_speed').value)
        msg.angular.z = float(self.get_parameter('angular_speed').value)
        self.publisher_.publish(msg)

    def stop(self):
        self.timer.cancel()
        # ROS context가 살아 있을 때 정지 명령을 반복 전송합니다.
        for _ in range(5):
            self.publisher_.publish(Twist())
            time.sleep(0.05)
        self.get_logger().info('정지 명령 전송 완료')


def main(args=None):
    # 기본 신호 처리기가 context를 먼저 종료하지 않도록 직접 처리합니다.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    stop_requested = threading.Event()
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda *_: stop_requested.set())
    node = None
    try:
        node = CircleTurtle()
        while rclpy.ok() and not stop_requested.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if rclpy.ok():
                node.stop()
            node.destroy_node()
        rclpy.try_shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    main()

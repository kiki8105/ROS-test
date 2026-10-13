"""빨간 선의 영상 위치로 조향하는 Gazebo 실습 노드. 지도/웨이포인트 불필요."""
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


def detect_line(frame, roi_top=0.52, roi_bottom=0.95, min_area=80):
    """가까운 빨간 연결 영역에서 근거리/원거리 중심을 측정한다."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 100, 60), (10, 255, 255))
    mask |= cv2.inRange(hsv, (170, 100, 60), (179, 255, 255))
    h, w = mask.shape
    top, bottom = int(h * roi_top), int(h * roi_bottom)
    mask[:top] = 0
    mask[bottom:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    valid = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not valid:
        return None, mask, []
    # 가장 아래까지 이어지는 영역 우선: 멀리 있는 반대편 선을 따라가지 않게 한다.
    selected = max(valid, key=lambda i: (stats[i, 1] + stats[i, 3], stats[i, 4]))
    ys, xs = np.where(labels == selected)
    span = max(1, int(ys.max() - ys.min()))
    near = ys >= ys.min() + 0.65 * span
    far = ys <= ys.min() + 0.35 * span
    near_x = float(np.median(xs[near]))
    far_x = float(np.median(xs[far]))
    # 영상 오른쪽은 +오차. ROS 좌회전은 +각속도이므로 제어 시 부호를 반전한다.
    offset = (near_x - w / 2) / (w / 2)
    heading = (far_x - near_x) / (w / 2)
    points = [(int(near_x), int(np.median(ys[near]))),
              (int(far_x), int(np.median(ys[far])))]
    return (offset, heading), mask, points


class RedLineFollower(Node):
    def __init__(self):
        super().__init__('red_line_follower')
        defaults = dict(image_topic='/camera/image_raw', cmd_vel_topic='/cmd_vel',
                        line_valid_topic='/vision/line_valid',
                        linear_speed=0.08, max_angular_speed=0.65,
                        offset_gain=1.0, heading_gain=0.8,
                        max_linear_accel=0.10, max_angular_accel=0.8,
                        image_timeout=0.6, roi_top=0.52, roi_bottom=0.95,
                        min_area=80, show_debug=False)
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.p = {key: self.get_parameter(key).value for key in defaults}
        if not 0 <= self.p['roi_top'] < self.p['roi_bottom'] <= 1:
            raise ValueError('ROI 비율은 0 <= top < bottom <= 1이어야 합니다.')
        if any(self.p[k] <= 0 for k in ['image_timeout', 'max_linear_accel',
                                       'max_angular_accel', 'max_angular_speed', 'min_area']):
            raise ValueError('시간/가속도/각속도/면적 설정은 양수여야 합니다.')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Twist, self.p['cmd_vel_topic'], 10)
        self.valid_pub = self.create_publisher(Bool, self.p['line_valid_topic'], 10)
        self.debug_pub = self.create_publisher(Image, '~/debug_image', qos_profile_sensor_data)
        self.sub = self.create_subscription(Image, self.p['image_topic'], self.on_image,
                                            qos_profile_sensor_data)
        self.target = None
        self.last_image = None
        self.v = self.w = 0.0
        self.previous_tick = time.monotonic()
        # 시뮬레이션을 일시정지해도 수신 타임아웃은 실제 경과 시간으로 동작한다.
        self.timer = self.create_timer(0.05, self.control,
                                      clock=Clock(clock_type=ClockType.STEADY_TIME))

    def on_image(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.target, mask, points = detect_line(
                frame, self.p['roi_top'], self.p['roi_bottom'], self.p['min_area'])
            self.last_image = time.monotonic()
            if self.p['show_debug'] or self.debug_pub.get_subscription_count():
                overlay = frame.copy()
                overlay[mask > 0] = (0, 255, 255)
                for point in points:
                    cv2.circle(overlay, point, 6, (0, 255, 0), -1)
                text = 'LINE LOST: STOP' if self.target is None else (
                    f'offset={self.target[0]:+.2f} heading={self.target[1]:+.2f}')
                cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                out = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
                out.header = msg.header
                self.debug_pub.publish(out)
                if self.p['show_debug']:
                    cv2.imshow('Red line follower', overlay)
                    cv2.waitKey(1)
        except (CvBridgeError, cv2.error, ValueError) as exc:
            self.target = None
            self.get_logger().error(str(exc), throttle_duration_sec=2.0)

    def stop(self):
        self.v = self.w = 0.0
        self.pub.publish(Twist())
        self.valid_pub.publish(Bool(data=False))

    def control(self):
        now = time.monotonic()
        dt = min(0.1, max(0.001, now - self.previous_tick))
        self.previous_tick = now
        if self.last_image is None or now - self.last_image > self.p['image_timeout']:
            self.stop()
            self.get_logger().warn('카메라 미수신/시간 초과: STOP', throttle_duration_sec=3.0)
            return
        if self.target is None:
            self.stop()
            self.get_logger().warn('빨간 선 미검출: STOP', throttle_duration_sec=2.0)
            return
        offset, heading = self.target
        self.valid_pub.publish(Bool(data=True))
        turn = self.p['offset_gain'] * offset + self.p['heading_gain'] * heading
        desired_w = float(np.clip(-turn, -self.p['max_angular_speed'], self.p['max_angular_speed']))
        desired_v = self.p['linear_speed'] * max(0.2, 1 - abs(turn))
        if abs(turn) > 0.8:
            desired_v = 0.0  # 선이 가장자리로 치우치면 전진보다 방향 복구 우선
        self.v += float(np.clip(desired_v - self.v, -self.p['max_linear_accel'] * dt,
                                self.p['max_linear_accel'] * dt))
        self.w += float(np.clip(desired_w - self.w, -self.p['max_angular_accel'] * dt,
                                self.p['max_angular_accel'] * dt))
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = self.v, self.w
        self.pub.publish(cmd)
        self.get_logger().info(f'v={self.v:.3f} m/s w={self.w:.3f} rad/s',
                               throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = RedLineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

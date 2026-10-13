"""Autorace 2020의 노란색/흰색 차선을 검출해 중심 오차를 발행한다."""

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32


def largest_centroid(mask, min_pixels=100):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    valid = [index for index in range(1, count)
             if stats[index, cv2.CC_STAT_AREA] >= min_pixels]
    if not valid:
        return None
    selected = max(valid, key=lambda index: stats[index, cv2.CC_STAT_AREA])
    return float(centroids[selected][0]), int(stats[selected, cv2.CC_STAT_AREA])


def detect_autorace_lane(frame, roi_top=0.55, lane_half_width_px=115.0,
                         min_pixels=100):
    """차선 중심의 정규화 오차와 디버그 마스크를 반환한다."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 80, 70), (40, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 175), (179, 80, 255))
    height, width = yellow.shape
    cutoff = int(height * roi_top)
    yellow[:cutoff] = 0
    white[:cutoff] = 0
    kernel = np.ones((5, 5), np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, kernel)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
    yellow_point = largest_centroid(yellow, min_pixels)
    white_point = largest_centroid(white, min_pixels)
    if yellow_point and white_point:
        center_x = 0.5 * (yellow_point[0] + white_point[0])
        confidence = 'both'
    elif yellow_point:
        center_x = yellow_point[0] + lane_half_width_px
        confidence = 'yellow'
    elif white_point:
        center_x = white_point[0] - lane_half_width_px
        confidence = 'white'
    else:
        return None, yellow, white, 'none'
    error = (center_x - width * 0.5) / max(width * 0.5, 1.0)
    return float(np.clip(error, -1.5, 1.5)), yellow, white, confidence


class AutoraceLaneFollower(Node):
    def __init__(self):
        super().__init__('autorace_lane_follower')
        defaults = {
            'image_topic': '/camera/image_raw',
            'line_valid_topic': '/vision/line_valid',
            'lane_error_topic': '/vision/lane_error',
            'roi_top': 0.55,
            'lane_half_width_px': 115.0,
            'min_pixels': 100,
            'valid_frames': 3,
            'lost_frames': 3,
            'image_timeout': 0.6,
            'publish_debug': True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.bridge = CvBridge()
        self.last_image = None
        self.last_error = 0.0
        self.valid_count = 0
        self.lost_count = 0
        self.valid = False
        self.valid_pub = self.create_publisher(
            Bool, str(self.get_parameter('line_valid_topic').value), 10)
        self.error_pub = self.create_publisher(
            Float32, str(self.get_parameter('lane_error_topic').value), 10)
        self.debug_pub = self.create_publisher(
            Image, '~/debug_image', qos_profile_sensor_data)
        self.create_subscription(
            Image, str(self.get_parameter('image_topic').value),
            self.image_callback, qos_profile_sensor_data)
        self.create_timer(
            0.1, self.watchdog,
            clock=Clock(clock_type=ClockType.STEADY_TIME))

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            result, yellow, white, confidence = detect_autorace_lane(
                frame,
                float(self.get_parameter('roi_top').value),
                float(self.get_parameter('lane_half_width_px').value),
                int(self.get_parameter('min_pixels').value))
            self.last_image = time.monotonic()
            if result is None:
                self.valid_count = 0
                self.lost_count += 1
                if self.lost_count >= int(self.get_parameter('lost_frames').value):
                    self.valid = False
            else:
                self.last_error = result
                self.lost_count = 0
                self.valid_count += 1
                if self.valid_count >= int(self.get_parameter('valid_frames').value):
                    self.valid = True
                self.error_pub.publish(Float32(data=self.last_error))
            self.valid_pub.publish(Bool(data=self.valid))

            if bool(self.get_parameter('publish_debug').value):
                overlay = frame.copy()
                overlay[yellow > 0] = (0, 255, 255)
                overlay[white > 0] = (255, 0, 255)
                text = (f'lane={confidence} error={self.last_error:+.2f}'
                        if self.valid else 'LANE INVALID')
                cv2.putText(overlay, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 0) if self.valid else (0, 0, 255), 2)
                debug = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
                debug.header = msg.header
                self.debug_pub.publish(debug)
        except (CvBridgeError, cv2.error, ValueError) as error:
            self.valid = False
            self.valid_pub.publish(Bool(data=False))
            self.get_logger().error(str(error), throttle_duration_sec=2.0)

    def watchdog(self):
        if (self.last_image is None or
                time.monotonic() - self.last_image >
                float(self.get_parameter('image_timeout').value)):
            if self.valid:
                self.get_logger().warn('카메라 시간 초과: vision invalid')
            self.valid = False
            self.valid_pub.publish(Bool(data=False))


def main(args=None):
    rclpy.init(args=args)
    node = AutoraceLaneFollower()
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

import cv2
import rclpy

from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')

        # CvBridge 클래스의 객체 생성
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data,
        )

    def image_callback(self, msg):
        try:
            # ROS Image → NumPy 배열
            # OpenCV의 컬러 표시 순서에 맞게 BGR로 변환
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )

        except CvBridgeError as error:
            # 변환 실패 시 오류를 출력하고 이번 콜백 종료
            self.get_logger().error(str(error))
            return

        # frame.shape는 (높이, 너비, 채널 수) 형태의 튜플
        self.get_logger().info(
            f'입력={msg.encoding}, 배열 크기={frame.shape}',
            throttle_duration_sec=2.0,
        )

        cv2.imshow('Camera viewer', frame)

        # 창의 키보드·화면 갱신 이벤트 처리
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
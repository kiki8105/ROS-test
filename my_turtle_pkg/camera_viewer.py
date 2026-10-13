# import는 다른 파일(라이브러리)의 기능을 현재 파일에서 사용할 수 있게 가져오는 문법입니다.
# cv2는 OpenCV의 파이썬 이름으로, 카메라 영상 창을 띄우고 화면 이벤트를 처리합니다.
import cv2
# rclpy는 ROS 2를 파이썬에서 사용하는 라이브러리입니다.
import rclpy

# from A import B는 A라는 묶음에서 B라는 이름만 골라 가져오는 문법입니다.
# CvBridge는 ROS 영상 메시지와 OpenCV 영상 배열을 서로 변환하고,
# CvBridgeError는 그 변환 중 발생할 수 있는 오류의 종류입니다.
from cv_bridge import CvBridge, CvBridgeError
# Node는 ROS 2 노드의 기본 기능(통신, 로그 등)을 제공하는 부모 클래스입니다.
from rclpy.node import Node
# 센서 데이터가 자주 들어올 때 사용하는 ROS 2 권장 통신 품질 설정입니다.
from rclpy.qos import qos_profile_sensor_data
# Image는 ROS 2에서 카메라 영상 한 장을 전달할 때 사용하는 메시지 형식입니다.
from sensor_msgs.msg import Image


# class는 새로운 자료형(클래스)을 정의하는 문법입니다.
# CameraViewer는 아래에서 Node를 상속받아 카메라 영상을 보여주는 ROS 2 노드입니다.
class CameraViewer(Node):
    # def는 함수를 정의하는 문법이고, __init__은 객체를 만들 때 자동 실행되는 초기화 함수입니다.
    # self는 지금 만들어지는 이 객체 자신을 가리키며, 메서드의 첫 매개변수로 씁니다.
    def __init__(self):
        # super()는 부모 클래스(Node)를 가리킵니다.
        # 부모 초기화를 먼저 실행하고, 괄호 안 문자열을 이 노드의 이름으로 등록합니다.
        super().__init__('camera_viewer')

        # = 는 오른쪽 값을 왼쪽 이름에 저장(대입)합니다.
        # CvBridge()는 변환 도구 객체를 새로 만들고, self.bridge에 보관합니다.
        self.bridge = CvBridge()

        # create_subscription은 특정 주제를 구독하고, 메시지가 도착할 때 실행할 함수를 등록합니다.
        # 구독 객체를 self.subscription에 저장해 노드가 구독을 유지하도록 합니다.
        self.subscription = self.create_subscription(
            Image,                         # 받을 메시지의 자료형: ROS 영상 메시지
            '/camera/image_raw',           # 구독할 주제(topic)의 이름
            self.image_callback,           # 메시지가 도착할 때 호출할 이 노드의 함수
            qos_profile_sensor_data,       # 센서 영상에 알맞은 통신 품질 설정
        )

    # image_callback은 영상 메시지가 새로 도착할 때 ROS 2가 호출하는 콜백 함수입니다.
    # callback은 어떤 사건(여기서는 메시지 도착)에 답해 실행되는 함수를 뜻합니다.
    # msg는 message의 줄임말로, 이번에 받은 영상 메시지가 들어옵니다.
    def image_callback(self, msg):
        # try 아래에서 오류가 날 수 있는 코드를 실행하고, 아래 except에서 지정한 오류를 처리합니다.
        try:
            # self.bridge는 위에서 만든 변환 도구이고, 점(.)은 객체 안의 기능/값에 접근한다는 뜻입니다.
            # imgmsg_to_cv2는 ROS Image 메시지를 OpenCV가 다룰 수 있는 NumPy 배열로 변환합니다.
            # msg는 변환할 입력이며, desired_encoding은 원하는 픽셀 색상 형식입니다.
            # bgr8은 파랑-초록-빨강(BGR) 순서의 8비트 컬러 영상이라는 뜻입니다.
            # 괄호를 여러 줄로 나누어 쓰면 인자가 많아도 읽기 쉽고, frame에 결과 배열을 저장합니다.
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8',
            )

        # except는 try에서 지정한 오류가 발생했을 때 실행할 처리 구역입니다.
        # as error는 발생한 오류 객체를 error라는 이름으로 받아 아래에서 사용한다는 뜻입니다.
        except CvBridgeError as error:
            # get_logger()는 ROS 로거를 가져오고, error()는 오류 수준의 로그를 남깁니다.
            # str(error)는 오류 객체를 사람이 읽을 수 있는 문자열로 바꿉니다.
            self.get_logger().error(str(error))
            # return은 현재 콜백 함수를 즉시 끝냅니다. 잘못된 영상으로 화면 표시를 계속하지 않습니다.
            return

        # frame.shape는 배열의 크기를 (높이, 너비, 색상 채널 수) 순서로 알려주는 값입니다.
        # get_logger().info(...)는 일반 정보 로그를 출력합니다.
        # f'...'는 f-string 문법으로, 중괄호 { } 안의 파이썬 값을 문자열에 끼워 넣습니다.
        # msg.encoding은 원본 ROS 메시지의 색상 인코딩이고, frame.shape는 변환된 배열의 크기입니다.
        # throttle_duration_sec는 같은 로그를 매번 찍지 않고 2초에 한 번으로 제한하는 키워드 인자입니다.
        self.get_logger().info(
            f'입력={msg.encoding}, 배열 크기={frame.shape}',
            throttle_duration_sec=2.0,
        )

        # cv2.imshow는 OpenCV 창에 영상을 표시합니다.
        # 첫 번째 인자는 창 제목이고, 두 번째 인자는 표시할 영상 배열입니다.
        cv2.imshow('Camera viewer', frame)

        # waitKey(1)은 약 1밀리초 동안 키보드 입력과 창 갱신 이벤트를 처리합니다.
        # OpenCV 창이 실제로 계속 갱신되도록 imshow 뒤에 호출해야 합니다.
        cv2.waitKey(1)


# main은 프로그램을 시작할 때 필요한 초기화와 실행 순서를 모아 둔 함수입니다.
# args=None은 호출자가 인자를 주지 않으면 기본값 None을 사용한다는 뜻입니다.
def main(args=None):
    # ROS 2 통신 기능을 초기화합니다. args가 있으면 ROS 2 명령행 인자로 전달합니다.
    rclpy.init(args=args)
    # CameraViewer()로 노드 객체를 만들고 node라는 이름에 저장합니다.
    node = CameraViewer()

    # try 안에서 실행 중 문제가 생기더라도 finally의 정리 작업은 실행됩니다.
    try:
        # spin은 노드를 계속 실행하며 메시지 수신과 콜백 호출을 기다립니다.
        rclpy.spin(node)
    # Ctrl+C로 종료할 때 발생하는 키보드 중단을 잡아 정상적인 정리 단계로 넘어갑니다.
    except KeyboardInterrupt:
        # pass는 여기서 추가로 할 작업이 없음을 나타내는 파이썬 문장입니다.
        pass
    # finally는 try가 정상 종료되거나 예외가 생겨도 항상 실행되는 정리 구역입니다.
    finally:
        # 열려 있는 OpenCV 창을 모두 닫습니다.
        cv2.destroyAllWindows()
        # destroy_node()는 ROS 2 노드를 종료하고 노드가 사용한 자원을 정리합니다.
        node.destroy_node()

        # if는 조건이 참일 때만 들여쓰기된 코드를 실행합니다.
        # rclpy.ok()가 참이면 ROS 2가 아직 초기화된 상태이므로 종료 처리를 호출합니다.
        if rclpy.ok():
            rclpy.shutdown()


# __name__은 파이썬이 파일에 자동으로 붙이는 특별한 변수입니다.
# 이 파일을 직접 실행하면 값이 '__main__'이므로 main()을 시작합니다.
# 다른 파일에서 import한 경우에는 이 조건이 거짓이어서 자동 실행하지 않습니다.
if __name__ == '__main__':
    main()

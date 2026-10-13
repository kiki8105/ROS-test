# import는 다른 파일(라이브러리)의 기능을 현재 파일에서 사용할 수 있게 가져오는 문법입니다.
# cv2는 OpenCV의 파이썬 이름으로, 카메라 영상 창을 띄우고 화면 이벤트를 처리합니다.
import cv2
# rclpy는 ROS 2를 파이썬에서 사용하는 라이브러리입니다.
import rclpy

# numpy는 숫자 배열을 빠르게 만들고 다루는 라이브러리입니다.
# 관례상 np라는 짧은 별명(alias)으로 부르며, 아래에서 np.frombuffer처럼 사용합니다.
import numpy as np
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
        super().__init__('camera_viewer_manual')

        # create_subscription은 특정 주제를 구독하고, 메시지가 도착할 때 실행할 함수를 등록합니다.
        # 구독 객체를 self.subscription에 저장해 노드가 구독을 유지하도록 합니다.
        self.subscription = self.create_subscription(
            Image,                         # 받을 메시지의 자료형: ROS 영상 메시지
            '/camera/image_raw',           # 구독할 주제(topic)의 이름
            self.image_callback,           # 메시지가 도착할 때 호출할 이 노드의 함수
            qos_profile_sensor_data,       # 센서 영상에 알맞은 통신 품질 설정
        )

    # convert_image는 ROS 영상 메시지(msg)를 OpenCV가 표시할 NumPy 배열로 바꾸는 메서드입니다.
    # 메서드는 클래스 안에 정의된 함수이고, self는 이 메서드를 가진 CameraViewer 객체 자신입니다.
    # msg는 message(메시지)의 줄임말이며, width/height/data/encoding 같은 영상 정보를 담습니다.
    def convert_image(self, msg):
        # channels_by_encoding은 딕셔너리입니다. {키: 값} 형태로 관련 정보를 짝지어 저장합니다.
        # 여기서는 색상 형식(키)을 픽셀 하나의 채널 수(값)에 연결합니다.
        channels_by_encoding = {
            'rgb8': 3,  # RGB 순서의 8비트 컬러: 빨강, 초록, 파랑 채널 3개
            'bgr8': 3,  # BGR 순서의 8비트 컬러: 파랑, 초록, 빨강 채널 3개
            'mono8': 1, # 8비트 흑백: 밝기 채널 1개
        }

        # not in은 "왼쪽 값이 오른쪽 묶음 안에 없다"는 검사입니다.
        # 메시지의 encoding이 위 딕셔너리에 없으면 이 코드가 참이 됩니다.
        if msg.encoding not in channels_by_encoding:
            # raise는 오류를 일부러 발생시켜 현재 처리를 중단합니다.
            # ValueError는 값이 이 함수에서 지원하는 형식이 아닐 때 알리는 오류 종류입니다.
            # f'...'는 f-string이며, {msg.encoding} 자리에 실제 인코딩 문자열을 넣습니다.
            raise ValueError(f'지원하지 않는 인코딩: {msg.encoding}')

        # 딕셔너리[키]는 해당 키의 값을 꺼냅니다.
        # 예를 들어 encoding이 'rgb8'이면 channels에는 3이 들어갑니다.
        channels = channels_by_encoding[msg.encoding]

        # row_bytes는 픽셀 자체만 차지하는 한 줄의 바이트 수입니다.
        # width는 가로 픽셀 수, channels는 픽셀당 채널 수입니다.
        # 곱하기(*)로 가로 픽셀마다 필요한 바이트를 계산합니다(8비트 채널 하나가 1바이트).
        row_bytes = msg.width * channels

        # or는 "또는"이라는 논리 연산자입니다. 가로 또는 세로 크기가 0이면 영상이 비어 있습니다.
        if msg.height == 0 or msg.width == 0:
            # ==는 두 값이 같은지 비교합니다. 크기가 0인 잘못된 영상은 ValueError로 알립니다.
            raise ValueError('이미지 크기가 0입니다.')

        # step은 메시지에서 한 행이 차지하는 전체 바이트 수입니다.
        # 실제 픽셀 뒤에 정렬을 위한 여분(padding)이 있을 수 있어 row_bytes보다 클 수 있습니다.
        # step이 row_bytes보다 작으면 픽셀 데이터가 한 행 안에 다 들어갈 수 없습니다.
        if msg.step < row_bytes:
            raise ValueError('step이 이미지 한 줄의 크기보다 작습니다.')

        # len(...)은 데이터의 길이를 셉니다. height * step은 전체 행이 차지해야 하는 바이트 수입니다.
        # !=는 "같지 않다"는 뜻이며, 예상 길이와 실제 data 길이가 다르면 메시지가 잘못된 것입니다.
        if len(msg.data) != msg.height * msg.step:
            raise ValueError('이미지 데이터 길이가 맞지 않습니다.')

        # np.frombuffer는 bytes 같은 버퍼를 복사하지 않고 NumPy 배열처럼 읽게 합니다.
        # msg.data는 원본 바이트 버퍼이고, dtype은 배열 원소 하나의 자료형을 지정합니다.
        # np.uint8은 부호 없는(unsigned) 8비트 정수로, 영상 바이트 값 0~255를 표현합니다.
        data = np.frombuffer(msg.data, dtype=np.uint8)

        # reshape는 배열 모양(shape)을 바꿉니다. 여기서는 높이 행, 행마다 step 바이트로 나눕니다.
        # step을 그대로 사용하므로 행 끝의 padding도 각 행과 함께 보존됩니다.
        rows = data.reshape(msg.height, msg.step)

        # 대괄호 안의 콤마는 [행 선택, 열 선택]을 뜻합니다.
        # 첫 번째 : 는 모든 행, 두 번째 :row_bytes는 처음부터 row_bytes 직전까지의 열입니다.
        # 따라서 행 끝의 padding은 빼고 실제 픽셀 바이트만 남깁니다.
        pixels = rows[:, :row_bytes]

        # 채널 수가 1이면 흑백 영상입니다.
        if channels == 1:
            # 흑백에는 채널 축이 따로 없으므로 (높이, 너비) 모양으로 배열을 다시 나눕니다.
            # reshape의 두 인자는 결과 배열의 행 수와 열 수입니다.
            gray = pixels.reshape(msg.height, msg.width)
            # cvtColor는 OpenCV의 색상 변환 함수입니다.
            # COLOR_GRAY2BGR는 흑백 한 채널을 BGR 세 채널로 바꾸어 일반 컬러 창에서도 표시되게 합니다.
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 여기까지 왔다면 channels는 3입니다. 픽셀 바이트를 (높이, 너비, 채널) 모양으로 나눕니다.
        frame = pixels.reshape(msg.height, msg.width, channels)

        # ROS의 rgb8은 RGB 순서지만 OpenCV는 기본적으로 BGR 순서를 사용합니다.
        if msg.encoding == 'rgb8':
            # frame[:, :, ::-1]에서 앞의 두 :는 모든 행과 열을 선택합니다.
            # 마지막 ::-1은 채널 축을 뒤에서 앞으로 읽어 RGB를 BGR 순서로 뒤집습니다.
            frame = frame[:, :, ::-1]

        # 색상 채널을 뒤집으면 메모리의 stride(다음 원소까지 건너뛸 간격)가 음수가 될 수 있습니다.
        # np.ascontiguousarray는 픽셀이 메모리에 연속해서 놓인 배열로 만들어 반환합니다.
        # return은 이 결과를 호출한 곳으로 돌려주고 convert_image 실행을 끝냅니다.
        return np.ascontiguousarray(frame)

    # image_callback은 영상 메시지가 새로 도착할 때 ROS 2가 호출하는 콜백 메서드입니다.
    # callback은 어떤 사건(여기서는 메시지 도착)에 답해 실행되는 함수를 뜻합니다.
    # msg에는 이번에 받은 영상 메시지가 전달됩니다.
    def image_callback(self, msg):
        # try는 오류가 날 수 있는 코드를 실행하고, except는 지정한 종류의 오류를 처리합니다.
        try:
            # self.convert_image(...)는 위에서 정의한 변환 메서드를 호출합니다.
            # 괄호 안 msg가 함수의 입력 인자이고, 변환 결과를 frame이라는 이름에 저장합니다.
            frame = self.convert_image(msg)

        # 변환 함수가 raise한 ValueError가 생기면 이 구역으로 이동합니다.
        # as error는 오류 객체를 error라는 변수에 받아 메시지를 기록할 수 있게 합니다.
        except ValueError as error:
            # get_logger()는 ROS 로그 도구를 가져오고, error(...)는 오류 수준의 로그를 출력합니다.
            # str(error)는 오류 객체를 화면에 쓸 수 있는 문자열로 바꿉니다.
            self.get_logger().error(str(error))
            # return만 쓰면 값을 돌려주지 않고 현재 콜백을 끝냅니다.
            # 따라서 잘못된 메시지로 로그나 화면 표시를 계속하지 않습니다.
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
        cv2.imshow('Camera viewer manual', frame)

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
import rclpy  # 모듈 전체를 가져옴. 이후 rclpy.init()처럼 사용
from rclpy.node import Node  # 모듈에서 Node 클래스만 가져옴
from geometry_msgs.msg import Twist  # 선속도·각속도를 담는 메시지 클래스
import math
import time

from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data

# class 클래스명(부모클래스): → 부모의 기능을 상속받는 클래스 정의
class ActionPublisher(Node):

    # __init__: 객체를 생성할 때 자동으로 호출되는 초기화 메서드
    # self: 생성된 객체 자신. 메서드의 첫 번째 인자로 받음
    def __init__(self):
        # super(): 부모 클래스에 접근
        # 부모인 Node를 초기화하고 ROS 노드 이름을 지정
        super().__init__('action_publisher')

        # self에 저장한 값은 다른 메서드에서도 읽고 변경할 수 있음
        self.action = 'stop'
        self.last_scan_time = None

        self.subscription = self.create_subscription(
            LaserScan,                # 받을 메시지 타입
            '/scan_mock',             # 구독할 토픽
            self.scan_callback,       # 메시지를 받을 때 실행할 함수
            qos_profile_sensor_data,  # 센서 데이터용 QoS
        )

        # self.publisher: 객체에 저장하는 속성으로, 다른 메서드에서도 사용 가능
        # 괄호 안에서는 여러 줄에 걸쳐 인자를 작성할 수 있음
        self.publisher = self.create_publisher(
            Twist,            # 메시지 타입: 클래스 자체를 전달
            '/cmd_vel_mock',  # 발행할 토픽 이름
            10,               # QoS 이력 깊이: Keep Last 기준 10개
        )                     # 마지막 인자 뒤의 쉼표도 허용됨

        # 0.1: 실수(float), 단위는 초 → 초당 약 10회 호출
        # self.publish_command: 함수 자체를 전달하여 나중에 실행하도록 등록
        # self.publish_command()처럼 괄호를 붙이면 여기서 즉시 실행됨
        self.timer = self.create_timer(0.1, self.publish_command)

    # def: 함수를 정의하는 키워드. 클래스 안의 함수는 메서드라고 부름
    def publish_command(self):
        # get_parameter()는 파라미터 객체를 반환
        # .value로 실제 값('stop', 'forward' 등)을 가져옴
        # action은 이 메서드 안에서 사용하는 지역 변수
        # action = self.get_parameter('action').value

        # 아직 센서를 못 받았거나 마지막 수신 후 n초가 지나면 정지
        if (
            self.last_scan_time is None
            or time.monotonic() - self.last_scan_time > 5.0
        ):
            action = 'stop'

        else:
            action = self.action  # self.action을 지역 변수 action에 복사

        
        # 클래스명() → 해당 클래스의 새 객체 생성
        # Twist의 선속도·각속도 성분은 기본적으로 모두 0.0
        command = Twist()

        # if: 조건이 참이면 들여쓴 블록 실행
        # ==는 같은지 비교하는 연산자, =는 값을 대입하는 연산자
        if action == 'forward':
            # .은 객체의 속성에 접근하는 문법
            # linear의 x 성분에 전진 속도 0.1m/s 대입
            command.linear.x = 0.1

        # elif: 앞선 조건이 거짓일 때 다음 조건 검사
        elif action == 'left':
            # 양의 Z축 각속도 → 왼쪽으로 제자리 회전, 단위 rad/s
            command.angular.z = 0.3

        elif action == 'right':
            # 음수 각속도 → 오른쪽으로 제자리 회전
            command.angular.z = -0.3

        # 조건에 해당하지 않으면 기본값 0.0이 유지됨
        # 따라서 'stop'이나 알 수 없는 문자열은 정지 명령이 됨
        # 이 줄은 if와 같은 들여쓰기이므로 조건과 관계없이 실행됨
        self.publisher.publish(command)

        # get_logger()로 로거 객체를 얻고, info()로 정보 로그 출력
        self.get_logger().info(
            # f'문자열': 중괄호 안의 변수·표현식을 문자열에 삽입
            f'action={action}, '

            # :.2f → 실수를 소수점 아래 2자리까지 표시
            # 괄호 안에 나란히 놓인 문자열들은 하나로 이어짐
            f'linear={command.linear.x:.2f}, '
            f'angular={command.angular.z:.2f}',

            # 이름=값: 함수 호출에서 사용하는 키워드 인자
            # 로그 출력 간격을 최소 1초로 제한. 발행 주기는 여전히 0.1초
            throttle_duration_sec=1.0,
        )

    def sector_min(self, scan, center_deg):
        """지정한 방향의 좌우 10도 범위에서 가장 가까운 거리 반환."""
        distances = []

        # enumerate(): 배열의 인덱스와 값을 함께 꺼냄
        for index, distance in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment

            # 기준 방향과의 차이를 -π~π 범위로 변환
            # 덕분에 전방 0도 주변의 359도도 함께 검사됨
            difference = angle - math.radians(center_deg)
            difference = math.atan2(
                math.sin(difference),
                math.cos(difference),
            )

            if abs(difference) <= math.radians(10):
                if (
                    math.isfinite(distance)
                    and scan.range_min <= distance <= scan.range_max
                ):
                    distances.append(distance)

        # 유효한 측정값이 없으면 여유 공간으로 판단하지 않음
        return min(distances) if distances else 0.0
    
    def scan_callback(self, scan):
        front = self.sector_min(scan, 0)
        left = self.sector_min(scan, 90)
        right = self.sector_min(scan, -90)

        threshold = 0.6

        if front > threshold:
            self.action = 'forward'
        elif left <= threshold and right <= threshold:
            self.action = 'stop'
        elif left >= right:
            self.action = 'left'
        else:
            self.action = 'right'

        # 시스템 시각 변경의 영향을 받지 않는 경과 시간 측정용 시계
        self.last_scan_time = time.monotonic()

        self.get_logger().info(
            f'전방={front:.2f}, 좌측={left:.2f}, '
            f'우측={right:.2f} → {self.action}'
        )


# 클래스 밖의 일반 함수
# args=None: 인자를 생략하면 기본값으로 None 사용
# None은 '값이 없음'을 나타내는 Python 객체
def main(args=None):
    # ROS 2 통신 환경 초기화
    # args=args: 왼쪽은 인자 이름, 오른쪽은 main에서 받은 변수
    rclpy.init(args=args)

    # ActionPublisher 객체 생성 → 위의 __init__이 실행됨
    node = ActionPublisher()

    # try: 실행 중 발생할 수 있는 예외를 처리하기 위한 블록
    try:
        # 콜백을 처리하며 대기. 여기서는 타이머 콜백을 반복 실행
        rclpy.spin(node)

    # Ctrl+C로 발생하는 KeyboardInterrupt 예외를 처리
    except KeyboardInterrupt:
        # pass: 아무 동작도 하지 않고 넘어가는 문법
        pass

    # finally: 정상 종료나 예외 발생 여부와 관계없이 정리 코드 실행
    finally:
        # 노드가 사용하던 발행자·타이머 등의 자원 해제
        node.destroy_node()

        # ok()가 True이면 ROS 환경이 아직 활성 상태
        if rclpy.ok():
            rclpy.shutdown()  # ROS 환경 종료


# 이 파일을 직접 실행하면 __name__의 값이 '__main__'이 됨
# 다른 파일에서 import하면 모듈 이름이 되므로 아래 main()은 실행되지 않음
if __name__ == '__main__':
    main()  # 함수를 실제로 호출
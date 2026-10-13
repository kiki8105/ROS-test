import json
import math
from datetime import datetime
from getpass import getpass

import pymysql
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LidarDbLogger(Node):
    def __init__(self, conn):
        super().__init__('lidar_db_logger')

        # main()에서 생성한 DB 연결을 객체에 보관
        self.conn = conn

        self.subscription = self.create_subscription(
            String,
            '/lidar_record',
            self.record_callback,
            10,
        )

    def record_callback(self, message):
        try:
            # JSON 문자열 → Python 딕셔너리
            record = json.loads(message.data)

            # 원본 측정값을 각도 기준 360칸으로 변환
            # 측정되지 않은 칸은 None으로 유지
            ranges = self.to_360_bins(record)

            # 변환 함수가 360칸을 생성하므로
            # 기존 길이 검사와 원본 각도 제한 검사는 제거

            with self.conn.cursor() as cursor:
                sql = """
                    INSERT INTO lidardata (`ranges`, `when`, action)
                    VALUES (%s, %s, %s)
                """

                values = (
                    json.dumps(ranges, allow_nan=False),
                    datetime.fromisoformat(record['when']),
                    record['action'],
                )

                cursor.execute(sql, values)

            # INSERT 결과를 DB에 확정
            self.conn.commit()

            # 실제 측정값이 들어 있는 칸의 개수
            valid_count = sum(
                value is not None for value in ranges
            )

            self.get_logger().info(
                f"저장 완료: 원본 {len(record['ranges'])}개 → "
                f"360칸 중 유효 {valid_count}칸, "
                f"action={record['action']}",
                throttle_duration_sec=1.0,
            )

        except Exception as error:
            try:
                self.conn.rollback()
            except pymysql.MySQLError:
                pass

            self.get_logger().error(f'DB 저장 실패: {error}')

    @staticmethod
    def to_360_bins(record):
        # @staticmethod:
        # self를 사용하지 않고 입력값만으로 계산하는 메서드
        angle_min = float(record['angle_min'])
        angle_increment = float(record['angle_increment'])

        if (
            not math.isfinite(angle_min)
            or not math.isfinite(angle_increment)
            or angle_increment == 0.0
        ):
            raise ValueError('유효하지 않은 스캔 각도 정보입니다.')

        # None 360개로 리스트 초기화
        # 측정값이 배치되지 않은 칸은 None으로 유지
        bins = [None] * 360

        for index, distance in enumerate(record['ranges']):
            if distance is None:
                continue

            distance = float(distance)

            if not math.isfinite(distance) or distance < 0.0:
                continue

            # 배열의 인덱스를 실제 측정 각도로 변환
            angle_rad = angle_min + index * angle_increment

            # 라디안 → 도, 음수 각도도 0~360도 범위로 변환
            angle_deg = math.degrees(angle_rad) % 360.0

            # 가장 가까운 정수 각도의 칸 선택
            # 359.8도는 0도 칸으로 들어감
            bin_index = int(math.floor(angle_deg + 0.5)) % 360

            # 같은 칸에 여러 측정값이 들어오면 가까운 장애물 유지
            previous = bins[bin_index]

            if previous is None or distance < previous:
                bins[bin_index] = distance

        return bins


def main(args=None):
    # 비밀번호는 실행 시 입력. 코드나 Git에 저장하지 않음
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password=getpass('MySQL 비밀번호: '),
        database='rosdb',
        charset='utf8mb4',
        connect_timeout=5,
        read_timeout=5,
        write_timeout=5,
    )

    rclpy.init(args=args)
    node = None

    try:
        node = LidarDbLogger(conn)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        conn.close()


if __name__ == '__main__':
    main()

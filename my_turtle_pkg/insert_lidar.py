import json
from datetime import datetime
from getpass import getpass

import pymysql

# 같은 폴더에 있는 생성 파일에서 함수 가져오기
from generate_lds02_dataset import generate_single_scan


# 전방에 벽이 있는 모의 스캔 생성
scan = generate_single_scan('front_wall')
ranges = scan['ranges']

# 이번 테스트는 좌우 여유가 같을 때 좌회전하는 기존 규칙에 맞춤
action = 'left'

# len(): 리스트 원소 개수 확인
if len(ranges) != 360:
    raise ValueError('거리 데이터는 360개여야 합니다.')

conn = pymysql.connect(
    host='localhost',
    user='root',
    password=getpass('MySQL 비밀번호: '),
    database='rosdb',
    charset='utf8mb4',
)

try:
    with conn.cursor() as cursor:
        # %s: 값을 전달할 자리 표시자
        # 직접 문자열을 이어 붙이지 않고 execute()에 값을 별도로 전달
        sql = """
            INSERT INTO lidardata (`ranges`, `when`, action)
            VALUES (%s, %s, %s)
        """

        # json.dumps(): Python 리스트 → JSON 문자열
        # allow_nan=False: NaN·Infinity가 있으면 잘못된 JSON 대신 오류 발생
        values = (
            json.dumps(ranges, allow_nan=False),
            datetime.now(),
            action,
        )

        cursor.execute(sql, values)
        inserted_id = cursor.lastrowid  # 자동 생성된 id

    # 변경 내용을 DB에 확정. 기본 설정에서는 반드시 필요
    conn.commit()
    print(f'저장 완료: id={inserted_id}, 거리={len(ranges)}개, action={action}')

except Exception:
    conn.rollback()  # 실패하면 아직 확정하지 않은 변경 취소
    raise            # 발생한 오류를 다시 전달하여 원인 표시

finally:
    conn.close()     # 성공·실패와 관계없이 연결 종료
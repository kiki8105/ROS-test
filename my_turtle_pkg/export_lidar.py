import json
from getpass import getpass

import pandas as pd
import pymysql


conn = pymysql.connect(
    host='localhost',
    user='root',
    password=getpass('MySQL 비밀번호: '),  # 입력 내용이 화면에 표시되지 않음
    database='rosdb',
    charset='utf8mb4',
)

try:
    # with: 블록이 끝나면 커서를 자동으로 닫음
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT id, `ranges`, action FROM lidardata ORDER BY id"
        )
        rows = cursor.fetchall()  # 조회 결과를 튜플들의 묶음으로 받음

    records = []

    # 각 행의 세 값을 변수 세 개에 나눠 담음: 언패킹
    for record_id, ranges_json, action in rows:
        # JSON 문자열 → Python 리스트
        ranges = json.loads(ranges_json)

        # 리스트인지, 거리 값이 360개인지 검사
        if not isinstance(ranges, list) or len(ranges) != 360:
            raise ValueError(
                f'id={record_id}: ranges는 길이가 360인 배열이어야 합니다.'
            )

        # 리스트 + 리스트 → 원소들을 이어 붙인 새 리스트
        # 거리 360개 뒤에 액션 하나를 붙임
        records.append(ranges + [action])

    # 리스트 컴프리헨션: range_0부터 range_359까지 이름 생성
    columns = [f'range_{i}' for i in range(360)] + ['action']

    # 모아둔 모든 행으로 DataFrame을 한 번에 생성
    df = pd.DataFrame(records, columns=columns)

    # 거리 열만 숫자형으로 변환. JSON null은 결측값으로 처리
    # 잘못된 문자열 등이 있으면 조용히 넘기지 않고 오류 발생
    distance_columns = columns[:-1]  # 마지막 action 열을 제외
    df[distance_columns] = df[distance_columns].apply(
        pd.to_numeric,
        errors='raise',
    )

    print(df)
    print(f'행 수: {df.shape[0]}, 열 수: {df.shape[1]}')

    # index=False: DataFrame의 행 번호를 CSV에 추가하지 않음
    df.to_csv('output.csv', index=False)

finally:
    # 중간에 오류가 발생해도 DB 연결 종료
    conn.close()
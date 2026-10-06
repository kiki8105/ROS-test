# 스캔 데이터를 읽어들여서 모의 주행하는 코드
import json
import numpy as np

with open("lds02_mock.json") as f:
    data = json.load(f)

ranges = np.array(data["ranges"])

# LDS-02는 angle_increment = 1도 라고 가정
front = np.r_[ranges[350:360], ranges[0:10]]
left  = ranges[80:100]
right = ranges[260:280]

front_dist = np.mean(front)
left_dist  = np.mean(left)
right_dist = np.mean(right)

# LDS-02의 안전거리 기준 예
safe_dist = 0.5  

if front_dist < safe_dist:
    action = "turn_left" if left_dist > right_dist else "turn_right"
else:
    action = "go_forward"

print("front:", round(front_dist, 2))
print("left :", round(left_dist, 2))
print("right:", round(right_dist, 2))
print("action:", action)
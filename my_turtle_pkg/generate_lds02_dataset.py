# 스캔 데이터 아무렇게나 생성하는 코드
import json
import math
import os
import random

ANGLE_MIN_DEG = 0
ANGLE_MAX_DEG = 359
ANGLE_INCREMENT_DEG = 1
NUM_POINTS = 360
RANGE_MIN = 0.12 # 미터
RANGE_MAX = 3.5 # 미터

def create_empty_scan():
    ranges = [RANGE_MAX for _ in range(NUM_POINTS)]
    intensities = [100 for _ in range(NUM_POINTS)]
    scan = {
        "angle_min": math.radians(ANGLE_MIN_DEG),
        "angle_max": math.radians(ANGLE_MAX_DEG),
        "angle_increment": math.radians(ANGLE_INCREMENT_DEG),
        "range_min": RANGE_MIN,
        "range_max": RANGE_MAX,
        "ranges": ranges,
        "intensities": intensities
    }
    return scan

# 벽 만들기 
def make_the_wall(ranges, center_deg, width_deg):
    half_width = width_deg // 2
    for offset in range(-half_width, half_width + 1):
        idx = (center_deg + offset) % NUM_POINTS
        ranges[idx] = 0.4

def pattern_front_wall(scan):
    make_the_wall(scan["ranges"], center_deg=0, width_deg=40)

def pattern_left_wall(scan):
    make_the_wall(scan["ranges"], center_deg=90, width_deg=30)

def pattern_right_wall(scan):
    make_the_wall(scan["ranges"], center_deg=270, width_deg=30)

def generate_single_scan(pattern_name):
    scan = create_empty_scan()
    if pattern_name == "front_wall":
        pattern_front_wall(scan)
    elif pattern_name == "left_wall":
        pattern_left_wall(scan)
    elif pattern_name == "right_wall":
        pattern_right_wall(scan)
    return scan

AVAILABLE_PATTERNS = [
    "front_wall",
    "left_wall",
    "right_wall"
]

def generate_dataset(num_scans, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(num_scans):
        pattern_name = random.choice(AVAILABLE_PATTERNS)
        scan = generate_single_scan(pattern_name)
        scan["meta"] = {
            "pattern": pattern_name,
            "index": i
        }
        filename = os.path.join(out_dir, f"lds02_mock_{i:03d}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(scan, f, ensure_ascii=False, indent=2)
        print(f"Saved {filename} (pattern={pattern_name})")

if __name__ == "__main__":
    generate_dataset(num_scans=1000, out_dir="lds02_dataset")
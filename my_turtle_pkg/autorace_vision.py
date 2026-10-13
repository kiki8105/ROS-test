import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tensorflow.keras.models import load_model


MODEL_DIR = Path(
    "/home/kiki/turtlebot3/train/autorace_keras"
)


class SignClassifier:
    def __init__(self):
        self.model = load_model(
            str(MODEL_DIR / "keras_model.h5"),
            compile=False,
        )

        # "0 construction" → "construction"
        # split(maxsplit=1)은 첫 공백을 기준으로 최대 한 번 분리
        lines = (MODEL_DIR / "labels.txt").read_text().splitlines()
        self.labels = [
            line.split(maxsplit=1)[1]
            for line in lines
            if line.strip()
        ]

        if self.model.input_shape != (None, 224, 224, 3):
            raise ValueError(
                f"입력 형태 확인 필요: {self.model.input_shape}"
            )

        if self.model.output_shape[-1] != len(self.labels):
            raise ValueError("모델 출력 개수와 라벨 개수가 다릅니다.")

    def predict(self, bgr_image):
        # OpenCV의 BGR 배열을 RGB 순서로 변환
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # 제공된 예제처럼 224×224로 크기 변경
        resized = cv2.resize(
            rgb, (224, 224), interpolation=cv2.INTER_AREA
        )

        # uint8 → float32, 값의 범위 0~255 → -1~1
        normalized = resized.astype(np.float32) / 127.5 - 1.0

        # (224, 224, 3) → (1, 224, 224, 3)
        # 첫 축은 한 번에 처리할 이미지 개수
        batch = np.expand_dims(normalized, axis=0)

        scores = self.model.predict(batch, verbose=0)[0]

        # 점수가 가장 큰 항목의 인덱스
        index = int(np.argmax(scores))
        return self.labels[index], float(scores[index])


class AutoraceVision(Node):
    def __init__(self, classifier):
        super().__init__("autorace_vision")

        self.classifier = classifier
        self.bridge = CvBridge()
        self.frame = None
        self.received_at = None

        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
            self.received_at = time.monotonic()
        except CvBridgeError as error:
            self.get_logger().error(str(error))


def color_masks(frame):
    # HSV: 색상(H), 채도(S), 밝기(V)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # OpenCV의 8비트 HSV에서 H 범위는 0~179
    # 아래 숫자는 초기 실습값이며 영상에 맞게 조절해야 함
    yellow = cv2.inRange(
        hsv,
        np.array([15, 80, 80], dtype=np.uint8),
        np.array([40, 255, 255], dtype=np.uint8),
    )

    # 흰색은 채도가 낮고 밝기가 높다는 조건으로 검출
    white = cv2.inRange(
        hsv,
        np.array([0, 0, 190], dtype=np.uint8),
        np.array([179, 65, 255], dtype=np.uint8),
    )

    # 차선 실습용으로 화면 아래쪽 45%만 사용
    # 위쪽 영역은 0으로 만들어 검출 결과에서 제외
    cutoff = int(frame.shape[0] * 0.55)
    yellow[:cutoff, :] = 0
    white[:cutoff, :] = 0

    return yellow, white


def find_sign_candidates(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 빨간색은 HSV의 H 범위 양 끝에 걸쳐 있어 두 구간을 사용
    red1 = cv2.inRange(hsv, (0, 90, 60), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 90, 60), (179, 255, 255))

    # 학습 이미지의 청록색 화살표까지 포함
    blue = cv2.inRange(hsv, (80, 70, 50), (130, 255, 255))

    # 노란색 표지판 후보. 차선도 포함될 수 있어 뒤에서 형태 검사
    yellow = cv2.inRange(hsv, (15, 90, 70), (40, 255, 255))

    # |는 마스크의 OR 연산: 하나라도 검출됐으면 흰색
    mask = red1 | red2 | blue | yellow

    # 작은 틈을 메워 하나의 윤곽으로 연결
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = frame.shape[:2]
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # 너무 작거나 화면을 크게 차지하는 후보 제외
        if area < 100 or area > height * width * 0.15:
            continue

        if min(w, h) < 12:
            continue

        # 지나치게 길쭉한 선 등 제외
        aspect_ratio = w / h
        if not 0.5 <= aspect_ratio <= 1.8:
            continue

        # 윤곽이 사각형에서 차지하는 면적 비율
        fill_ratio = area / (w * h)
        if fill_ratio < 0.25:
            continue

        # 표지판 테두리가 잘리지 않도록 주변 여백 추가
        padding = max(3, int(max(w, h) * 0.15))

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        candidates.append((area, (x1, y1, x2, y2)))

    # 면적이 큰 후보부터 최대 3개만 추론해 처리량 제한
    candidates.sort(key=lambda item: item[0], reverse=True)
    boxes = [box for _, box in candidates[:3]]

    return mask, boxes



def run_camera(classifier):
    rclpy.init()
    node = AutoraceVision(classifier)

    last_inference = 0.0
    processed_stamp = None

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            # 영상 창에서 ESC를 누르면 종료
            if cv2.waitKey(1) & 0xFF == 27:
                break

            if node.frame is None:
                continue

            now = time.monotonic()

            # 영상 수신이 끊기면 이전 검출 결과를 지우고 안내
            if now - node.received_at > 2.0:
                display = node.frame.copy()
                cv2.putText(
                    display, "NO NEW CAMERA FRAME", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2,
                )
                cv2.imshow("Autorace vision", display)
                continue

            # 새 프레임에 대해서만 최대 약 3회/초 처리
            if (
                node.received_at == processed_stamp
                or now - last_inference < 0.3
            ):
                continue

            frame = node.frame.copy()
            processed_stamp = node.received_at
            last_inference = now

            sign_mask, boxes = find_sign_candidates(frame)
            yellow, white = color_masks(frame)
            display = frame.copy()

            display[yellow > 0] = (0, 255, 255)
            display[white > 0] = (255, 0, 255)

            for x1, y1, x2, y2 in boxes:
                # 중요: 마스크나 강조된 화면이 아닌 원본에서 잘라냄
                crop = frame[y1:y2, x1:x2]
                label, score = classifier.predict(crop)

                if score >= 0.80:
                    text = f"{label}: {score:.1%}"
                    color = (0, 255, 0)
                else:
                    text = f"UNCERTAIN: {score:.1%}"
                    color = (0, 165, 255)

                cv2.rectangle(
                    display, (x1, y1), (x2, y2), color, 2
                )

                cv2.putText(
                    display,
                    text,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

            cv2.imshow("Autorace vision", display)
            cv2.imshow("Sign mask", sign_mask)
            cv2.imshow("Yellow mask", yellow)
            cv2.imshow("White mask", white)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()




# def run_camera(classifier):
#     rclpy.init()
#     node = AutoraceVision(classifier)

#     roi = None
#     result = "Select sign: press R"
#     last_inference = 0.0

#     try:
#         while rclpy.ok():
#             # ROS 메시지를 처리한 뒤 아래 화면 처리도 실행
#             rclpy.spin_once(node, timeout_sec=0.01)

#             if node.frame is None:
#                 continue

#             frame = node.frame.copy()
#             display = frame.copy()
#             now = time.monotonic()
#             fresh = now - node.received_at < 2.0

#             yellow, white = color_masks(frame)

#             # 색이 검출된 부분을 원본 영상 위에 강조 표시
#             display[yellow > 0] = (0, 255, 255)
#             display[white > 0] = (255, 0, 255)

#             if roi is not None:
#                 x, y, w, h = roi
#                 crop = frame[y:y + h, x:x + w]

#                 # 추론은 최대 약 3회/초로 제한
#                 if fresh and crop.size and now - last_inference >= 0.3:
#                     label, score = classifier.predict(crop)

#                     # 0.80은 실습용 기준이며 성능을 보며 조정
#                     if score >= 0.80:
#                         result = f"{label}: {score:.1%}"
#                     else:
#                         result = f"UNCERTAIN ({label}: {score:.1%})"

#                     last_inference = now

#                 cv2.rectangle(
#                     display,
#                     (x, y),
#                     (x + w, y + h),
#                     (0, 255, 0),
#                     2,
#                 )

#             text = result if fresh else "NO NEW CAMERA FRAME"

#             cv2.putText(
#                 display, text, (15, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.65, (0, 255, 0), 2,
#             )

#             cv2.imshow("Autorace vision", display)
#             cv2.imshow("Yellow mask", yellow)
#             cv2.imshow("White mask", white)

#             key = cv2.waitKey(1) & 0xFF

#             if key == 27:  # ESC
#                 break

#             if key == ord("r") and fresh:
#                 # 마우스로 표지판 영역을 선택하고 Enter
#                 selected = cv2.selectROI(
#                     "Select sign", frame,
#                     showCrosshair=True,
#                     fromCenter=False,
#                 )
#                 cv2.destroyWindow("Select sign")

#                 if selected[2] > 0 and selected[3] > 0:
#                     roi = tuple(int(v) for v in selected)
#                     result = "Waiting for inference"
#                     last_inference = 0.0

#             if key == ord("c"):
#                 roi = None
#                 result = "Select sign: press R"

#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         cv2.destroyAllWindows()
#         if rclpy.ok():
#             rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser()

    # --image를 주면 파일 한 장, 생략하면 ROS 카메라 사용
    parser.add_argument("--image", type=str)
    args = parser.parse_args()

    classifier = SignClassifier()

    if args.image:
        frame = cv2.imread(args.image)

        if frame is None:
            raise FileNotFoundError(args.image)

        label, score = classifier.predict(frame)
        print(f"분류: {label}, 모델 점수: {score:.1%}")

        cv2.imshow("Test image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        run_camera(classifier)


if __name__ == "__main__":
    main()
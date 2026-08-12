import time
from pathlib import Path
import cv2
import numpy as np

from src.camera import Camera
from src.gesture_classifier import GestureClassifier
from src.hand_detector import HandDetector

ROOT_DIR = Path(__file__).resolve().parent

YOLO_MODEL = ROOT_DIR / "models" / "best-yolo26s-100epochs.onnx"
GESTURE_MODEL = ROOT_DIR / "models" / "gesture_classifier_xgb.pkl"

CAMERA_ID = 0
IMG_SIZE = 800
HAND_CONF_THRESHOLD = 0.35
GESTURE_CONF_THRESHOLD = 0.5


def main():
    print("[+] Starting Camera & Models.")
    camera = Camera(CAMERA_ID)

    detector = HandDetector(
        model_path=str(YOLO_MODEL),
        imgsz=IMG_SIZE,
        conf=HAND_CONF_THRESHOLD,
    )
    classifier = GestureClassifier(str(GESTURE_MODEL))

    prev_time = time.time()
    print("[+] Starting recognition, press 'q' to quit.")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[-] Can't read data from Camera.")
                break

            # 1. Phát hiện bàn tay qua YOLO
            output, results = detector.detect(frame)

            # 2. Tách dữ liệu ra danh sách các bàn tay (N x 21 x 3)
            hands_list = detector.extract_hands_keypoints(results)

            # 3. Lặp qua từng bàn tay tìm thấy
            for idx, pts in enumerate(hands_list):
                # Dự đoán cử chỉ cho từng tay
                gesture_name, confidence = classifier.predict(pts)

                # Chuẩn bị nội dung hiển thị
                if confidence >= GESTURE_CONF_THRESHOLD:
                    text = f"Hand #{idx+1}: {gesture_name} ({confidence * 100:.0f}%)"
                    color = (0, 255, 0)  # Xanh lá
                else:
                    text = f"Hand #{idx+1}: Unknown"
                    color = (0, 0, 255)  # Đỏ

                # Vẽ Label ngay trên bàn tay
                # Tìm điểm cao nhất (min Y) của bàn tay để đặt chữ
                min_y = int(np.min(pts[:, 1]))
                avg_x = int(np.mean(pts[:, 0]))

                # Tọa độ chữ (đặt phía trên bàn tay 15px)
                text_x = max(10, avg_x - 80)
                text_y = max(25, min_y - 15)

                # Vẽ nền màu đen phía sau chữ giúp xem rõ hơn
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    2
                )
                cv2.rectangle(
                    output,
                    (text_x - 5, text_y - text_h - 5),
                    (text_x + text_w + 5, text_y + baseline),
                    (0, 0, 0),
                    -1,
                )

                # In tên cử chỉ
                cv2.putText(
                    output,
                    text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # 4. Tính toán FPS & Tổng số bàn tay
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # Hiển thị số bàn tay phát hiện được & FPS góc trên
            info_str = f"Hands: {len(hands_list)} | FPS: {int(fps)}"
            cv2.putText(
                output,
                info_str,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # Display
            cv2.imshow("Hand Detector & Gesture Recognition", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[+] Exiting.")
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
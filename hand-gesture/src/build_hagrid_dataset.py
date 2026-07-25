import csv
import os
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

HAGRID_DATASET_DIR = ROOT_DIR / "datasets" / "hagrid"
OUTPUT_CSV_PATH = ROOT_DIR / "datasets" / "hagrid_dataset.csv"

LABELS = ["dislike", "fist", "like", "one", "palm", "peace", "rock"]


def extract_hand_features(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # 1. Dời tâm về Cổ tay (Wrist - index 0)
    wrist = pts[0].copy()
    relative_pts = pts - wrist

    # 2. Chia cho khoảng cách xa nhất từ cổ tay tới các ngón để chuẩn hóa Quy mô (Scale)
    distances = np.linalg.norm(relative_pts, axis=1)
    max_distance = np.max(distances)

    if max_distance > 0:
        normalized_pts = relative_pts / max_distance
    else:
        normalized_pts = relative_pts

    return normalized_pts.flatten().tolist()


def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    # Tạo Header CSV: f_0 đến f_62 và label
    headers = [f"f_{i}" for i in range(63)] + ["label"]

    mp_hands = mp.solutions.hands
    # Static_image_mode=True để MediaPipe tối ưu phát hiện bàn tay trên ảnh tĩnh
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    total_saved = 0

    with open(OUTPUT_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for label in LABELS:
            label_dir = os.path.join(HAGRID_DATASET_DIR, label)

            if not os.path.exists(label_dir):
                print(f"[-] Bỏ qua nhãn '{label}': Không tìm thấy thư mục {label_dir}")
                continue

            image_files = [
                img
                for img in os.listdir(label_dir)
                if img.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            print(f"[+] Đang xử lý nhãn '{label}' ({len(image_files)} ảnh)...")

            saved_for_label = 0
            for img_name in image_files:
                img_path = os.path.join(label_dir, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                # MediaPipe đọc RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Trích xuất 63 đặc trưng chuẩn hóa
                        features = extract_hand_features(
                            hand_landmarks.landmark
                        )
                        row = features + [label]
                        writer.writerow(row)
                        saved_for_label += 1
                        total_saved += 1
                        break  # Chỉ lấy 1 bàn tay chính trong ảnh

            print(f"    ✓ Thu được {saved_for_label} mẫu sạch từ nhãn '{label}'")

    hands.close()
    print(
        f"\n[✓] THÀNH CÔNG! Đã trích xuất tổng cộng {total_saved} mẫu vào {OUTPUT_CSV_PATH}"
    )


if __name__ == "__main__":
    main()
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

MODEL_PATH = ROOT_DIR / "models" / "best-yolo26s-v2-finetuned.onnx"


# ==============================================================================
# 1. HÀM LETTERBOX CHỐNG MÉO ẢNH
# ==============================================================================
def letterbox(img, new_shape=(800, 800), color=(114, 114, 114)):
    """Thêm viền padded để đưa ảnh Webcam về kích thước vuông 800x800 mà KHÔNG làm co dãn bàn tay"""
    shape = img.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img


# ==============================================================================
# 2. HÀM CHUẨN HÓA & PHÂN LOẠI TAY VỚI DOT PRODUCT & MAPPING CỐ ĐỊNH
# ==============================================================================
def classify_and_remap_hand(raw_kpts):
    if raw_kpts is None or len(raw_kpts) < 21:
        return raw_kpts, "Hand"

    pts = np.array(raw_kpts, dtype=np.float32)

    # Bảng remap khớp khi YOLO bị gán nhầm chỉ số ngón tay
    right_hand_mapping = {
        0: 0,
        1: 1,
        2: 2,
        3: 4,
        4: 3,  # Ngón cái
        5: 10,
        6: 11,
        7: 12,
        8: 13,  # Ngón trỏ
        9: 14,
        10: 5,
        11: 6,
        12: 7,  # Ngón giữa
        13: 8,
        14: 9,
        15: 15,
        16: 16,  # Ngón áp út
        17: 17,
        18: 18,
        19: 19,
        20: 20,  # Ngón út
    }

    # KIỂM TRA TÌNH TRẠNG CHÉO DÂY KEYPOINTS:
    # Vector hướng ngón trỏ (5 -> 8) và ngón giữa (9 -> 12) từ raw YOLO
    v_index = pts[8][:2] - pts[5][:2]
    v_middle = pts[12][:2] - pts[9][:2]

    # Tính độ tương đồng góc (dot product) giữa ngón trỏ và ngón giữa
    dot_fingers = np.dot(v_index, v_middle)

    # Nếu 2 ngón bị đan chéo/ngược hướng nhau (dot product < 0) -> Remap lại về chuẩn
    if dot_fingers < 0:
        remapped_pts = np.zeros_like(pts)
        for mp_idx, yolo_idx in right_hand_mapping.items():
            remapped_pts[mp_idx] = pts[yolo_idx]
        return remapped_pts, "Hand"

    # Nếu keypoints đã chuẩn sẵn -> Giữ nguyên
    return pts, "Hand"


# ==============================================================================
# 3. ĐỊNH NGHĨA KHUNG XƯƠNG CHUẨN MEDIAPIPE (0->20)
# ==============================================================================
MP_HAND_CONNECTIONS = [
    # Cổ tay & Ngón cái
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Ngón trỏ
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Ngón giữa
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # Ngón áp út
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # Ngón út
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    # Nối lòng bàn tay
    (5, 9),
    (9, 13),
    (13, 17),
]


# ==============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================
def main():
    model_path = MODEL_PATH

    print(f"🚀 Loading model: {model_path}")
    model = YOLO(model_path, task="pose")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Lỗi: Không mở được Webcam!")
        return

    print("\n" + "=" * 50)
    print("BẮT ĐẦU CHẠY DEBUG POSE - THUẬT TOÁN DOT PRODUCT & MAPPING CỐ ĐỊNH")
    print("Nhấn 'q' để thoát.")
    print("=" * 50 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Lật ảnh ngang để trùng góc nhìn gương thực tế
        frame = cv2.flip(frame, 1)

        # 2. Áp dụng letterbox chống méo tỷ lệ
        padded_frame = letterbox(frame, new_shape=800)

        # 3. Dự đoán với YOLO Pose
        results = model.predict(source=padded_frame, conf=0.40, verbose=False)

        for result in results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                for kpts_raw in result.keypoints.xy:
                    raw_kpts = kpts_raw.cpu().numpy()

                    if len(raw_kpts) >= 21:
                        # 👉 Phân loại/Sắp xếp lại keypoints bằng Dot Product & Mapping cố định
                        kpts, hand_type = classify_and_remap_hand(raw_kpts)

                        # A. Vẽ các đường nối khung xương (Skeleton lines)
                        for p1, p2 in MP_HAND_CONNECTIONS:
                            x1, y1 = int(kpts[p1][0]), int(kpts[p1][1])
                            x2, y2 = int(kpts[p2][0]), int(kpts[p2][1])

                            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                                cv2.line(
                                    padded_frame,
                                    (x1, y1),
                                    (x2, y2),
                                    (255, 255, 255),
                                    2,
                                    cv2.LINE_AA,
                                )

                        # B. Vẽ các điểm khớp đỏ và in số thứ tự Index màu vàng (0 -> 20)
                        for i, pt in enumerate(kpts):
                            x, y = int(pt[0]), int(pt[1])
                            if x > 0 and y > 0:
                                radius = 6 if i in [0, 5, 17] else 4
                                color = (
                                    (0, 255, 0)
                                    if i in [0, 5, 17]
                                    else (0, 0, 255)
                                )

                                cv2.circle(
                                    padded_frame, (x, y), radius, color, -1
                                )
                                cv2.putText(
                                    padded_frame,
                                    str(i),
                                    (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.45,
                                    (0, 255, 255),
                                    1,
                                    cv2.LINE_AA,
                                )

                        # C. Hiển thị thông tin
                        cv2.putText(
                            padded_frame,
                            f"Detected: {hand_type}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

        cv2.imshow("Debug Pose - Dot Product & Fixed Mapping", padded_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
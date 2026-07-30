from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

class HandDetector:

    def __init__(
        self, 
        model_path: str, 
        imgsz: int = 640, 
        conf: float = 0.35
    ):
        print(f"[+] Loading YOLO model from {model_path}.")
        self.model = YOLO(model_path, task="pose")
        self.imgsz = imgsz
        self.conf = conf

        # Bảng kết nối khung xương chuẩn MediaPipe (0->20)
        self.connections = [
            (0, 1),(1, 2),(2, 3),(3, 4),        # Ngón cái
            (0, 5),(5, 6),(6, 7),(7, 8),        # Ngón trỏ
            (0, 9),(9, 10),(10, 11),(11, 12),   # Ngón giữa
            (0, 13),(13, 14),(14, 15),(15, 16), # Ngón áp út
            (0, 17),(17, 18),(18, 19),(19, 20), # Ngón út
            (5, 9),(9, 13),(13, 17),            # Lòng bàn tay
        ]

        # Bảng remap khớp khi YOLO bị gán nhầm chỉ số ngón tay
        self.right_hand_mapping = {
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

    def letterbox(self, img, color=(114, 114, 114)):
        """Thêm viền padded đưa ảnh về kích thước vuông imgsz x imgsz mà không làm méo tỷ lệ."""
        shape = img.shape[:2]
        r = min(self.imgsz / shape[0], self.imgsz / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = self.imgsz - new_unpad[0], self.imgsz - new_unpad[1]

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    def classify_and_remap_hand(self, raw_kpts):
        """Kiểm tra độ tương đồng hướng (Dot Product) ngón trỏ & giữa để remap keypoint nếu bị chéo dây."""
        if raw_kpts is None or len(raw_kpts) < 21:
            return raw_kpts

        pts = np.array(raw_kpts, dtype=np.float32)

        # Vector hướng ngón trỏ (5 -> 8) và ngón giữa (9 -> 12)
        v_index = pts[8][:2] - pts[5][:2]
        v_middle = pts[12][:2] - pts[9][:2]

        # Dot product kiểm tra góc nghiêng/chéo
        dot_fingers = np.dot(v_index, v_middle)

        # Nếu bị chéo dây (ngược hướng nhau) -> Remap lại
        if dot_fingers < 0:
            remapped_pts = np.zeros_like(pts)
            for mp_idx, yolo_idx in self.right_hand_mapping.items():
                remapped_pts[mp_idx] = pts[yolo_idx]
            return remapped_pts

        return pts

    def detect(self, frame):
        """Phát hiện bàn tay, chuẩn hóa keypoint và vẽ khung xương."""
        padded_frame = self.letterbox(frame)
        results = self.model.predict(
            source=padded_frame, 
            conf=self.conf, 
            verbose=False
        )

        hands_keypoints = []

        for result in results:
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                for kpts_raw in result.keypoints.xy:
                    raw_kpts = kpts_raw.cpu().numpy()

                    if len(raw_kpts) >= 21:
                        # Áp dụng Dot Product & Mapping
                        remapped_kpts = self.classify_and_remap_hand(raw_kpts[:21])
                        hands_keypoints.append(remapped_kpts)

                        # A. Vẽ các đường nối khung xương
                        for p1, p2 in self.connections:
                            x1, y1 = int(remapped_kpts[p1][0]), int(remapped_kpts[p1][1])
                            x2, y2 = int(remapped_kpts[p2][0]), int(remapped_kpts[p2][1])

                            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                                cv2.line(
                                    padded_frame,
                                    (x1, y1),
                                    (x2, y2),
                                    (255, 255, 255),
                                    2,
                                    cv2.LINE_AA,
                                )

                        # B. Vẽ các điểm khớp và chỉ số Index
                        for i, pt in enumerate(remapped_kpts):
                            x, y = int(pt[0]), int(pt[1])
                            if x > 0 and y > 0:
                                radius = 6 if i in [0, 5, 17] else 4
                                color = (
                                    (0, 255, 0)
                                    if i in [0, 5, 17]
                                    else (0, 0, 255)
                                )

                                cv2.circle(padded_frame, (x, y), radius, color, -1)
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

        return padded_frame, hands_keypoints

    def extract_hands_keypoints(self, result):
        if result is None or len(result) == 0:
            return []

        pts_all = np.array(result, dtype=np.float32)
        hands = []

        # Trường hợp 1: Chỉ phát hiện 1 bàn tay -> shape (21, 2) hoặc (21, 3)
        if pts_all.ndim == 2:
            if pts_all.shape == (21, 2):
                z_zero = np.zeros((21, 1), dtype=np.float32)
                pts_all = np.hstack([pts_all, z_zero])
            if pts_all.shape == (21, 3):
                hands.append(pts_all)

        # Trường hợp 2: Phát hiện nhiều bàn tay (2, 3, 4...) -> shape (N, 21, 2) hoặc (N, 21, 3)
        elif pts_all.ndim == 3:
            for single_hand in pts_all:
                if single_hand.shape == (21, 2):
                    z_zero = np.zeros((21, 1), dtype=np.float32)
                    single_hand = np.hstack([single_hand, z_zero])
                if single_hand.shape == (21, 3):
                    hands.append(single_hand)

        return hands
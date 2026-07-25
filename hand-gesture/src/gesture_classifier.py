import joblib
import numpy as np

class GestureClassifier:

    def __init__(self, model_path):
        data = joblib.load(model_path)
        self.scaler = data["scaler"]
        self.model = data["model"]
        self.label_encoder = data.get("label_encoder", None)

    def extract_features(self, landmarks):
        pts = np.array(landmarks, dtype=np.float32)

        # Nếu đầu vào chỉ có 2D (21, 2) -> Tự động thêm z = 0
        if pts.shape == (21, 2):
            z_zero = np.zeros((21, 1), dtype=np.float32)
            pts = np.hstack([pts, z_zero])

        # 1. Tọa độ chuẩn hóa (63 đặc trưng)
        wrist = pts[0].copy()
        relative_pts = pts - wrist
        max_dist = np.max(np.linalg.norm(relative_pts, axis=1))
        norm_pts = (
            relative_pts / max_dist if max_dist > 0 else relative_pts
        ).flatten()

        # 2. Góc gập 5 ngón tay (5 đặc trưng)
        finger_joints = [
            (2, 3, 4),     # Ngón cái
            (6, 7, 8),     # Ngón trỏ
            (10, 11, 12),  # Ngón giữa
            (14, 15, 16),  # Ngón áp út
            (18, 19, 20),  # Ngón út
        ]

        angles = []
        for a, b, c in finger_joints:
            v1 = pts[a] - pts[b]
            v2 = pts[c] - pts[b]
            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
            )
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(angle)

        # 3. Khoảng cách giữa các đầu ngón tay (10 đặc trưng)
        tips = [4, 8, 12, 16, 20]
        tip_distances = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                d = np.linalg.norm(pts[tips[i]] - pts[tips[j]]) / (
                    max_dist + 1e-6
                )
                tip_distances.append(d)

        # Ghép thành 1 vector 78 chiều
        return np.hstack([norm_pts, angles, tip_distances])

    def predict(self, landmarks):
        feats = self.extract_features(landmarks)
        feats_scaled = self.scaler.transform([feats])

        # Dự đoán
        probs = self.model.predict_proba(feats_scaled)[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])

        # Decode tên cử chỉ nếu có label_encoder
        if self.label_encoder is not None:
            gesture_name = self.label_encoder.inverse_transform([pred_idx])[0]
        else:
            gesture_name = str(self.model.classes_[pred_idx])

        return gesture_name, confidence
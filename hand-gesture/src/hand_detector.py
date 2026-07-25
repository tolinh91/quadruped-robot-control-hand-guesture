import time
import cv2
import numpy as np
from ultralytics import YOLO


class HandDetector:

    def __init__(
        self,
        model_path="best.onnx",
        imgsz=640,
        conf=0.5,
    ):
        print(f"[+] Loading YOLO model from {model_path}.")
        self.model = YOLO(model_path, task="pose")
        self.imgsz = imgsz
        self.conf = conf

    def detect(self, frame):
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
        )

        result_obj = results[0]

        # 1. Vẽ bounding box & keypoints từ YOLO lên khung hình
        output = result_obj.plot()

        # 2. Bóc tách keypoints cho TẤT CẢ bàn tay phát hiện được
        all_keypoints = []

        if result_obj.keypoints is not None and len(result_obj.keypoints) > 0:
            # .xy trả về Tensor shape (N, 21, 2) với N là số lượng bàn tay
            kps_tensor = result_obj.keypoints.xy.cpu().numpy()

            for kps_single in kps_tensor:
                # Kiểm tra đủ 21 điểm keypoint của một bàn tay
                if kps_single.shape[0] == 21:
                    all_keypoints.append(kps_single)

        # Trả về: (ảnh đã vẽ skeleton, danh sách mảng keypoints của các bàn tay)
        return output, all_keypoints
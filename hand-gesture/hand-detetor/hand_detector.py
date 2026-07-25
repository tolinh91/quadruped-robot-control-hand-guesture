import time
import cv2
from ultralytics import YOLO


class HandDetector:

    def __init__(
        self,
        model_path="best.onnx",
        imgsz=640,
        conf=0.5,
    ):

        print("Loading model...")
        self.model = YOLO(model_path, task="pose")

        self.imgsz = imgsz
        self.conf = conf

        self.prev_time = time.time()

    def detect(self, frame):

        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
        )

        result = results[0]

        output = result.plot()

        current_time = time.time()

        fps = 1 / (current_time - self.prev_time)

        self.prev_time = current_time

        cv2.putText(
            output,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return output, result
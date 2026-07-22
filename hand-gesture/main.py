import cv2

from camera import Camera
from hand_detector import HandDetector

MODEL = "best.pt"
CAMERA_ID = 0
IMG_SIZE = 640
CONFIDENCE = 0.5

camera = Camera(CAMERA_ID)

detector = HandDetector(
    model_path=MODEL,
    imgsz=IMG_SIZE,
    conf=CONFIDENCE,
)

while True:

    ret, frame = camera.read()

    if not ret:
        break

    output, result = detector.detect(frame)

    cv2.imshow("YOLO Hand Pose", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
import cv2
import sys

class Camera:

    def __init__(self, camera_id=0, width=1280, height=720, fps=30):
        print(f"[+] Openning Camera ID: {camera_id}")
        # OpenCV 5 may try the OBSensor backend first on Linux.  USB/UVC
        # webcams exposed by WSL via usbipd are V4L2 devices, so select that
        # backend explicitly instead of treating the index as an OBSensor ID.
        if sys.platform.startswith("linux"):
            # Open the concrete device node.  This is more reliable than a
            # numeric index for USB/IP forwarded UVC cameras under WSL.
            self.device = f"/dev/video{camera_id}"
            self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        else:
            self.device = camera_id
            self.cap = cv2.VideoCapture(camera_id, cv2.CAP_ANY)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"[-] Can't connect camera {self.device}. "
                "Check USB/IP attachment, V4L2 capabilities, and video-group access."
            )

        # 1. Cấu hình độ phân giải và tốc độ khung hình
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        # 2. Giảm bộ nhớ đệm (buffer) về 1 để tránh độ trễ (lag/delay) hình ảnh
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self, mirror=True):
        ret, frame = self.cap.read()

        if ret and frame is not None:
            if mirror:
                # Lật ngang ảnh (flipCode = 1) giúp như soi gương
                frame = cv2.flip(frame, 1)

        return ret, frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
            print("[+] Release Camera.")

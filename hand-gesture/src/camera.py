import cv2

class Camera:

    def __init__(self, camera_id=0, width=1280, height=720, fps=30):
        print(f"[+] Openning Camera ID: {camera_id}")
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"[-] Can't connect Camera ID {camera_id}.")

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
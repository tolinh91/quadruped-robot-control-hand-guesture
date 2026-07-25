import cv2


class Camera:

    def __init__(self, camera_id=0):
        print("Opening camera...")
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError("Can't open camera!")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
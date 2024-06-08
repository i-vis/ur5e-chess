import cv2
import threading
import time

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.fps = 5
        self.cap = cv2.VideoCapture(name)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            with self.lock:
                ret = self.cap.grab()
            if not ret:
                break
            time.sleep(1/self.fps)

    # retrieve latest frame
    def read(self):
        with self.lock:
            ret, frame = self.cap.retrieve()
        return ret, frame

import numpy as np
import cv2
import mss

class ScreenCapture:
    def __init__(self , monitor_index):
        self.screen = mss.mss()
        self.capture_area = self.screen.monitors[monitor_index]

    def capture_frame(self):
        screenshot = self.screen.grab(self.capture_area)
        frame = np.array(screenshot) #[h , w , ch = 4(BGRA)]ndarray
        frame = cv2.cvtColor(frame , cv2.COLOR_BGRA2BGR) #[h , w , ch = 3(BGR)]ndarray
        return frame
    
    def native_resolution(self):
        return self.capture_area["width"] , self.capture_area["height"]

    def show_frame(self , frame):
        cv2.imshow("Screen Capture" , frame)

    def close(self):
        self.screen.close()
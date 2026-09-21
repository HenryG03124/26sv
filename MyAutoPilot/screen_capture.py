import numpy as np
import cv2
import mss
import ctypes
from ctypes import wintypes

class ScreenCapture:
    def __init__(self , monitor_index):
        self.screen = mss.mss()
        self.capture_area = self.screen.monitors[monitor_index]
        self.user32 = ctypes.WinDLL("user32" , use_last_error = True)
        user32 = self.user32
        user32.FindWindowW.argtypes = [wintypes.LPCWSTR , wintypes.LPCWSTR]
        user32.FindWindowW.restype = wintypes.HWND
        user32.GetClientRect.argtypes = [wintypes.HWND , ctypes.POINTER(wintypes.RECT)]
        user32.GetClientRect.restype = wintypes.BOOL
        user32.ClientToScreen.argtypes = [wintypes.HWND , ctypes.POINTER(wintypes.POINT)]
        user32.ClientToScreen.restype = wintypes.BOOL
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.IsIconic.restype = wintypes.BOOL

    def capture_window(self , window_name = "Euro Truck Simulator 2"):
        user32 = self.user32
        window = user32.FindWindowW(None , window_name)
        if not window:
            raise RuntimeError(f"Window not found: {window_name}")
        if user32.IsIconic(window):
            raise RuntimeError(f"Window is minimized: {window_name}")

        rect = wintypes.RECT()
        position = wintypes.POINT(0 , 0)
        if not user32.GetClientRect(window , ctypes.byref(rect)) or not user32.ClientToScreen(window , ctypes.byref(position)):
            raise ctypes.WinError(ctypes.get_last_error())
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Window has no capture area: {window_name}")

        # MSS captures visible screen pixels; keep the game unobstructed.
        area = {"left": position.x , "top": position.y , "width": width , "height": height}
        screenshot = self.screen.grab(area)
        frame = np.array(screenshot)
        return cv2.cvtColor(frame , cv2.COLOR_BGRA2BGR)

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

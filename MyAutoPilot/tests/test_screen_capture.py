import unittest
from unittest.mock import Mock , patch

import numpy as np

from MyAutoPilot.screen_capture import ScreenCapture


class ScreenCaptureTests(unittest.TestCase):
    def setUp(self):
        self.screen = Mock()
        self.screen.monitors = [{}, {"left": 0 , "top": 0 , "width": 3840 , "height": 2160}]
        self.user32 = Mock()
        self.user32.FindWindowW.return_value = 0x100000001
        self.user32.IsIconic.return_value = False
        self.rect = (1280 , 720)
        self.position = (120 , 80)

        def get_rect(window , rect):
            rect._obj.right , rect._obj.bottom = self.rect
            return True

        def get_position(window , position):
            position._obj.x , position._obj.y = self.position
            return True

        def grab(area):
            frame = np.empty((area["height"] , area["width"] , 4) , dtype = np.uint8)
            frame[:] = [10 , 20 , 30 , 255]
            return frame

        self.user32.GetClientRect.side_effect = get_rect
        self.user32.ClientToScreen.side_effect = get_position
        self.screen.grab.side_effect = grab
        factory = patch("MyAutoPilot.screen_capture.mss.mss" , return_value = self.screen)
        factory.start()
        self.addCleanup(factory.stop)
        library = patch("MyAutoPilot.screen_capture.ctypes.WinDLL" , return_value = self.user32 , create = True)
        library.start()
        self.addCleanup(library.stop)
        self.capture = ScreenCapture(1)
        self.addCleanup(self.capture.close)

    def test_exact_title_client_area_and_bgr_output(self):
        frame = self.capture.capture_window()
        self.user32.FindWindowW.assert_called_once_with(None , "Euro Truck Simulator 2")
        self.screen.grab.assert_called_once_with({"left": 120 , "top": 80 , "width": 1280 , "height": 720})
        self.assertEqual(frame.shape , (720 , 1280 , 3))
        np.testing.assert_array_equal(frame[0 , 0] , [10 , 20 , 30])

    def test_move_resize_and_negative_monitor_coordinates(self):
        self.capture.capture_window()
        self.position = (-1800 , 200)
        self.rect = (960 , 540)
        frame = self.capture.capture_window()
        self.screen.grab.assert_called_with({"left": -1800 , "top": 200 , "width": 960 , "height": 540})
        self.assertEqual(frame.shape , (540 , 960 , 3))
        self.assertEqual(self.capture.native_resolution() , (3840 , 2160))

    def test_missing_window_does_not_fall_back_to_desktop(self):
        self.user32.FindWindowW.return_value = 0
        with self.assertRaisesRegex(RuntimeError , "Window not found"):
            self.capture.capture_window()
        self.screen.grab.assert_not_called()

    def test_minimized_window_is_not_captured(self):
        self.user32.IsIconic.return_value = True
        with self.assertRaisesRegex(RuntimeError , "minimized"):
            self.capture.capture_window()
        self.screen.grab.assert_not_called()

    def test_empty_client_area_is_not_captured(self):
        self.rect = (0 , 0)
        with self.assertRaisesRegex(RuntimeError , "no capture area"):
            self.capture.capture_window()
        self.screen.grab.assert_not_called()

    def test_original_full_screen_capture_is_preserved(self):
        self.capture.capture_window()
        self.capture.capture_frame()
        self.screen.grab.assert_called_with(self.screen.monitors[1])


if __name__ == "__main__":
    unittest.main()

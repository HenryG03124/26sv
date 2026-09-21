import unittest
from unittest.mock import patch

import numpy as np

from MyAutoPilot.draw import Draw


class DrawStatusTests(unittest.TestCase):
    def test_status_text_is_right_aligned_and_separated(self):
        frame = np.zeros((352 , 640 , 3) , dtype = np.uint8)
        with patch("MyAutoPilot.draw.cv2.putText") as put_text:
            Draw.show_fps(frame , 7.1)
            Draw.show_speed(frame , {"speed_kmh": 45.2 , "status": "ok"})
        calls = put_text.call_args_list
        self.assertEqual(len(calls) , 2)
        for call in calls:
            self.assertEqual(call.args[5] , (0 , 255 , 255))
            self.assertEqual(call.args[6] , 1)
        self.assertEqual(calls[0].args[1] , "FPS: 7.1")
        self.assertEqual(calls[1].args[1] , "SPEED: 45.2 km/h")
        self.assertGreater(calls[0].args[2][0] , 320)
        self.assertEqual(calls[0].args[2][1] , 26)
        self.assertEqual(calls[1].args[2][1] , 52)

    def test_missing_telemetry_is_not_displayed_as_zero_speed(self):
        frame = np.zeros((352 , 640 , 3) , dtype = np.uint8)
        with patch("MyAutoPilot.draw.cv2.putText") as put_text:
            Draw.show_speed(frame , {"speed_kmh": None , "status": "missing"})
            Draw.show_fps(frame , None)
        self.assertEqual(put_text.call_args_list[0].args[1] , "SPEED: -- (missing)")
        self.assertEqual(put_text.call_args_list[1].args[1] , "FPS: --")

    def test_text_does_not_draw_a_dark_outline(self):
        frame = np.full((352 , 640 , 3) , 100 , dtype = np.uint8)
        Draw.show_fps(frame , 12.8)
        Draw.show_speed(frame , {"speed_kmh": 0.0 , "status": "ok"})
        self.assertTrue(np.any(frame[: , : , 1] > 100))
        self.assertTrue(np.all(frame[: , : , 1:] >= 100))


if __name__ == "__main__":
    unittest.main()

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock , patch

from MyAutoPilot.throttle import Throttle


class ThrottleTests(unittest.TestCase):
    def test_fifth_tracking_frame_enables_throttle(self):
        controller = Throttle()
        telemetry = {"status": "ok" , "speed_kmh": 20}
        for frames in range(5):
            self.assertEqual(controller.throttle(telemetry , frames , 0 , frames * 0.1) , 0)
        self.assertGreater(controller.throttle(telemetry , 5 , 0 , 0.5) , 0)

    def test_output_is_finite_and_bounded(self):
        controller = Throttle()
        for step , speed in enumerate((0 , 10 , 25 , 45 , 50 , 60 , 100)):
            with self.subTest(speed = speed):
                value = controller.throttle({"status": "ok" , "speed_kmh": speed} , 5 , 0 , step * 0.1)
                self.assertTrue(math.isfinite(value))
                self.assertGreaterEqual(value , 0)
                self.assertLessEqual(value , 1)

    def test_overspeed_releases_throttle(self):
        controller = Throttle()
        for step in range(100):
            controller.throttle({"status": "ok" , "speed_kmh": 45} , 5 , 0 , step * 0.1)
        value = controller.throttle({"status": "ok" , "speed_kmh": 60} , 5 , 0 , 10)
        self.assertEqual(value , 0)

    def test_invalid_telemetry_disables_throttle(self):
        telemetry_values = (
            {"status": "disconnected" , "speed_kmh": 20} ,
            {"status": "paused" , "speed_kmh": 20} ,
            {"status": "ok" , "speed_kmh": None} ,
            {"status": "ok" , "speed_kmh": float("nan")} ,
            {"status": "ok" , "speed_kmh": float("inf")} ,
            {"status": "ok" , "speed_kmh": -1}
        )
        for telemetry in telemetry_values:
            with self.subTest(telemetry = telemetry):
                controller = Throttle()
                controller.throttle({"status": "ok" , "speed_kmh": 20} , 5 , 0 , 0)
                self.assertEqual(controller.throttle(telemetry , 5 , 0 , 0.1) , 0)

    def test_braking_or_lost_tracking_clears_integral(self):
        telemetry = {"status": "ok" , "speed_kmh": 45}
        for frames , brake in ((0 , 0) , (4 , 0) , (5 , 0.01) , (5 , 1)):
            with self.subTest(frames = frames , brake = brake):
                controller = Throttle()
                for step in range(50):
                    controller.throttle(telemetry , 5 , 0 , step * 0.1)
                self.assertEqual(controller.throttle(telemetry , frames , brake , 5) , 0)
                recovered = controller.throttle(telemetry , 5 , 0 , 5.1)
                fresh = Throttle().throttle(telemetry , 5 , 0 , 5.1)
                self.assertAlmostEqual(recovered , fresh)

    def test_invalid_telemetry_clears_integral(self):
        controller = Throttle()
        telemetry = {"status": "ok" , "speed_kmh": 45}
        for step in range(50):
            controller.throttle(telemetry , 5 , 0 , step * 0.1)
        invalid = {"status": "disconnected" , "speed_kmh": 45}
        self.assertEqual(controller.throttle(invalid , 5 , 0 , 5) , 0)
        recovered = controller.throttle(telemetry , 5 , 0 , 5.1)
        fresh = Throttle().throttle(telemetry , 5 , 0 , 5.1)
        self.assertAlmostEqual(recovered , fresh)

    def test_integral_keeps_throttle_at_target_speed(self):
        controller = Throttle()
        for step in range(50):
            controller.throttle({"status": "ok" , "speed_kmh": 45} , 5 , 0 , step * 0.1)
        value = controller.throttle({"status": "ok" , "speed_kmh": 50} , 5 , 0 , 5)
        self.assertGreater(value , 0)
        self.assertLessEqual(value , 1)

    def test_saturation_does_not_accumulate_full_throttle(self):
        controller = Throttle()
        for step in range(1000):
            controller.throttle({"status": "ok" , "speed_kmh": 0} , 5 , 0 , step * 0.1)
        value = controller.throttle({"status": "ok" , "speed_kmh": 50} , 5 , 0 , 100)
        self.assertLess(value , 0.25)

    def test_integral_removes_speed_error_with_drag(self):
        # Synthetic dynamics only; this does not validate in-game control.
        controller = Throttle()
        speed = 0
        for step in range(2000):
            value = controller.throttle({"status": "ok" , "speed_kmh": speed} , 5 , 0 , step * 0.1)
            speed += (5 * value - 0.03 * speed) * 0.1
        self.assertAlmostEqual(speed , 50 , delta = 1)
        self.assertGreater(value , 0.1)
        self.assertLess(value , 0.8)


class ThrottleInputControllerTests(unittest.TestCase):
    def test_throttle_uses_right_trigger_and_clamps_output(self):
        gamepad = Mock()
        vgamepad = types.SimpleNamespace(VX360Gamepad = Mock(return_value = gamepad))
        spec = importlib.util.spec_from_file_location("input_controller_under_test" , Path(__file__).resolve().parents[1] / "input_controller.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules , {"vgamepad": vgamepad}):
            spec.loader.exec_module(module)
            controller = module.InputController()
        for value , expected in ((-1 , 0) , (0.4 , 0.4) , (2 , 1)):
            with self.subTest(value = value):
                gamepad.reset_mock()
                controller.throttle_controller(value)
                gamepad.right_trigger_float.assert_called_once_with(value_float = expected)
                gamepad.update.assert_called_once_with()
                gamepad.left_trigger_float.assert_not_called()
                gamepad.left_joystick_float.assert_not_called()


if __name__ == "__main__":
    unittest.main()

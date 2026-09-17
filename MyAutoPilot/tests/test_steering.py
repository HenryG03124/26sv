import math
import unittest
from unittest.mock import patch

from MyAutoPilot import steering as steering_module
from MyAutoPilot.steering import Steering


class SteeringTests(unittest.TestCase):
    def setUp(self):
        Steering.reset()
        self.now = 10.0
        clock = patch.object(steering_module.time , "monotonic" , side_effect = lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.addCleanup(Steering.reset)

    def step(self , points , previous = 0.0 , dt = 0.1 , gain = 3 , y_far = 220):
        self.now += dt
        return Steering.steering(points , y_far , gain , previous)

    def trajectory(self , points , fps = 20 , seconds = 1.0 , previous = 0.0 , initial_points = None):
        # Start every independent scenario with the same history and clock.
        Steering.reset()
        self.now = 10.0
        self.step([[320 , 220] , [320 , 320]] if initial_points is None else initial_points)
        values = []
        for _ in range(round(seconds * fps)):
            previous = self.step(points , previous , 1 / fps)
            values.append(previous)
        return values

    def build_offset_history(self):
        previous = 0.0
        for x in (320 , 323 , 326 , 329 , 332):
            previous = self.step([[x , 220] , [x , 320]] , previous)
        self.assertGreater(Steering._offset_rate , 0)
        return previous

    def test_screenshot_target_left_returns_before_reversing(self):
        points = [[283 , 300] , [313 , 200]]
        previous = 0.06
        first = self.step(points , previous , y_far = 200)
        self.assertGreater(first , 0)
        self.assertLess(first , previous)
        previous = first
        for _ in range(5):
            value = self.step(points , previous , y_far = 200)
            self.assertLessEqual(abs(value - previous) , steering_module.return_rate * 0.1 + 1e-9)
            previous = value
        self.assertLess(previous , 0)

    def test_left_and_right_targets_are_symmetric(self):
        left = self.trajectory([[313 , 220] , [283 , 320]])
        right = self.trajectory([[327 , 220] , [357 , 320]])
        self.assertLess(left[-1] , 0)
        self.assertGreater(right[-1] , 0)
        for left_value , right_value in zip(left , right):
            self.assertAlmostEqual(left_value , -right_value)

    def test_centered_target(self):
        values = self.trajectory([[320 , 220] , [320 , 320]])
        self.assertTrue(all(value == 0 for value in values))

    def test_large_near_offset_takes_priority_over_far_preview(self):
        for near_x , far_x , direction in ((280 , 400 , -1) , (360 , 240 , 1)):
            with self.subTest(near_x = near_x):
                value = self.trajectory([[far_x , 220] , [near_x , 320]])[-1]
                self.assertGreater(value * direction , 0)

    def test_building_steering_respects_time_based_rate_limit(self):
        for fps in (10 , 20 , 60):
            with self.subTest(fps = fps):
                previous = 0.0
                values = self.trajectory([[400 , 220] , [400 , 320]] , fps = fps)
                for value in values:
                    self.assertLessEqual(abs(value - previous) , steering_module.steering_rate / fps + 1e-9)
                    previous = value

    def test_output_stays_bounded(self):
        # Both paths are valid, so this exercises actual steering saturation.
        for target_x in (0 , 639):
            with self.subTest(target_x = target_x):
                values = self.trajectory([[target_x , 220] , [target_x , 320]] , seconds = 2)
                self.assertTrue(all(abs(value) <= steering_module.max_steering for value in values))
                self.assertAlmostEqual(abs(values[-1]) , steering_module.max_steering)

    def test_invalid_paths_return_at_the_same_rate_across_frame_rates(self):
        invalid_paths = (
            [] , [[320 , 320]] , [[320 , 221] , [320 , 320]] ,
            [[320 , 220] , [320 , 351]] , [[640 , 220] , [320 , 320]] ,
            [[float("nan") , 220] , [320 , 320]]
        )
        for points in invalid_paths:
            for fps in (10 , 20 , 60):
                with self.subTest(points = points , fps = fps):
                    value = self.trajectory(points , fps = fps , seconds = 0.1 , previous = 0.18)[-1]
                    self.assertAlmostEqual(value , 0.18 - steering_module.return_rate * 0.1)

    def test_invalid_gains_preserve_the_return_clock(self):
        for gain in (0 , -1 , float("nan") , float("inf")):
            with self.subTest(gain = gain):
                Steering.reset()
                self.step([[320 , 220] , [320 , 320]])
                previous = 0.18
                for _ in range(6):
                    previous = self.step([[340 , 220] , [340 , 320]] , previous , 1 / 60 , gain)
                self.assertAlmostEqual(previous , 0.18 - steering_module.return_rate * 0.1)

    def test_return_reaches_neutral_without_overshoot(self):
        for previous in (-0.18 , 0.18):
            with self.subTest(previous = previous):
                values = self.trajectory([] , previous = previous)
                self.assertTrue(all(value * previous >= 0 for value in values))
                self.assertEqual(values[-1] , 0)
                for before , after in zip([previous] + values , values):
                    self.assertLessEqual(abs(after) , abs(before))

    def test_invalid_path_clears_tracking_but_preserves_clock(self):
        previous = self.build_offset_history()
        self.step([] , previous)
        self.assertIsNone(Steering._prev_offset)
        self.assertEqual(Steering._offset_rate , 0)
        self.assertEqual(Steering._offset_history , [])
        self.assertEqual(Steering._prev_time , self.now)

    def test_reacquired_path_matches_fresh_tracking(self):
        previous = self.build_offset_history()
        previous = self.step([] , previous)
        points = [[340 , 220] , [340 , 320]]
        recovered = self.step(points , previous)
        Steering.reset()
        fresh = self.step(points , previous)
        self.assertAlmostEqual(recovered , fresh)

    def test_path_switch_returns_toward_neutral_without_derivative_spike(self):
        previous = self.trajectory([[340 , 220] , [340 , 320]])[-1]
        switched = self.step([[500 , 220] , [500 , 320]] , previous)
        self.assertLess(abs(switched) , abs(previous))
        self.assertEqual(Steering._offset_rate , 0)

    def test_constant_path_rate_limits_are_consistent_across_frame_rates(self):
        # Start with an already observed path, so this measures the output clock
        # without introducing a sampled offset jump into the derivative history.
        points = [[340 , 220] , [340 , 320]]
        traces = {fps: self.trajectory(points , fps = fps , initial_points = points) for fps in (10 , 20 , 60)}
        for tenth in range(1 , 11):
            values = [trace[round(fps * tenth / 10) - 1] for fps , trace in traces.items()]
            with self.subTest(seconds = tenth / 10):
                self.assertLess(max(values) - min(values) , 1e-9)

    def test_long_gap_discards_old_derivative_history(self):
        previous = self.build_offset_history()
        points = [[340 , 220] , [340 , 320]]
        after_gap = self.step(points , previous , dt = 1.0)
        Steering.reset()
        fresh = self.step(points , previous)
        self.assertAlmostEqual(after_gap , fresh)

    def test_small_near_pixel_jitter_does_not_create_steering(self):
        previous = 0.0
        for near_x in (319 , 320 , 321 , 320) * 10:
            previous = self.step([[320 , 220] , [near_x , 320]] , previous , 1 / 20)
            self.assertEqual(previous , 0)

    def test_nonfinite_previous_output_is_sanitized(self):
        for previous in (float("nan") , float("inf") , -float("inf")):
            with self.subTest(previous = previous):
                Steering.reset()
                value = self.step([[340 , 220] , [340 , 320]] , previous)
                self.assertTrue(math.isfinite(value))
                self.assertLessEqual(abs(value) , steering_module.max_steering)

    def test_endpoint_validity_and_missing_path_reset(self):
        self.assertTrue(Steering.existence_filter([[313 , 188] , [283 , 300]] , 188 , 300))
        for points in (None , [] , [[283 , 300]] , [[640 , 188] , [283 , 300]]):
            with self.subTest(points = points):
                self.build_offset_history()
                self.assertFalse(Steering.existence_filter(points , 188 , 300))
                self.assertIsNone(Steering._prev_time)
                self.assertIsNone(Steering._prev_offset)
                self.assertEqual(Steering._offset_rate , 0)
                self.assertEqual(Steering._offset_history , [])

    def test_explicit_reset_matches_new_controller(self):
        self.build_offset_history()
        Steering.reset()
        self.assertIsNone(Steering._prev_time)
        self.assertIsNone(Steering._prev_offset)
        self.assertEqual(Steering._offset_rate , 0)
        self.assertEqual(Steering._offset_history , [])
        self.assertEqual(self.step([[320 , 220] , [320 , 320]]) , 0)


if __name__ == "__main__":
    unittest.main()

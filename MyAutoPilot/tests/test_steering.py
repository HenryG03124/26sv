import math
import unittest

from MyAutoPilot.steering import Steering , prev_weight , raw_weight


class SteeringTests(unittest.TestCase):
    def test_screenshot_target_left_despite_positive_lane_slope(self):
        center_points = [[283 , 300] , [313 , 200]]
        old_steering_raw = 2 * (283 - 320) / 320 + math.atan2(30 , 100)
        self.assertGreater(old_steering_raw , 0)
        self.assertLess(Steering.steering(center_points , 200 , 3 , 0.06) , 0)

    def test_left_and_right_targets_are_symmetric(self):
        left_points = [[283 , 300] , [313 , 188]]
        right_points = [[357 , 300] , [327 , 188]]
        left = Steering.steering(left_points , 188 , 3 , 0)
        right = Steering.steering(right_points , 188 , 3 , 0)
        self.assertLess(left , 0)
        self.assertGreater(right , 0)
        self.assertAlmostEqual(left , -right)

    def test_centered_target(self):
        self.assertEqual(Steering.steering([[320 , 188] , [320 , 300]] , 188 , 3 , 0) , 0)

    def test_near_point_cannot_reverse_target_direction(self):
        for near_x in (200 , 283 , 320 , 400):
            with self.subTest(near_x = near_x):
                self.assertLess(Steering.steering([[near_x , 300] , [313 , 188]] , 188 , 3 , 0) , 0)

    def test_existing_smoothing_is_preserved(self):
        center_points = [[313 , 188]]
        steering_raw = 3 * math.atan2(-7 , 163)
        expected = prev_weight * 0.2 + raw_weight * steering_raw
        self.assertAlmostEqual(Steering.steering(center_points , 188 , 3 , 0.2) , expected)

    def test_output_stays_bounded(self):
        for target_x in (0 , 640):
            for steering_prev in (-1 , 0 , 1):
                steering = Steering.steering([[target_x , 188]] , 188 , 100 , steering_prev)
                self.assertGreaterEqual(steering , -1)
                self.assertLessEqual(steering , 1)

    def test_endpoint_validity_check_is_preserved(self):
        self.assertTrue(Steering.existence_filter([[313 , 188] , [283 , 300]] , 188 , 300))
        self.assertFalse(Steering.existence_filter([[283 , 300]] , 188 , 300))
        self.assertFalse(Steering.existence_filter([] , 188 , 300))


if __name__ == "__main__":
    unittest.main()

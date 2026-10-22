import unittest

from MyAutoPilot.lane_change import LaneChange


class LaneChangeTests(unittest.TestCase):
    def test_left_and_right_paths_connect_camera_center_to_target_center(self):
        for direction , current_x , neighbour_x , target_x in (("L" , 240 , 120 , 180) , ("R" , 400 , 520 , 460)):
            with self.subTest(direction = direction):
                lane_change = LaneChange(200 , 320 , 40)
                self.assertTrue(lane_change.request(direction))
                current_points = [[current_x , 320] , [current_x , 200]]
                neighbour_points = [[neighbour_x , 200] , [neighbour_x , 248]]
                path , exiting = lane_change.update(current_points , neighbour_points , 640)
                self.assertFalse(exiting)
                self.assertFalse(lane_change.exiting)
                self.assertEqual(lane_change.direction , direction)
                self.assertEqual(path[0] , [320 , 320])
                self.assertEqual(path[-1] , [target_x , 200])
                self.assertEqual([y for x , y in path] , list(range(320 , 199 , -1)))
                for x , y in path:
                    expected_x = 320 + (target_x - 320) * (320 - y) / 120
                    self.assertAlmostEqual(x , expected_x)

    def test_camera_center_uses_supplied_width(self):
        lane_change = LaneChange(200 , 320 , 40)
        lane_change.request("R")
        path , exiting = lane_change.update([[600 , 200]] , [[800 , 200]] , 1000)
        self.assertEqual(path[0] , [500 , 320])
        self.assertEqual(path[-1] , [700 , 200])
        self.assertFalse(exiting)

    def test_request_rejects_invalid_or_active_direction(self):
        lane_change = LaneChange(200 , 320 , 40)
        for direction in (None , "" , "up"):
            self.assertFalse(lane_change.request(direction))
            self.assertIsNone(lane_change.direction)
        self.assertTrue(lane_change.request("L"))
        self.assertFalse(lane_change.request("R"))
        self.assertEqual(lane_change.direction , "L")

    def test_inactive_update_does_not_generate_path(self):
        lane_change = LaneChange(200 , 320 , 40)
        path , exiting = lane_change.update([[240 , 200]] , [[120 , 200]] , 640)
        self.assertEqual(path , [])
        self.assertFalse(exiting)

    def test_small_frame_movements_do_not_accumulate_completion(self):
        lane_change = LaneChange(200 , 320 , 40)
        lane_change.request("L")
        for shift in (0 , 20 , 40 , 60 , 80):
            path , exiting = lane_change.update([[240 + shift , 200]] , [[120 + shift , 200]] , 640)
            self.assertTrue(path)
            self.assertFalse(exiting)
            self.assertEqual(lane_change.direction , "L")

    def test_single_boundary_jump_does_not_complete(self):
        lane_change = LaneChange(200 , 320 , 40)
        lane_change.request("L")
        lane_change.update([[240 , 200]] , [[120 , 200]] , 640)
        path , exiting = lane_change.update([[300 , 200]] , [[120 , 200]] , 640)
        self.assertTrue(path)
        self.assertFalse(exiting)
        path , exiting = lane_change.update([[300 , 200]] , [[180 , 200]] , 640)
        self.assertTrue(path)
        self.assertFalse(exiting)
        self.assertEqual(lane_change.direction , "L")

    def test_threshold_is_strictly_greater_than_diff(self):
        lane_change = LaneChange(200 , 320 , 40)
        lane_change.request("L")
        lane_change.update([[240 , 200]] , [[120 , 200]] , 640)
        path , exiting = lane_change.update([[280 , 200]] , [[160 , 200]] , 640)
        self.assertTrue(path)
        self.assertFalse(exiting)
        path , exiting = lane_change.update([[321 , 200]] , [[201 , 200]] , 640)
        self.assertEqual(path , [])
        self.assertTrue(exiting)
        self.assertTrue(lane_change.exiting)
        self.assertIsNone(lane_change.direction)

    def test_negative_and_positive_boundary_jumps_complete_without_next_lane_path(self):
        for direction , current_x , neighbour_x , shift in (("L" , 240 , 120 , 80) , ("R" , 400 , 520 , -80)):
            with self.subTest(direction = direction):
                lane_change = LaneChange(200 , 320 , 40)
                lane_change.request(direction)
                lane_change.update([[current_x , 200]] , [[neighbour_x , 200]] , 640)
                path , exiting = lane_change.update([[current_x + shift , 200]] , [[neighbour_x + shift , 200]] , 640)
                self.assertEqual(path , [])
                self.assertTrue(exiting)
                self.assertIsNone(lane_change.direction)

    def test_matching_rows_are_compared_independently_of_point_order(self):
        lane_change = LaneChange(200 , 320 , 40)
        lane_change.request("L")
        current_points = [[220 , 200] , [300 , 220] , [380 , 240]]
        neighbour_points = [[100 , 200] , [180 , 220] , [260 , 240]]
        lane_change.update(current_points , neighbour_points , 640)
        path , exiting = lane_change.update(list(reversed(current_points)) , list(reversed(neighbour_points)) , 640)
        self.assertTrue(path)
        self.assertFalse(exiting)
        current_points = [[430 , 240] , [270 , 200] , [350 , 220]]
        neighbour_points = [[230 , 220] , [310 , 240] , [150 , 200]]
        path , exiting = lane_change.update(current_points , neighbour_points , 640)
        self.assertEqual(path , [])
        self.assertTrue(exiting)

    def test_isolated_row_outliers_do_not_complete(self):
        lane_change = LaneChange(200 , 320 , 40)
        lane_change.request("L")
        current_points = [[240 , y] for y in (200 , 220 , 240)]
        neighbour_points = [[120 , y] for y in (200 , 220 , 240)]
        lane_change.update(current_points , neighbour_points , 640)
        current_points = [[240 , 200] , [240 , 220] , [340 , 240]]
        neighbour_points = [[120 , 200] , [120 , 220] , [220 , 240]]
        path , exiting = lane_change.update(current_points , neighbour_points , 640)
        self.assertTrue(path)
        self.assertFalse(exiting)

    def test_invalid_target_exits_without_reusing_previous_path(self):
        invalid_points = (
            ([] , [[120 , 200]]) ,
            ([[240 , 200]] , []) ,
            ([[240 , 201]] , [[120 , 200]]) ,
            ([[240 , 200]] , [[120 , 201]]) ,
            ([[240 , 200]] , [[260 , 200]]) ,
            ([[240 , 200]] , [[-1 , 200]]) ,
            ([[640 , 200]] , [[120 , 200]]) ,
        )
        for current_points , neighbour_points in invalid_points:
            with self.subTest(current_points = current_points , neighbour_points = neighbour_points):
                lane_change = LaneChange(200 , 320 , 40)
                lane_change.request("L")
                path , _ = lane_change.update([[240 , 200]] , [[120 , 200]] , 640)
                self.assertTrue(path)
                path , exiting = lane_change.update(current_points , neighbour_points , 640)
                self.assertEqual(path , [])
                self.assertTrue(exiting)
                self.assertTrue(lane_change.exiting)
                self.assertIsNone(lane_change.direction)
                path , exiting = lane_change.update([[240 , 200]] , [[120 , 200]] , 640)
                self.assertEqual(path , [])
                self.assertTrue(exiting)

    def test_new_request_clears_previous_exit_and_comparison_history(self):
        lane_change = LaneChange(200 , 320 , 40)
        lane_change.request("L")
        lane_change.update([[240 , 200]] , [[120 , 200]] , 640)
        lane_change.update([[320 , 200]] , [[200 , 200]] , 640)
        self.assertTrue(lane_change.exiting)
        self.assertTrue(lane_change.request("R"))
        self.assertFalse(lane_change.exiting)
        path , exiting = lane_change.update([[400 , 200]] , [[520 , 200]] , 640)
        self.assertTrue(path)
        self.assertFalse(exiting)
        self.assertEqual(lane_change.direction , "R")

    def test_new_request_clears_exit_after_invalid_target(self):
        lane_change = LaneChange(200 , 320 , 40)
        lane_change.request("L")
        lane_change.update([[240 , 200]] , [] , 640)
        self.assertTrue(lane_change.exiting)
        self.assertTrue(lane_change.request("L"))
        self.assertFalse(lane_change.exiting)
        path , exiting = lane_change.update([[240 , 200]] , [[120 , 200]] , 640)
        self.assertTrue(path)
        self.assertFalse(exiting)


if __name__ == "__main__":
    unittest.main()

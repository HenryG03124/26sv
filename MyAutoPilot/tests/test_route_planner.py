import importlib
import sys
import unittest
from unittest.mock import patch

from MyAutoPilot import lane_change
from MyAutoPilot.steering import Steering


with patch.dict(sys.modules , {"lane_change": lane_change}):
    RoutePlanner = importlib.import_module("MyAutoPilot.route_planner").RoutePlanner


class RoutePlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = RoutePlanner(200 , 320 , 40)
        self.center_points = [[320 , y] for y in range(320 , 199 , -1)]
        self.cl_lane_points = [[240 , y] for y in range(200 , 321)]
        self.cr_lane_points = [[400 , y] for y in range(200 , 321)]
        self.nl_lane_points = [[120 , y] for y in range(200 , 249)]
        self.nr_lane_points = [[520 , y] for y in range(200 , 249)]
        self.lane_available = {"current": True , "L": True , "R": True}
        self.steering_controller = Steering()
        self.addCleanup(self.steering_controller.reset)

    def update(self , direction = None):
        return self.planner.update(self.center_points , self.cl_lane_points , self.cr_lane_points ,
                                   self.nl_lane_points , self.nr_lane_points , self.lane_available , direction = direction , width = 640)

    def test_center_path_is_returned_unchanged(self):
        path = self.update()
        self.assertIs(path , self.center_points)
        self.assertEqual(self.planner.control_priority , "center")
        self.assertIsNone(self.planner.lane_change.direction)
        self.assertEqual(self.planner.lane_change_available , {"L": True , "R": True})
        self.center_points = []
        self.assertEqual(self.update() , [])

    def test_left_and_right_requests_select_matching_boundary_pair(self):
        for direction , target_x in (("L" , 180) , ("R" , 460)):
            with self.subTest(direction = direction):
                self.planner = RoutePlanner(200 , 320 , 40)
                path = self.update(direction)
                self.assertEqual(path[0] , [320 , 320])
                self.assertEqual(path[-1] , [target_x , 200])
                self.assertEqual(self.planner.control_priority , "lane change")
                self.assertEqual(self.planner.lane_change.direction , direction)
                self.assertEqual(self.planner.lane_change_available , {"L": False , "R": False})

    def test_active_change_ignores_opposite_request(self):
        first_path = self.update("L")
        path = self.update("R")
        self.assertEqual(path , first_path)
        self.assertEqual(self.planner.control_priority , "lane change")
        self.assertEqual(self.planner.lane_change.direction , "L")

    def test_completion_returns_center_in_same_frame_without_starting_queued_request(self):
        self.update("L")
        self.cl_lane_points = [[320 , y] for y in range(200 , 321)]
        self.nl_lane_points = [[200 , y] for y in range(200 , 249)]
        path = self.update("R")
        self.assertIs(path , self.center_points)
        self.assertEqual(self.planner.control_priority , "center")
        self.assertTrue(self.planner.lane_change.exiting)
        self.assertIsNone(self.planner.lane_change.direction)
        self.assertEqual(self.planner.lane_change_available , {"L": True , "R": True})
        self.assertIs(self.update() , self.center_points)
        path = self.update("R")
        self.assertEqual(path[-1] , [460 , 200])
        self.assertEqual(self.planner.control_priority , "lane change")
        self.assertFalse(self.planner.lane_change.exiting)

    def test_missing_target_aborts_and_returns_current_center(self):
        self.update("R")
        self.nr_lane_points = []
        self.lane_available["R"] = False
        self.center_points = [[330 , y] for y in range(320 , 199 , -1)]
        path = self.update()
        self.assertIs(path , self.center_points)
        self.assertEqual(self.planner.control_priority , "center")
        self.assertTrue(self.planner.lane_change.exiting)
        self.assertIsNone(self.planner.lane_change.direction)
        self.assertEqual(self.planner.lane_change_available , {"L": True , "R": False})
        self.nr_lane_points = [[520 , y] for y in range(200 , 249)]
        self.lane_available["R"] = True
        self.assertIs(self.update() , self.center_points)
        self.assertEqual(self.update("R")[-1] , [460 , 200])

    def test_missing_target_at_request_keeps_center_control(self):
        self.nl_lane_points = []
        self.lane_available["L"] = False
        with patch.object(self.planner.lane_change , "request" , wraps = self.planner.lane_change.request) as request:
            self.assertIs(self.update("L") , self.center_points)
            request.assert_not_called()
        self.assertEqual(self.planner.control_priority , "center")
        self.assertFalse(self.planner.lane_change.exiting)
        self.assertIsNone(self.planner.lane_change.direction)

    def test_unavailable_request_is_not_queued(self):
        self.lane_available["L"] = False
        with patch.object(self.planner.lane_change , "request" , wraps = self.planner.lane_change.request) as request:
            self.assertIs(self.update("L") , self.center_points)
            request.assert_not_called()
        self.lane_available["L"] = True
        self.assertIs(self.update() , self.center_points)
        self.assertIsNone(self.planner.lane_change.direction)

    def test_left_and_right_availability_are_independent(self):
        for available_side , unavailable_side in (("L" , "R") , ("R" , "L")):
            with self.subTest(available_side = available_side):
                self.planner = RoutePlanner(200 , 320 , 40)
                self.lane_available[available_side] = True
                self.lane_available[unavailable_side] = False
                self.assertIs(self.update(unavailable_side) , self.center_points)
                self.assertTrue(self.planner.lane_change_available[available_side])
                self.assertFalse(self.planner.lane_change_available[unavailable_side])
                self.assertIsNot(self.update(available_side) , self.center_points)
                self.assertEqual(self.planner.lane_change.direction , available_side)

    def test_invalid_direction_does_not_request_lane_change(self):
        with patch.object(self.planner.lane_change , "request" , wraps = self.planner.lane_change.request) as request:
            self.assertIs(self.update("up") , self.center_points)
            request.assert_not_called()

    def test_unavailable_state_does_not_cancel_active_change_with_valid_points(self):
        first_path = self.update("L")
        self.lane_available = {"current": False , "L": False , "R": False}
        path = self.update()
        self.assertEqual(path , first_path)
        self.assertEqual(self.planner.control_priority , "lane change")
        self.assertEqual(self.planner.lane_change.direction , "L")
        self.assertFalse(self.planner.lane_change.exiting)
        self.assertEqual(self.planner.lane_change_available , {"L": False , "R": False})

    def test_planned_path_is_accepted_by_steering_with_correct_direction(self):
        for direction , sign in (("L" , -1) , ("R" , 1)):
            with self.subTest(direction = direction):
                self.planner = RoutePlanner(200 , 320 , 40)
                self.steering_controller.reset()
                previous = self.steering_controller.steering(self.center_points , 200 , 3 , 0 ,
                                                             speed_kmh = 45 , timestamp = 0 , y_near = 320)
                accepted = False
                for frame in range(1 , 11):
                    path = self.update(direction if frame == 1 else None)
                    self.assertTrue(self.steering_controller.existence_filter(path , 200 , 320))
                    previous = self.steering_controller.steering(path , 200 , 3 , previous ,
                                                                 speed_kmh = 45 , timestamp = frame / 10 , y_near = 320)
                    accepted |= self.steering_controller.diagnostics.get("path_reacquired" , False)
                self.assertTrue(accepted)
                self.assertEqual(self.steering_controller.diagnostics["status"] , "tracking")
                self.assertGreater(previous * sign , 0)


if __name__ == "__main__":
    unittest.main()

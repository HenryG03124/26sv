import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock , patch

from MyAutoPilot.auto_brakes import AutoBrakes


class AutoBrakesTests(unittest.TestCase):
    def setUp(self):
        self.brakes = AutoBrakes()
        self.left_points = [[220 , 188] , [180 , 268] , [168 , 292]]
        self.right_points = [[420 , 188] , [460 , 268] , [472 , 292]]

    def tracking(self , box , direction = "stationary" , status = "tracked"):
        return {"box": box , "direction": direction , "status": status}

    def step(self , timestamp , box = None , track_id = 1 , direction = "stationary" , lanes = True):
        trackings = {} if box is None else {track_id: self.tracking(box , direction)}
        return self.brakes.brake(self.left_points if lanes else [] , self.right_points if lanes else [] , trackings , timestamp , 352 , 640)

    def test_static_distant_car_does_not_brake(self):
        for i in range(20):
            self.assertEqual(self.step(i * 0.05 , [290 , 210 , 350 , 270]) , 0)

    def test_backward_alone_does_not_brake(self):
        for i in range(10):
            self.assertEqual(self.step(i * 0.05 , [290 , 210 , 350 , 270] , direction = "backward") , 0)

    def test_height_growth_brakes_even_when_direction_is_stationary(self):
        for i in range(13):
            timestamp = i * 0.05
            value = self.step(timestamp , [290 , 280 - (40 + 30 * timestamp) , 350 , 280])
        self.assertGreater(value , 0)
        self.assertEqual(self.brakes.target_id , 1)
        self.assertAlmostEqual(self.brakes.ttc , 58 / 30)
        self.assertEqual(self.brakes.reason , "ttc")

    def test_approach_result_is_similar_across_frame_rates(self):
        values = []
        for fps in (10 , 20 , 60):
            self.brakes = AutoBrakes()
            for i in range(fps + 1):
                timestamp = i / fps
                value = self.step(timestamp , [290 , 280 - (40 + 30 * timestamp) , 350 , 280])
            values.append(value)
        for value in values:
            self.assertGreater(value , 0)
            self.assertAlmostEqual(value , values[0])

    def test_small_height_jitter_does_not_brake(self):
        for i in range(30):
            self.assertEqual(self.step(i * 0.05 , [290 , 210 + (i % 2) , 350 , 270]) , 0)

    def test_receding_car_does_not_brake(self):
        for i in range(15):
            self.assertEqual(self.step(i * 0.05 , [290 , 190 + i , 350 , 270]) , 0)

    def test_adjacent_close_car_does_not_brake(self):
        for i in range(10):
            self.assertEqual(self.step(i * 0.05 , [520 , 220 , 630 , 330]) , 0)

    def test_partial_cut_in_brakes_without_center_inside_lane(self):
        # Near left boundary is x=154; box center is x=150, but its body overlaps.
        self.assertEqual(self.step(0 , [100 , 250 , 200 , 320] , direction = "right") , 1)

    def test_near_car_requires_confirmation(self):
        box = [290 , 250 , 350 , 300]
        self.assertEqual(self.step(0 , box) , 0)
        self.assertEqual(self.step(0.1 , box) , 0)
        self.assertEqual(self.brakes.reason , "none")
        self.assertEqual(self.step(0.2 , box) , 0.4)
        self.assertEqual(self.brakes.reason , "near")

    def test_emergency_car_beyond_y_near_brakes_immediately(self):
        self.assertEqual(self.step(0 , [290 , 250 , 350 , 320] , direction = "unknown") , 1)
        self.assertEqual(self.brakes.reason , "emergency")

    def test_box_clipped_to_image_bottom_is_not_missed(self):
        self.assertEqual(self.step(0 , [290 , 240 , 350 , 352]) , 1)

    def test_near_extension_is_bounded(self):
        self.left_points = [[220 , 188] , [200 , 230]]
        self.right_points = [[420 , 188] , [440 , 230]]
        self.assertEqual(self.step(0 , [290 , 250 , 350 , 352]) , 0)

    def test_lost_track_cannot_trigger_braking(self):
        trackings = {1: self.tracking([290 , 250 , 350 , 320] , status = "lost")}
        self.assertEqual(self.brakes.brake(self.left_points , self.right_points , trackings , 0 , 352 , 640) , 0)

    def test_release_holds_through_short_detection_dropout(self):
        self.step(0 , [290 , 250 , 350 , 320])
        self.assertEqual(self.step(0.2) , 1)
        self.assertEqual(self.brakes.reason , "hold")
        self.assertEqual(self.step(0.5) , 0)
        self.assertEqual(self.brakes.reason , "none")

    def test_new_id_does_not_inherit_approach_history(self):
        self.step(0 , [290 , 240 , 350 , 280])
        self.assertEqual(self.step(0.3 , [290 , 210 , 350 , 280] , track_id = 2) , 0)
        self.assertEqual(set(self.brakes.histories) , {2})

    def test_reused_id_after_frame_gap_restarts_confirmation(self):
        self.step(0 , [290 , 250 , 350 , 300])
        self.assertEqual(self.step(0.55 , [290 , 250 , 350 , 300]) , 0)
        self.assertEqual(len(self.brakes.histories[1]) , 1)

    def test_missing_lanes_have_short_hold_and_explicit_status(self):
        box = [290 , 250 , 350 , 320]
        self.assertEqual(self.step(0 , box) , 1)
        self.assertEqual(self.step(0.2 , box , lanes = False) , 1)
        self.assertEqual(self.brakes.lane_status , "held")
        self.assertEqual(self.step(0.7 , box , lanes = False) , 0)
        self.assertEqual(self.brakes.lane_status , "missing")

    def test_far_and_crossed_boundaries_are_not_used(self):
        self.assertEqual(self.step(0 , [290 , 80 , 350 , 180]) , 0)
        self.left_points , self.right_points = self.right_points , self.left_points
        self.assertEqual(self.step(0.1 , [290 , 250 , 350 , 320]) , 0)

    def test_strongest_target_wins_without_mutating_trackings(self):
        trackings = {
            1: self.tracking([280 , 250 , 320 , 300]) ,
            2: self.tracking([320 , 250 , 380 , 320]) ,
        }
        for timestamp in (0 , 0.2):
            value = self.brakes.brake(self.left_points , self.right_points , trackings , timestamp , 352 , 640)
        self.assertEqual(value , 1)
        self.assertEqual(self.brakes.target_id , 2)
        self.assertEqual(self.brakes.reason , "emergency")
        self.assertEqual(set(trackings[1]) , {"box" , "direction" , "status"})

    def test_near_condition_does_not_hide_stronger_ttc_reason(self):
        for i in range(13):
            timestamp = i * 0.05
            self.step(timestamp , [290 , 300 - (50 + 40 * timestamp) , 350 , 300])
        self.assertGreater(self.brakes.brake_value , 0.4)
        self.assertEqual(self.brakes.reason , "ttc")


class BrakeControllerTests(unittest.TestCase):
    def test_trigger_output_and_exit_release_without_real_gamepad(self):
        gamepad = Mock()
        fake_vgamepad = types.SimpleNamespace(VX360Gamepad = lambda: gamepad)
        path = Path(__file__).resolve().parents[1] / "input_controller.py"
        spec = importlib.util.spec_from_file_location("brake_test_input_controller" , path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules , {"vgamepad": fake_vgamepad}):
            spec.loader.exec_module(module)
            controller = module.InputController()
            controller.brake_controller(1.5)
            gamepad.left_trigger_float.assert_called_with(value_float = 1.0)
            controller.brake_controller(-1)
            gamepad.left_trigger_float.assert_called_with(value_float = 0.0)
            controller.close()
        gamepad.reset.assert_called_once()
        self.assertEqual(gamepad.update.call_count , 3)


if __name__ == "__main__":
    unittest.main()

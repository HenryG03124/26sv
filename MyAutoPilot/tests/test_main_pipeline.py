import importlib
import importlib.util
import json
import sys
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock , patch

import numpy as np


class MainPipelineTests(unittest.TestCase):
    def test_pipeline_wires_throttle_and_releases_on_braking_or_lane_loss(self):
        aliases = {name: importlib.import_module("MyAutoPilot." + name)
                   for name in ("screen_capture" , "draw" , "pixel_classifier_v2" , "object_detector" ,
                                "mov_detector" , "lane_refiner" , "lane_change" , "steering" ,
                                "auto_brakes" , "throttle" , "logger" , "scs_telemetry")}
        controller = Mock()
        aliases["input_controller"] = types.SimpleNamespace(InputController = Mock(return_value = controller))
        aliases["auto_config"] = types.SimpleNamespace(auto_config = Mock())
        spec = importlib.util.spec_from_file_location("main_pipeline_under_test" , Path(__file__).resolve().parents[1] / "main.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules , aliases):
            sys.modules["route_planner"] = importlib.import_module("MyAutoPilot.route_planner")
            spec.loader.exec_module(module)
        module.autoconfig = "enable"

        mask = np.zeros((352 , 640) , dtype = np.uint8)
        for y in range(module.current_y_far , module.current_y_near + 1):
            mask[y , 80] = 3
            mask[y , 200] = 3
            mask[y , 220] = 3
            mask[y , 440] = 3
            mask[y , 460] = 3
            mask[y , 580] = 3
        left_mask = mask.copy()
        left_mask[: , 320:] = 0
        right_mask = mask.copy()
        right_mask[: , :320] = 0
        missing_mask = np.zeros_like(mask)
        masks = [mask] * 6 + [left_mask] * 8 + [right_mask] * 8 + [missing_mask] + [left_mask] * 5 + [missing_mask] + [right_mask] * 5
        frame_count = len(masks)
        capture = Mock()
        capture.capture_window.return_value = np.zeros((352 , 640 , 3) , dtype = np.uint8)
        capture.native_resolution.return_value = (1280 , 704)
        classifier = Mock()
        classifier.predict.side_effect = masks
        detector = Mock()
        detector.predict.return_value = (None , None , None , None)
        detector.box_filter.return_value = ([] , [] , [])
        movement = Mock()
        movement.update.return_value = {}
        telemetry = Mock()
        telemetry.read.return_value = {"status": "ok" , "speed_kmh": 45}
        user32 = Mock()
        user32.GetAsyncKeyState.return_value = 1
        logger = Mock()
        logger.frame_id = 0
        records = []
        available_states = []
        auto_brakes = module.AutoBrakes()
        original_brake = auto_brakes.brake

        def write(timestamp , data):
            records.append(json.loads(json.dumps(data)))
            logger.frame_id += 1

        def brake(cl , cr , trackings , timestamp , height , width , lane_available):
            value = original_brake(cl , cr , trackings , timestamp , height , width , lane_available)
            return 0.5 if len(records) == 5 else value

        def show_available(frame , lane_change_available):
            available_states.append(lane_change_available.copy())

        logger.write.side_effect = write
        steering_controller = module.Steering()
        try:
            with ExitStack() as stack:
                for name , value in (("ScreenCapture" , capture) , ("PixelClassifierV2" , classifier) ,
                                     ("ObjectDetectorLite" , detector) , ("MovDetector" , movement) ,
                                     ("Logger" , logger) , ("SCSTelemetry" , telemetry) ,
                                     ("AutoBrakes" , auto_brakes) , ("Steering" , steering_controller)):
                    stack.enter_context(patch.object(module , name , return_value = value))
                stack.enter_context(patch.object(module.ctypes , "WinDLL" , return_value = user32 , create = True))
                stack.enter_context(patch.object(module.cv2 , "imshow"))
                stack.enter_context(patch.object(module.cv2 , "destroyAllWindows"))
                stack.enter_context(patch.object(module.cv2 , "waitKey" , side_effect = [-1] * (frame_count - 1) + [ord("q")]))
                stack.enter_context(patch.object(module.time , "monotonic" , side_effect = [10 + index * 0.01 for index in range(frame_count * 2)]))
                brakes = stack.enter_context(patch.object(auto_brakes , "brake" , side_effect = brake))
                output = stack.enter_context(patch("builtins.print"))
                fps = stack.enter_context(patch.object(module.Draw , "show_fps"))
                speed = stack.enter_context(patch.object(module.Draw , "show_speed"))
                throttle = stack.enter_context(patch.object(module.Draw , "show_throttle"))
                available = stack.enter_context(patch.object(module.Draw , "show_available" , side_effect = show_available))
                module.main()
        finally:
            steering_controller.reset()
        aliases["auto_config"].auto_config.assert_called_once_with(module.controls_path)
        self.assertEqual(len(records) , frame_count)
        self.assertEqual(capture.capture_window.call_count , frame_count)
        self.assertEqual(movement.update.call_count , frame_count)
        for call in movement.update.call_args_list:
            self.assertEqual(call.args[:3] , ([] , [] , []))
            self.assertEqual(call.args[4] , module.spd)
        capture.capture_window.assert_called_with("Euro Truck Simulator 2")
        capture.capture_frame.assert_not_called()
        self.assertIsNone(records[0]["fps"])
        self.assertGreater(records[1]["fps"] , 0)
        self.assertEqual(records[2]["steering_debug"]["speed_kmh"] , 45)
        self.assertFalse(records[2]["steering_debug"]["speed_fallback"])
        for record in records[:6]:
            self.assertEqual(record["center_points"] , [[330 , y] for y in range(module.current_y_near , module.current_y_far - 1 , -1)])
            self.assertEqual(record["NL_lane_points"] , [[80 , y] for y in range(module.neighbour_y_far , module.neighbour_y_near + 1)])
            self.assertEqual(record["NR_lane_points"] , [[580 , y] for y in range(module.neighbour_y_far , module.neighbour_y_near + 1)])
            for side in ("NL" , "NR"):
                self.assertGreater(record["refiner_sample_count"][side] , 0)
                self.assertEqual(record["refiner_status"][side] , "tracking")
                for x , y in record["lane_sample_points"][side]:
                    self.assertTrue(module.neighbour_y_far <= y <= module.neighbour_y_near)
            self.assertNotIn("road_center_mode" , record)
            self.assertNotIn("timings_ms" , record)
            self.assertNotIn("timings_frame_id" , record)
        for record in records:
            self.assertEqual(record["path_points"] , record["center_points"])
            self.assertEqual(record["control_priority"] , "center")
            self.assertIsNone(record["lane_change_direction"])
            self.assertFalse(record["lane_change_exiting"])
            self.assertNotIn("lane_change_completed" , record)
            self.assertEqual(record["lane_change_diff"] , module.lane_change_diff)
            self.assertEqual(record["lane_change_available"] , {side: record["lane_available"][side] for side in ("L" , "R")})
        self.assertEqual([record["lane_available"]["current"] for record in records] , [True] * 6 + [False] * (frame_count - 6))
        self.assertEqual(records[6]["lane_status"] , "held")
        self.assertFalse(records[6]["lane_available"]["current"])
        self.assertTrue(records[6]["center_points"])
        self.assertEqual([call.args[-1] for call in brakes.call_args_list] , [record["lane_available"]["current"] for record in records])
        self.assertEqual(available_states , [record["lane_change_available"] for record in records])
        self.assertEqual([record["lane_tracking_frames"] for record in records] , [1 , 2 , 3 , 4 , 5 , 6] + [0] * (frame_count - 6))
        self.assertEqual([record["center_tracking_frames"] for record in records] , list(range(1 , 23)) + [0 , 1 , 2 , 3 , 4 , 5 , 0 , 1 , 2 , 3 , 4 , 5])
        self.assertEqual([record["throttle"] for record in records[:4]] , [0.0] * 4)
        self.assertGreater(records[4]["throttle"] , 0)
        self.assertEqual(records[5]["throttle"] , 0)
        for record in records[6:22]:
            self.assertGreater(record["throttle"] , 0)
        for start , end , present , missing in ((6 , 14 , "CL" , "CR") , (14 , 22 , "CR" , "CL") , (23 , 28 , "CL" , "CR") , (29 , 34 , "CR" , "CL")):
            for record in records[start:end]:
                self.assertTrue(record[present + "_lane_points"])
                self.assertEqual(record[missing + "_lane_points"] , [])
                self.assertEqual(record["center_points"] , records[4]["center_points"])
        for index in (22 , 28):
            self.assertEqual(records[index]["center_points"] , [])
            self.assertEqual(records[index]["throttle"] , 0)
            self.assertEqual([record["throttle"] for record in records[index + 1:index + 5]] , [0.0] * 4)
            self.assertGreater(records[index + 5]["throttle"] , 0)
        self.assertEqual(fps.call_count , frame_count)
        self.assertEqual(speed.call_count , frame_count)
        self.assertEqual(throttle.call_count , frame_count)
        self.assertEqual(available.call_count , frame_count)
        self.assertEqual(brakes.call_count , frame_count)
        self.assertEqual(output.call_count , frame_count)
        self.assertEqual(controller.steering_controller.call_count , frame_count)
        self.assertEqual(controller.throttle_controller.call_count , frame_count)
        for index , record in enumerate(records):
            self.assertTrue(record["throttle_sent"])
            self.assertEqual(controller.throttle_controller.call_args_list[index].args , (record["throttle"] ,))
            self.assertEqual(throttle.call_args_list[index].args[1] , record["throttle"])
            self.assertEqual(output.call_args_list[index].args[-2:] , ("throttle:" , record["throttle"]))
        controller.close.assert_called_once()
        telemetry.close.assert_called_once()
        logger.close.assert_called_once()

    def test_arrow_keys_route_path_and_return_control_after_lane_change(self):
        aliases = {name: importlib.import_module("MyAutoPilot." + name)
                   for name in ("screen_capture" , "draw" , "pixel_classifier_v2" , "object_detector" ,
                                "mov_detector" , "lane_refiner" , "lane_change" , "steering" ,
                                "auto_brakes" , "throttle" , "logger" , "scs_telemetry")}
        controller = Mock()
        aliases["input_controller"] = types.SimpleNamespace(InputController = Mock(return_value = controller))
        aliases["auto_config"] = types.SimpleNamespace(auto_config = Mock())
        spec = importlib.util.spec_from_file_location("main_lane_change_under_test" , Path(__file__).resolve().parents[1] / "main.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules , aliases):
            sys.modules["route_planner"] = importlib.import_module("MyAutoPilot.route_planner")
            spec.loader.exec_module(module)
        module.autoconfig = "disable"

        keys = [(True , True) , (False , False) , (True , False) , (True , False) ,
                (True , False) , (True , False) , (False , False) , (True , False) ,
                (True , False) , (False , False) , (False , True) , (True , False) ,
                (False , False) , (True , True) , (False , False) , (True , False)]
        speed_keys = [(False , False) , (True , False) , (True , False) , (False , False) ,
                      (True , False) , (False , False) , (False , True) , (False , True) ,
                      (False , False) , (True , True) , (True , False) , (False , False) ,
                      (False , True) , (False , False) , (False , True) , (False , False)]
        target_speeds = [50 , 55 , 55 , 55 , 60 , 60 , 55 , 55 , 55 , 55 , 55 , 55 , 50 , 50 , 45 , 45]
        frame_count = len(keys)
        left_offsets = [0] * 4 + [-35] * 4 + [-70] * 8
        right_offsets = [0] * 12 + [35] * 4
        lanes = []
        centers = []
        for left_offset , right_offset in zip(left_offsets , right_offsets):
            cl = [[220 + left_offset , y] for y in range(module.current_y_far , module.current_y_near + 1)]
            cr = [[440 + right_offset , y] for y in range(module.current_y_far , module.current_y_near + 1)]
            nl = [[80 + left_offset , y] for y in range(module.neighbour_y_far , module.neighbour_y_near + 1)]
            nr = [[580 + right_offset , y] for y in range(module.neighbour_y_far , module.neighbour_y_near + 1)]
            lanes.append((cl , cr , nl , nr))
            centers.append([[(660 + left_offset + right_offset) // 2 , y]
                            for y in range(module.current_y_near , module.current_y_far - 1 , -1)])
        centers[3] = []
        lanes[15] = (lanes[15][0] , lanes[15][1] , [] , lanes[15][3])

        mask = np.zeros((352 , 640) , dtype = np.uint8)
        capture = Mock()
        capture.capture_window.return_value = np.zeros((352 , 640 , 3) , dtype = np.uint8)
        classifier = Mock()
        classifier.predict.return_value = mask
        detector = Mock()
        detector.predict.return_value = (None , None , None , None)
        detector.box_filter.return_value = ([] , [] , [])
        movement = Mock()
        movement.update.return_value = {}
        telemetry = Mock()
        telemetry.read.return_value = {"status": "ok" , "speed_kmh": 45}
        user32 = Mock()
        key_states = []
        for (left_pressed , right_pressed) , (up_pressed , down_pressed) in zip(keys , speed_keys):
            key_states.append(0x8000 if left_pressed else 0)
            key_states.append(0x8000 if right_pressed else 0)
            key_states.append(0x8000 if up_pressed else 0)
            key_states.append(0x8000 if down_pressed else 0)
        user32.GetAsyncKeyState.side_effect = key_states
        refiner = Mock()
        refiner.lane_center = 301
        refiner.avalible_points = {"CL": 30 , "CR": 30 , "NL": 12 , "NR": 12}
        refiner.status = {"CL": "tracking" , "CR": "tracking" , "NL": "tracking" , "NR": "tracking" , "center": "accurate"}
        refiner.tracking_frames = 10
        refiner.lane_available = {"current": False , "L": False , "R": False}
        refiner.calculate_center_points.side_effect = centers
        sample_centers = []

        def refine(mask , degree , sgl_lane_x_offset , min_neighbour_diff , return_sample):
            points = lanes[len(sample_centers)]
            refiner.lane_available = {"current": bool(points[0] and points[1]) ,
                                     "L": bool(points[0] and points[2]) ,
                                     "R": bool(points[1] and points[3])}
            sample_centers.append(refiner.lane_center)
            refiner.lane_center = 301
            return points + points

        refiner.refine.side_effect = refine
        planner = module.RoutePlanner(module.current_y_far , module.current_y_near , module.lane_change_diff)
        logger = Mock()
        records = []
        available_states = []

        def write(timestamp , data):
            records.append(json.loads(json.dumps(data)))

        def show_available(frame , lane_change_available):
            available_states.append(lane_change_available.copy())

        logger.write.side_effect = write
        steering_controller = module.Steering()
        try:
            with ExitStack() as stack:
                for name , value in (("ScreenCapture" , capture) , ("PixelClassifierV2" , classifier) ,
                                     ("ObjectDetectorLite" , detector) , ("MovDetector" , movement) ,
                                     ("Logger" , logger) , ("SCSTelemetry" , telemetry) ,
                                     ("LaneRefiner" , refiner) , ("RoutePlanner" , planner) ,
                                     ("Steering" , steering_controller)):
                    stack.enter_context(patch.object(module , name , return_value = value))
                keyboard = stack.enter_context(patch.object(module.ctypes , "WinDLL" , return_value = user32 , create = True))
                stack.enter_context(patch.object(module.cv2 , "imshow"))
                stack.enter_context(patch.object(module.cv2 , "destroyAllWindows"))
                stack.enter_context(patch.object(module.cv2 , "waitKey" , side_effect = [-1] * (frame_count - 1) + [ord("q")]))
                brakes = stack.enter_context(patch.object(module.AutoBrakes , "brake" , return_value = 0.0))
                stack.enter_context(patch.object(module.time , "monotonic" , side_effect = [10 + index * 0.01 for index in range(frame_count * 2)]))
                stack.enter_context(patch("builtins.print"))
                request = stack.enter_context(patch.object(planner.lane_change , "request" , wraps = planner.lane_change.request))
                draw = stack.enter_context(patch.object(module.Draw , "draw_road_center" , wraps = module.Draw.draw_road_center))
                available = stack.enter_context(patch.object(module.Draw , "show_available" , side_effect = show_available))
                steering = stack.enter_context(patch.object(steering_controller , "steering" , wraps = steering_controller.steering))
                target = stack.enter_context(patch.object(module.Draw , "show_target_speed"))
                module.main()
        finally:
            steering_controller.reset()

        aliases["auto_config"].auto_config.assert_not_called()
        keyboard.assert_called_once_with("user32" , use_last_error = True)
        self.assertEqual(user32.GetAsyncKeyState.argtypes , [module.ctypes.c_int])
        self.assertEqual(user32.GetAsyncKeyState.restype , module.ctypes.c_short)
        self.assertEqual([call.args[0] for call in user32.GetAsyncKeyState.call_args_list] , [0x25 , 0x27 , 0x26 , 0x28] * frame_count)
        self.assertEqual([call.args[0] for call in request.call_args_list] , ["L" , "L" , "R"])
        self.assertEqual(len(records) , frame_count)
        self.assertEqual([record["target_speed"] for record in records] , target_speeds)
        self.assertEqual([call.args[1] for call in target.call_args_list] , target_speeds)
        self.assertEqual(records[4]["throttle"] , 1)
        self.assertAlmostEqual(records[12]["throttle"] , 0.502)
        for index , record in enumerate(records):
            self.assertEqual(controller.throttle_controller.call_args_list[index].args , (record["throttle"] ,))
        self.assertEqual(draw.call_count , frame_count)
        self.assertEqual(available.call_count , frame_count)
        self.assertEqual(steering.call_count , frame_count)
        self.assertEqual([call.args[-1] for call in brakes.call_args_list] , [record["lane_available"]["current"] for record in records])
        self.assertEqual(available_states , [record["lane_change_available"] for record in records])
        self.assertEqual([record["center_tracking_frames"] for record in records] , list(range(1 , frame_count + 1)))
        active_frames = (2 , 3 , 7 , 10 , 11)
        camera_center_frames = (2 , 3 , 4 , 7 , 8 , 10 , 11 , 12 , 15)
        for index , record in enumerate(records):
            self.assertEqual(record["center_points"] , centers[index])
            self.assertEqual(draw.call_args_list[index].args[1] , record["path_points"])
            self.assertEqual(steering.call_args_list[index].args[0] , record["path_points"])
            self.assertEqual(sample_centers[index] , 320 if index in camera_center_frames else 301)
            self.assertEqual(record["lane_change_diff"] , module.lane_change_diff)
            if index in active_frames:
                direction = "L" if index < 10 else "R"
                current , neighbour = (lanes[index][0] , lanes[index][2]) if direction == "L" else (lanes[index][1] , lanes[index][3])
                self.assertEqual(record["control_priority"] , "lane change")
                self.assertEqual(record["lane_change_direction"] , direction)
                self.assertFalse(record["lane_change_exiting"])
                self.assertEqual(record["lane_change_available"] , {"L": False , "R": False})
                self.assertEqual(record["path_points"][0] , [320 , module.current_y_near])
                self.assertEqual(record["path_points"][-1] , [(current[0][0] + neighbour[0][0]) / 2 , module.current_y_far])
                self.assertNotEqual(record["path_points"] , record["center_points"])
            else:
                self.assertEqual(record["control_priority"] , "center")
                self.assertIsNone(record["lane_change_direction"])
                self.assertEqual(record["path_points"] , record["center_points"])
                self.assertEqual(record["lane_change_available"] , {side: record["lane_available"][side] for side in ("L" , "R")})
        for index in (4 , 8 , 12):
            self.assertTrue(records[index]["lane_change_exiting"])
        self.assertEqual(records[15]["lane_change_available"] , {"L": False , "R": True})
        self.assertEqual(records[15]["lane_available"] , {"current": True , "L": False , "R": True})
        self.assertEqual(records[15]["control_priority"] , "center")
        self.assertTrue(records[3]["path_points"])
        self.assertEqual(records[3]["center_points"] , [])
        controller.close.assert_called_once()
        telemetry.close.assert_called_once()
        logger.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

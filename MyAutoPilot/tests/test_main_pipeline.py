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
                                "mov_detector" , "lane_refiner" , "steering" ,
                                "auto_brakes" , "throttle" , "logger" , "scs_telemetry")}
        controller = Mock()
        aliases["input_controller"] = types.SimpleNamespace(InputController = Mock(return_value = controller))
        spec = importlib.util.spec_from_file_location("main_pipeline_under_test" , Path(__file__).resolve().parents[1] / "main.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules , aliases):
            spec.loader.exec_module(module)

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
        logger = Mock()
        logger.frame_id = 0
        records = []

        def write(timestamp , data):
            records.append(json.loads(json.dumps(data)))
            logger.frame_id += 1

        logger.write.side_effect = write
        module.Steering.reset()
        try:
            with ExitStack() as stack:
                for name , value in (("ScreenCapture" , capture) , ("PixelClassifierV2" , classifier) ,
                                     ("ObjectDetectorLite" , detector) , ("MovDetector" , movement) ,
                                     ("Logger" , logger) , ("SCSTelemetry" , telemetry)):
                    stack.enter_context(patch.object(module , name , return_value = value))
                stack.enter_context(patch.object(module.cv2 , "imshow"))
                stack.enter_context(patch.object(module.cv2 , "destroyAllWindows"))
                stack.enter_context(patch.object(module.cv2 , "waitKey" , side_effect = [-1] * (frame_count - 1) + [ord("q")]))
                stack.enter_context(patch.object(module.AutoBrakes , "brake" , side_effect = [0.0] * 5 + [0.5] + [0.0] * (frame_count - 6)))
                output = stack.enter_context(patch("builtins.print"))
                fps = stack.enter_context(patch.object(module.Draw , "show_fps"))
                speed = stack.enter_context(patch.object(module.Draw , "show_speed"))
                throttle = stack.enter_context(patch.object(module.Draw , "show_throttle"))
                module.main()
        finally:
            module.Steering.reset()
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


if __name__ == "__main__":
    unittest.main()

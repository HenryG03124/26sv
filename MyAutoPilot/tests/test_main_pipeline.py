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
    def test_three_frames_wire_telemetry_and_fps_without_devices(self):
        aliases = {name: importlib.import_module("MyAutoPilot." + name)
                   for name in ("screen_capture" , "draw" , "pixel_classifier_v2" , "object_detector" ,
                                "mov_detector" , "lane_refiner" , "steering" ,
                                "auto_brakes" , "logger" , "scs_telemetry")}
        controller = Mock()
        aliases["input_controller"] = types.SimpleNamespace(InputController = Mock(return_value = controller))
        spec = importlib.util.spec_from_file_location("main_pipeline_under_test" , Path(__file__).resolve().parents[1] / "main.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules , aliases):
            spec.loader.exec_module(module)

        mask = np.zeros((352 , 640) , dtype = np.uint8)
        for y in range(module.y_far , module.y_near + 1):
            mask[y , 220] = 3
            mask[y , 440] = 3
        capture = Mock()
        capture.capture_window.return_value = np.zeros((352 , 640 , 3) , dtype = np.uint8)
        capture.native_resolution.return_value = (1280 , 704)
        classifier = Mock()
        classifier.predict.return_value = mask
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
                stack.enter_context(patch.object(module.cv2 , "waitKey" , side_effect = [-1 , -1 , ord("q")]))
                fps = stack.enter_context(patch.object(module.Draw , "show_fps"))
                speed = stack.enter_context(patch.object(module.Draw , "show_speed"))
                module.main()
        finally:
            module.Steering.reset()
        self.assertEqual(len(records) , 3)
        self.assertEqual(capture.capture_window.call_count , 3)
        self.assertEqual(movement.update.call_count , 3)
        for call in movement.update.call_args_list:
            self.assertEqual(call.args[:3] , ([] , [] , []))
            self.assertEqual(call.args[4] , module.spd)
        capture.capture_window.assert_called_with("Euro Truck Simulator 2")
        capture.capture_frame.assert_not_called()
        self.assertIsNone(records[0]["fps"])
        self.assertGreater(records[1]["fps"] , 0)
        self.assertEqual(records[2]["steering_debug"]["speed_kmh"] , 45)
        self.assertFalse(records[2]["steering_debug"]["speed_fallback"])
        for record in records:
            self.assertEqual(record["center_points"] , [[330 , y] for y in range(module.y_near , module.y_far - 1 , -1)])
            self.assertNotIn("road_center_mode" , record)
            self.assertNotIn("timings_ms" , record)
            self.assertNotIn("timings_frame_id" , record)
        self.assertEqual(fps.call_count , 3)
        self.assertEqual(speed.call_count , 3)
        self.assertEqual(controller.steering_controller.call_count , 3)
        telemetry.close.assert_called_once()
        logger.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

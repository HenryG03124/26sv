import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from MyAutoPilot.logger import Logger
from MyAutoPilot.mov_detector import MovDetector
from MyAutoPilot.road_center import RoadCenter


class LoggerTests(unittest.TestCase):
    def test_frames_are_written_immediately_with_independent_values(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("MyAutoPilot.logger.time.monotonic" , return_value = 100.0):
                logger = Logger(directory)
            try:
                data = {"trackings": {7: {"box": [290.0 , 224.0 , 350.0 , 280.0]}} , "ttc": None}
                logger.write(100.1 , data)
                path = Path(logger.file.name)
                first = json.loads(path.read_text(encoding = "utf-8"))
                self.assertEqual(first["frame_id"] , 0)
                self.assertAlmostEqual(first["elapsed"] , 0.1)
                self.assertEqual(first["timestamp"] , 100.1)

                data["trackings"][7]["box"][1] = 200.0
                logger.write(100.2 , data)
                records = [json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines()]
                self.assertEqual([record["frame_id"] for record in records] , [0 , 1])
                self.assertEqual(records[0]["data"]["trackings"]["7"]["box"][1] , 224.0)
                self.assertEqual(records[1]["data"]["trackings"]["7"]["box"][1] , 200.0)
                self.assertIsNone(records[0]["data"]["ttc"])
            finally:
                logger.close()
            self.assertTrue(logger.file.closed)

    def test_real_tracking_and_both_center_modes_are_json_serializable(self):
        mask = np.ones((352 , 640) , dtype = np.uint8)
        mask[: , 200] = 3
        mask[: , 440] = 3
        left_points = [[200 , y] for y in range(188 , 293)]
        right_points = [[440 , y] for y in range(188 , 293)]
        trackings = MovDetector().update([[290 , 224 , 350 , 280]] , [0.9] , [1] , 100.0 , 70)
        centers = [
            RoadCenter.detect_from_lr(mask , left_points , right_points) ,
            RoadCenter.detect_from_mask(mask , 188 , 292) ,
        ]
        with tempfile.TemporaryDirectory() as directory:
            logger = Logger(Path(directory) / "logs")
            try:
                for center_points in centers:
                    logger.write(logger.start_time , {
                        "left_points": left_points ,
                        "right_points": right_points ,
                        "center_points": RoadCenter.center_points_filter(center_points , 10) ,
                        "trackings": trackings ,
                    })
                records = [json.loads(line) for line in Path(logger.file.name).read_text(encoding = "utf-8").splitlines()]
                self.assertEqual(len(records) , 2)
                self.assertEqual(records[1]["data"]["trackings"]["0"]["status"] , "tracked")
            finally:
                logger.close()


if __name__ == "__main__":
    unittest.main()

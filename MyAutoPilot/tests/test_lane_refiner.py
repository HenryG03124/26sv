import unittest

import numpy as np

from MyAutoPilot.lane_refiner import LaneRefiner


class LaneRefinerTests(unittest.TestCase):
    def test_width_updates_after_five_tracking_frames(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        mask = np.zeros((352 , 640) , dtype = np.uint8)
        mask[200 : 321 , 200] = 3
        mask[200 : 321 , 440] = 3

        for frame in range(1 , 6):
            cl_lane_points , cr_lane_points , _ , _ = refiner.refine(mask , 2 , 10 , 30)
            center_points = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
            self.assertEqual(refiner.tracking_frames , frame)
            self.assertEqual(center_points , [[320 , y] for y in range(320 , 199 , -1)])
            if frame < 5:
                self.assertEqual(refiner.prev_lane_width , [])
            else:
                self.assertEqual(refiner.prev_lane_width , [[240 , y] for y in range(320 , 199 , -1)])

        mask[: , 440] = 0
        mask[200 : 321 , 460] = 3
        cl_lane_points , cr_lane_points , _ , _ = refiner.refine(mask , 2 , 10 , 30 , return_sample = True)[4 : ]
        refiner.calculate_center_points(cl_lane_points , cr_lane_points)
        self.assertEqual(refiner.tracking_frames , 6)
        self.assertEqual(refiner.prev_lane_width , [[260 , y] for y in range(320 , 199 , -1)])

    def test_missing_lane_restarts_count_and_preserves_width(self):
        for missing in ("CL" , "CR" , "both"):
            with self.subTest(missing = missing):
                refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
                mask = np.zeros((352 , 640) , dtype = np.uint8)
                mask[200 : 321 , 200] = 3
                mask[200 : 321 , 440] = 3
                for frame in range(5):
                    cl_lane_points , cr_lane_points , _ , _ = refiner.refine(mask , 2 , 10 , 30)
                    refiner.calculate_center_points(cl_lane_points , cr_lane_points)
                previous_width = [point[:] for point in refiner.prev_lane_width]

                missing_mask = mask.copy()
                if missing in ("CL" , "both"):
                    missing_mask[: , 200] = 0
                if missing in ("CR" , "both"):
                    missing_mask[: , 440] = 0
                cl_lane_points , cr_lane_points , _ , _ = refiner.refine(missing_mask , 2 , 10 , 30)
                center_points = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
                self.assertEqual(refiner.tracking_frames , 0)
                self.assertEqual(refiner.prev_lane_width , previous_width)
                if missing == "both":
                    self.assertEqual(center_points , [])
                else:
                    self.assertEqual(center_points , [[320 , y] for y in range(320 , 199 , -1)])

                mask[: , 440] = 0
                mask[200 : 321 , 460] = 3
                for frame in range(1 , 6):
                    cl_lane_points , cr_lane_points , _ , _ = refiner.refine(mask , 2 , 10 , 30)
                    refiner.calculate_center_points(cl_lane_points , cr_lane_points)
                    self.assertEqual(refiner.tracking_frames , frame)
                    if frame < 5:
                        self.assertEqual(refiner.prev_lane_width , previous_width)
                    else:
                        self.assertEqual(refiner.prev_lane_width , [[260 , y] for y in range(320 , 199 , -1)])


if __name__ == "__main__":
    unittest.main()

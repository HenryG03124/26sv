import unittest
from unittest.mock import patch

import numpy as np

from MyAutoPilot.auto_brakes import AutoBrakes
from MyAutoPilot.lane_refiner import LaneRefiner


class LaneRefinerTests(unittest.TestCase):
    def test_lanes_are_unavailable_before_first_frame(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        self.assertEqual(refiner.lane_available , {"current": False , "L": False , "R": False})

    def test_missing_boundaries_reset_each_lane_available(self):
        cases = [
            ([100] , {"current": True , "L": False , "R": True}) ,
            ([540] , {"current": True , "L": True , "R": False}) ,
            ([100 , 220] , {"current": False , "L": False , "R": True}) ,
            ([420 , 540] , {"current": False , "L": True , "R": False}) ,
            ([100 , 220 , 420 , 540] , {"current": False , "L": False , "R": False})
        ]
        for missing_x_values , expected in cases:
            with self.subTest(missing_x_values = missing_x_values):
                refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
                mask = np.zeros((352 , 640) , dtype = np.uint8)
                for x in (100 , 220 , 420 , 540):
                    mask[200 : 321 , x] = 3
                refiner.refine(mask , 2 , 10 , 30)
                self.assertEqual(refiner.lane_available , {"current": True , "L": True , "R": True})

                for x in missing_x_values:
                    mask[: , x] = 0
                refiner.refine(mask , 2 , 10 , 30 , return_sample = True)
                self.assertEqual(refiner.lane_available , expected)

    def test_neighbour_lanes_must_cover_far_target(self):
        refiner = LaneRefiner(200 , 320 , 10 , 260 , 300)
        mask = np.zeros((352 , 640) , dtype = np.uint8)
        for x in (100 , 220 , 420 , 540):
            mask[200 : 321 , x] = 3
        _ , _ , nl_lane_points , nr_lane_points = refiner.refine(mask , 2 , 10 , 30)
        self.assertTrue(nl_lane_points)
        self.assertTrue(nr_lane_points)
        self.assertEqual(refiner.lane_available , {"current": True , "L": False , "R": False})

    def test_crossed_fitted_boundaries_are_unavailable(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        mask = np.zeros((352 , 640) , dtype = np.uint8)
        cases = [
            ([[220 , 200] , [430 , 248]] , [[420 , 200] , [420 , 248]] , [[100 , 200] , [100 , 248]] , [[540 , 200] , [540 , 248]] , {"current": False , "L": True , "R": True}) ,
            ([[220 , 200] , [220 , 248]] , [[420 , 200] , [420 , 248]] , [[100 , 200] , [220 , 248]] , [[540 , 200] , [540 , 248]] , {"current": True , "L": False , "R": True}) ,
            ([[220 , 200] , [220 , 248]] , [[420 , 200] , [420 , 248]] , [[100 , 200] , [100 , 248]] , [[540 , 200] , [400 , 248]] , {"current": True , "L": True , "R": False})
        ]
        for cl_lane_points , cr_lane_points , nl_lane_points , nr_lane_points , expected in cases:
            with self.subTest(expected = expected):
                with patch.object(refiner , "fit_lane_points" , side_effect = [cl_lane_points , cr_lane_points , nl_lane_points , nr_lane_points]):
                    refiner.refine(mask , 2 , 10 , 30)
                self.assertEqual(refiner.lane_available , expected)

    def test_cached_width_and_brake_lanes_do_not_restore_available(self):
        refiner = LaneRefiner(200 , 320 , 10 , 200 , 248)
        brakes = AutoBrakes()
        mask = np.zeros((352 , 640) , dtype = np.uint8)
        mask[200 : 321 , 220] = 3
        mask[200 : 321 , 420] = 3
        for frame in range(5):
            cl_lane_points , cr_lane_points , _ , _ = refiner.refine(mask , 2 , 10 , 30)
            refiner.calculate_center_points(cl_lane_points , cr_lane_points)
            brakes.update_lanes(cl_lane_points , cr_lane_points , frame * 0.1 , refiner.lane_available["current"])

        mask[: , 220] = 0
        cl_lane_points , cr_lane_points , _ , _ = refiner.refine(mask , 2 , 10 , 30)
        center_points = refiner.calculate_center_points(cl_lane_points , cr_lane_points)
        brakes.update_lanes(cl_lane_points , cr_lane_points , 0.5 , refiner.lane_available["current"])
        self.assertTrue(center_points)
        self.assertTrue(refiner.prev_lane_width)
        self.assertEqual(brakes.lane_status , "held")
        self.assertTrue(brakes.cl_lane_points)
        self.assertFalse(refiner.lane_available["current"])
        self.assertFalse(refiner.lane_available["L"])

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

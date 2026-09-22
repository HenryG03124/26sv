import unittest

import numpy as np

from MyAutoPilot.lane_refiner import LaneRefiner


class LaneRefinerTests(unittest.TestCase):
    def test_missing_boundary_does_not_produce_center(self):
        for sides in ((), (200,), (440,)):
            with self.subTest(sides = sides):
                mask = np.zeros((352 , 640) , dtype = np.uint8)
                for x in sides:
                    mask[: , x] = 3
                refiner = LaneRefiner(188 , 292 , 10)
                llane_points , rlane_points = refiner.refine(mask , 2 , 10)
                center = refiner.calculate_center_points(llane_points , rlane_points)
                self.assertEqual(bool(llane_points) , 200 in sides)
                self.assertEqual(bool(rlane_points) , 440 in sides)
                self.assertEqual(center , [])

    def test_center_preserves_common_rows_order_and_midpoints(self):
        refiner = LaneRefiner(188 , 292 , 10)
        llane_points = [[200 , 292], [280 , 291], [204 , 290], [206 , 289], [208 , 288]]
        rlane_points = [[440 , 292], [520 , 291], [444 , 290], [446 , 289]]
        self.assertEqual(refiner.calculate_center_points(llane_points , rlane_points) ,
                         [[320 , 292], [400 , 291], [324 , 290], [326 , 289]])
        self.assertEqual(llane_points[1] , [280 , 291])
        self.assertEqual(refiner.calculate_center_points([[200 , 292]] , [[440 , 292]]) , [[320 , 292]])
        self.assertEqual(refiner.calculate_center_points([[200 , 292]] , [[440 , 291]]) , [])


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import Mock , patch

import numpy as np
import torch

from MyAutoPilot import pixel_classifier_v2
from MyAutoPilot.pixel_classifier_v2 import PixelClassifierV2


class PixelClassifierV2Tests(unittest.TestCase):
    def test_preprocess_resizes_and_converts_bgr_to_rgb(self):
        classifier = PixelClassifierV2.__new__(PixelClassifierV2)
        frame = np.full((40 , 80 , 3) , [10 , 20 , 30] , dtype = np.uint8)
        with patch.object(pixel_classifier_v2 , "device" , "cpu"):
            model_input = classifier.preprocess(frame)
        self.assertEqual(tuple(model_input.shape) , (1 , 3 , 352 , 640))
        self.assertEqual(model_input.dtype , torch.float32)
        np.testing.assert_allclose(model_input[0 , : , 0 , 0].numpy() , np.array([30 , 20 , 10]) / 255 , rtol = 1e-6)

    def test_lane_threshold_and_vehicle_priority(self):
        classifier = PixelClassifierV2.__new__(PixelClassifierV2)
        classifier.preprocess = Mock(return_value = torch.zeros((1 , 3 , 2 , 2)))
        classifier.model = Mock()
        road_output = torch.tensor([[[[4.0 , 0.0] , [0.0 , 0.0]] , [[0.0 , 4.0] , [0.0 , 4.0]] , [[0.0 , 0.0] , [4.0 , 0.0]]]])
        lane_output = torch.tensor([[[[0.0 , 0.0] , [0.0 , 4.0]] , [[4.0 , 4.0] , [4.0 , 0.0]]]])
        classifier.model.head.side_effect = lambda model_input , output , task: road_output if task == "road" else lane_output

        mask = classifier.predict(None , 0.7)
        self.assertEqual(mask.dtype , np.uint8)
        np.testing.assert_array_equal(mask , [[3 , 3] , [2 , 1]])
        np.testing.assert_array_equal(classifier.predict(None , 0.99) , [[0 , 1] , [2 , 1]])


if __name__ == "__main__":
    unittest.main()

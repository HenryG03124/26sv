import numpy as np
import cv2
import torch

from Models.pixel_classifier_v2 import UNet

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

INPUT_HEIGHT = 352
INPUT_WIDTH = 640
INPUT_SIZE = (INPUT_WIDTH , INPUT_HEIGHT)

class PixelClassifierV2():
    def __init__(self):
        self.model = UNet().to(device)

    def load_model(self , model_path):
        self.model.load_state_dict(torch.load(model_path , map_location = device , weights_only = True))
        self.model.eval()

    def preprocess(self , frame):
        model_input = cv2.resize(frame , INPUT_SIZE , interpolation = cv2.INTER_LINEAR)
        model_input = cv2.cvtColor(model_input , cv2.COLOR_BGR2RGB)
        model_input = torch.from_numpy(model_input).permute(2 , 0 , 1).contiguous().to(dtype = torch.float32).div(255.0).unsqueeze(0).to(device)
        return model_input

    def predict(self , frame , lane_prob_threshold):
        model_input = self.preprocess(frame)

        with torch.no_grad():
            output = self.model(model_input)
            road_output = self.model.head(model_input , output , "road")
            lane_output = self.model.head(model_input , output , "lane")
            mask = road_output.argmax(dim = 1).squeeze(0)
            lane_probability = torch.softmax(lane_output , dim = 1)[: , 1].squeeze(0)
            lane_mask = lane_probability > lane_prob_threshold
            mask[lane_mask & (mask != 2)] = 3
            mask = mask.cpu().numpy().astype(np.uint8)

        return mask #h * w 2d

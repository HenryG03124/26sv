from pathlib import Path

import numpy as np
from PIL import Image
import torch

from Models.pc_v2 import UNet

SCRIPT_DIR = Path(__file__).resolve().parent

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = UNet().to(device)

model.load_state_dict(torch.load(SCRIPT_DIR / "pc_model_v2_a2d2.pth" , map_location = device , weights_only = True))
model.eval()

image = Image.open(SCRIPT_DIR / "test.jpg").convert("RGB")
original_size = image.size
image = image.resize((640 , 352) , Image.Resampling.BILINEAR)

image_array = np.array(image , dtype = np.uint8 , copy = True)
image_tensor = torch.from_numpy(image_array).permute(2 , 0 , 1).float().div(255.0)
image_tensor = image_tensor.unsqueeze(0).to(device)

lane_prob_threshold = 0.7

with torch.no_grad():
    features = model(image_tensor)
    road_output = model.head(image_tensor , features , "road")
    lane_output = model.head(image_tensor , features , "lane")
    mask = road_output.argmax(dim = 1).squeeze(0)
    lane_probability = torch.softmax(lane_output , dim = 1)[: , 1]
    lane_mask = (lane_probability > lane_prob_threshold).squeeze(0)
    mask[(lane_mask == 1) & (mask != 2)] = 3
    mask = mask.cpu().numpy().astype(np.uint8)

mask_image = Image.fromarray(mask , mode = "P")
mask_image.putpalette([0 , 0 , 0 , 0 , 255 , 0 , 255 , 0 , 0 , 255 , 255 , 0] + [0 , 0 , 0] * 252)
mask_image = mask_image.resize(original_size , Image.Resampling.NEAREST)
mask_image.save(SCRIPT_DIR / "test_mask_v2_a2d2.png")

print("mask saved: test_mask_v2_a2d2.png")

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = nn.Sequential(
    nn.Conv2d(3 , 6 , kernel_size = 5 , stride = 1 , padding = 2) ,
    nn.ReLU() ,
    nn.MaxPool2d(kernel_size = 2 , stride = 2) ,
    nn.Conv2d(6 , 16 , kernel_size = 5 , stride = 1 , padding = 2) ,
    nn.ReLU() ,
    nn.MaxPool2d(kernel_size = 2 , stride = 2) ,
    nn.Conv2d(16 , 32 , kernel_size = 3 , stride = 1 , padding = 1) ,
    nn.ReLU() ,
    nn.ConvTranspose2d(32 , 16 , kernel_size = 2 , stride = 2) ,
    nn.ReLU() ,
    nn.ConvTranspose2d(16 , 3 , kernel_size = 2 , stride = 2)
).to(device)

model.load_state_dict(torch.load("pc_model_LiteCNN.pth" , map_location = device , weights_only = True))
model.eval()

image = Image.open("test.jpg").convert("RGB")
original_size = image.size
image = image.resize((448 , 256) , Image.Resampling.BILINEAR)

image_array = np.array(image , dtype = np.uint8 , copy = True)
image_tensor = torch.from_numpy(image_array).permute(2 , 0 , 1).float().div(255.0)
image_tensor = image_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)
    mask = output.argmax(dim = 1).squeeze(0).cpu().numpy().astype(np.uint8)

mask_image = Image.fromarray(mask , mode = "P")
mask_image.putpalette([0 , 0 , 0 , 0 , 255 , 0 , 255 , 0 , 0] + [0 , 0 , 0] * 253)
mask_image = mask_image.resize(original_size , Image.Resampling.NEAREST)
mask_image.save("test_mask_LiteCNN.png")

print("mask saved: test_mask_LiteCNN.png")

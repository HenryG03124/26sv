import numpy as np
from PIL import Image
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class BasicBlock(nn.Module):
    def __init__(self , in_channels , out_channels , stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels , out_channels , kernel_size = 3 , stride = stride , padding = 1 , bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels , out_channels , kernel_size = 3 , stride = 1 , padding = 1 , bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels , out_channels , kernel_size = 1 , stride = stride , bias = False) ,
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self , x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

ResNet18_encoder = nn.Sequential(
    nn.Conv2d(3 , 64 , kernel_size = 7 , stride = 2 , padding = 3 , bias = False) ,
    nn.BatchNorm2d(64) ,
    nn.ReLU(inplace = True) ,
    nn.MaxPool2d(kernel_size = 3 , stride = 2 , padding = 1) ,
    BasicBlock(64 , 64 , stride = 1) ,
    BasicBlock(64 , 64 , stride = 1) ,

    BasicBlock(64 , 128 , stride = 2) ,
    BasicBlock(128 , 128 , stride = 1) ,

    BasicBlock(128 , 256 , stride = 2) ,
    BasicBlock(256 , 256 , stride = 1) ,

    BasicBlock(256 , 512 , stride = 2) ,
    BasicBlock(512 , 512 , stride = 1)
)

decoder = nn.Sequential(
    nn.ConvTranspose2d(512 , 256 , kernel_size = 2 , stride = 2) ,
    nn.BatchNorm2d(256) ,
    nn.ReLU(inplace = True) ,

    nn.ConvTranspose2d(256 , 128 , kernel_size = 2 , stride = 2) ,
    nn.BatchNorm2d(128) ,
    nn.ReLU(inplace = True) ,

    nn.ConvTranspose2d(128 , 64 , kernel_size = 2 , stride = 2) ,
    nn.BatchNorm2d(64) ,
    nn.ReLU(inplace = True) ,

    nn.ConvTranspose2d(64 , 64 , kernel_size = 2 , stride = 2) ,
    nn.BatchNorm2d(64) ,
    nn.ReLU(inplace = True) ,

    nn.Conv2d(64 , 32 , kernel_size = 1 , stride = 1) ,
    nn.BatchNorm2d(32) ,
    nn.ReLU(inplace = True) ,
    
    nn.ConvTranspose2d(32 , 3 , kernel_size = 2 , stride = 2)
)

model = nn.Sequential(
    ResNet18_encoder ,
    decoder
).to(device)

model.load_state_dict(torch.load("pc_model_ResNet.pth" , map_location = device , weights_only = True))
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
mask_image.save("test_mask_ResNet.png")

print("mask saved: test_mask_ResNet.png")

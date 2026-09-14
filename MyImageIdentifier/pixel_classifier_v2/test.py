import numpy as np
from PIL import Image
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

class EncoderBlock(nn.Module):
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

class DecoderBlock(nn.Module):
    def __init__(self , in_channels , out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels , out_channels , kernel_size = 3 , padding = 1 , bias = False) ,
            nn.BatchNorm2d(out_channels) ,
            nn.ReLU(inplace = True) ,

            nn.Conv2d(out_channels , out_channels , kernel_size = 3 , padding = 1 , bias = False) ,
            nn.BatchNorm2d(out_channels) ,
            nn.ReLU(inplace = True)
        )

    def forward(self , x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.root = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size = 7 , stride = 2 , padding = 3 , bias = False) , #320 * 176
            nn.BatchNorm2d(64) ,
            nn.ReLU(inplace = True)
        )

        self.pool = nn.MaxPool2d(kernel_size = 3 , stride = 2 , padding = 1) #160 * 88

        self.encoder1 = nn.Sequential(
            EncoderBlock(64 , 64 , stride = 1) ,
            EncoderBlock(64 , 64 , stride = 1)
        )

        self.encoder2 = nn.Sequential(
            EncoderBlock(64 , 128 , stride = 2) , #80 * 44
            EncoderBlock(128 , 128 , stride = 1)
        )

        self.encoder3 = nn.Sequential(
            EncoderBlock(128 , 256 , stride = 2) , #40 * 22
            EncoderBlock(256 , 256 , stride = 1)
        )

        self.encoder4 = nn.Sequential(
            EncoderBlock(256 , 512 , stride = 2) , #20 * 11
            EncoderBlock(512 , 512 , stride = 1)
        )

        self.up4 = nn.ConvTranspose2d(512 , 256 , kernel_size = 2 , stride = 2)
        self.decoder4 = DecoderBlock(512 , 256)

        self.up3 = nn.ConvTranspose2d(256 , 128 , kernel_size = 2 , stride = 2)
        self.decoder3 = DecoderBlock(256 , 128)

        self.up2 = nn.ConvTranspose2d(128 , 64 , kernel_size = 2 , stride = 2)
        self.decoder2 = DecoderBlock(128 , 64)

        self.up1 = nn.ConvTranspose2d(64 , 32 , kernel_size = 2 , stride = 2)
        self.decoder1 = DecoderBlock(96 , 32)

        self.road_head = nn.Sequential(
            nn.ConvTranspose2d(32 , 16 , kernel_size = 2 , stride = 2) ,
            nn.ReLU(inplace = True) ,
            nn.Conv2d(16 , 3 , kernel_size = 1)
        )

        self.lane_head = nn.Sequential(
            nn.Conv2d(24 , 16 , kernel_size = 3 , padding = 1 , bias = False) ,
            nn.BatchNorm2d(16) ,
            nn.ReLU(inplace = True) ,
            nn.Conv2d(16 , 2 , kernel_size = 1)
        )

        self.lane_up = nn.Sequential(
            nn.ConvTranspose2d(32 , 16 , kernel_size = 2 , stride = 2 , bias = False) ,
            nn.BatchNorm2d(16) ,
            nn.ReLU(inplace = True)
        )

        self.lane_detail = nn.Sequential(
            nn.Conv2d(3 , 8 , kernel_size = 3 , padding = 1 , bias = False) ,
            nn.BatchNorm2d(8) ,
            nn.ReLU(inplace = True)
        )

    def forward(self , x):
        root = self.root(x) #320 * 176 ch = 64
    
        encoder1 = self.pool(root) #160 * 88 ch = 64
        encoder1 = self.encoder1(encoder1)
    
        encoder2 = self.encoder2(encoder1) #80 * 44 ch = 128
        encoder3 = self.encoder3(encoder2) #40 * 22 ch = 256
        encoder4 = self.encoder4(encoder3) #20 * 11 ch = 512
    
        decoder4 = self.up4(encoder4) #40 * 22 ch = 256
        decoder4 = torch.cat((decoder4 , encoder3) , dim = 1) #40 * 22 ch = 512
        #join tensor in channel dimension ([batch , ch , h , w])
        decoder4 = self.decoder4(decoder4) #40 * 22 ch = 256
    
        decoder3 = self.up3(decoder4) #80 * 44 ch = 128
        decoder3 = torch.cat((decoder3 , encoder2) , dim = 1) #80 * 44 ch = 256
        decoder3 = self.decoder3(decoder3) #80 * 44 ch = 128
    
        decoder2 = self.up2(decoder3) #160 * 88 ch = 64
        decoder2 = torch.cat((decoder2 , encoder1) , dim = 1) #160 * 88 ch = 128
        decoder2 = self.decoder2(decoder2) #160 * 88 ch = 64
    
        decoder1 = self.up1(decoder2) #320 * 176 ch = 32
        decoder1 = torch.cat((decoder1 , root) , dim = 1) #320 * 176 ch = 96
        decoder1 = self.decoder1(decoder1) #320 * 176 ch = 32
    
        return decoder1

    def head(self , x , decoder1 , task):
        if task == "road":
            return self.road_head(decoder1) #640 * 352 ch = 3
        else:
            lane_semantic = self.lane_up(decoder1) #640 * 352 ch = 16 processed
            lane_detail = self.lane_detail(x) #640 * 352 ch = 8 unprocessed
            lane_feature = torch.cat((lane_semantic , lane_detail) , dim = 1) #640 * 352 ch = 24
            return self.lane_head(lane_feature) #640 * 352 ch = 2

model = UNet().to(device)

model.load_state_dict(torch.load("pc_model_v2.pth" , map_location = device , weights_only = True))
model.eval()

image = Image.open("test.jpg").convert("RGB")
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
mask_image.save("test_mask.png")

print("mask saved: test_mask.png")

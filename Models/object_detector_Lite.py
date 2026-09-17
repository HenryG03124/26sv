#Lite detection network; numeric Sequential keys preserve existing checkpoints.

import torch.nn as nn

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

class ResNet(nn.Sequential):
    def __init__(self):
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

            BasicBlock(256 , 512 , stride = 1) ,
            BasicBlock(512 , 512 , stride = 1)
        )

        detector = nn.Sequential(
            nn.Conv2d(512 , 256 , kernel_size = 3 , padding = 1 , bias = False) ,
            nn.BatchNorm2d(256) ,
            nn.ReLU(inplace = True) ,

            nn.Conv2d(256 , 256 , kernel_size = 3 , padding = 1 , bias = False) ,
            nn.BatchNorm2d(256) ,
            nn.ReLU(inplace = True) ,

            nn.Conv2d(256 , 7 , kernel_size = 1 , stride = 1)
        )
        super().__init__(ResNet18_encoder , detector)

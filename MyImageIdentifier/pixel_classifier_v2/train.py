import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as tud
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent / "datasets"

from MyImageIdentifier.pixel_classifier_v2.bdd_pc_v2_ds import BDDLaneDataset

road_train_dataset = BDDLaneDataset(DATASETS_DIR / "processed_pc_ds/train_pairs.csv")
road_val_dataset = BDDLaneDataset(DATASETS_DIR / "processed_pc_ds/val_pairs.csv")
full_lane_train_dataset = BDDLaneDataset(DATASETS_DIR / "processed_pc_v2_ds/train_pairs.csv")
lane_val_dataset = BDDLaneDataset(DATASETS_DIR / "processed_pc_v2_ds/val_pairs.csv")

lane_train_sample_size = min(15000 , len(full_lane_train_dataset))
lane_sample_generator = torch.Generator().manual_seed(42)
lane_train_indices = torch.randperm(len(full_lane_train_dataset) , generator = lane_sample_generator)[ : lane_train_sample_size].tolist()
lane_train_dataset = tud.Subset(full_lane_train_dataset , lane_train_indices)

batch_size = 16
road_train_loader = tud.DataLoader(road_train_dataset , batch_size = batch_size , shuffle = True)
road_val_loader = tud.DataLoader(road_val_dataset , batch_size = batch_size , shuffle = False)
lane_train_loader = tud.DataLoader(lane_train_dataset , batch_size = batch_size , shuffle = True)
lane_val_loader = tud.DataLoader(lane_val_dataset , batch_size = batch_size , shuffle = False)

device = "cuda" if torch.cuda.is_available() else "cpu"

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

class LaneLoss(nn.Module):
    def __init__(self , class_weights , ce_weight , dice_weight):
        super().__init__()

        self.ce = nn.CrossEntropyLoss(weight = class_weights , ignore_index = 255)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = 1.0 #avoid zero division

    def forward(self , outputs , masks):
        ce_loss = self.ce(outputs , masks)

        lane_probability = torch.softmax(outputs , dim = 1)[: , 1]
        valid = masks != 255

        lane_probability = lane_probability[valid]
        lane_target = (masks[valid] == 1).to(torch.float32)

        intersection = (lane_probability * lane_target).sum() #keep only when target = 1 (soft intersection)

        dice_score = (2.0 * intersection + self.smooth) / (lane_probability.sum() + lane_target.sum() + self.smooth)
        dice_loss = 1 - dice_score

        lane_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        return lane_loss

model = UNet().to(device)

road_criterion = nn.CrossEntropyLoss(
    weight = torch.tensor([1.0 , 1.5 , 3.0] , dtype = torch.float32 , device = device) ,
    ignore_index = 255
)
lane_criterion = LaneLoss(
    class_weights = torch.tensor([1.0 , 5.0] , dtype = torch.float32 , device = device) , 
    ce_weight = 1.0 , dice_weight = 0.5
)
optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)
lane_prob_threshold = 0.7

print("current device:" , device)
print("road/car training samples:" , len(road_train_dataset))
print("lane training samples:" , len(lane_train_dataset))

def train_task(loader , task , criterion , classes):
    model.train()
    total_loss = 0
    inter = torch.zeros(classes , dtype = torch.float64 , device = device)
    union = torch.zeros(classes , dtype = torch.float64 , device = device)

    for images , masks in loader:
        images , masks = images.to(device) , masks.to(device)
        optimizer.zero_grad()
        features = model(images)
        outputs = model.head(images , features , task) #[batch , ch , 352 , 640]
        loss = criterion(outputs , masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

        if task == "road":
            predictions = outputs.argmax(dim = 1)
        else:
            lane_probability = torch.softmax(outputs , dim = 1)[: , 1]
            predictions = (lane_probability > lane_prob_threshold).to(torch.int64)

        valid = masks != 255
        for class_id in range(classes):
            predc = (predictions == class_id) & valid
            truec = (masks == class_id) & valid
            inter[class_id] += (predc & truec).sum()
            union[class_id] += (predc | truec).sum()

    IoU = torch.where(union > 0 , inter / union , torch.tensor(float("nan") , device = device)
)
    return total_loss / len(loader.dataset) , IoU , torch.nanmean(IoU)

def evaluate_task(loader , task , criterion , classes):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    inter = torch.zeros(classes , dtype = torch.float64 , device = device)
    union = torch.zeros(classes , dtype = torch.float64 , device = device)

    with torch.no_grad():
        for images , masks in loader:
            images , masks = images.to(device) , masks.to(device)
            features = model(images)
            outputs = model.head(images , features , task)
            loss = criterion(outputs , masks)
            total_loss += loss.item() * images.size(0)

            if task == "road":
                predictions = outputs.argmax(dim = 1)
            else:
                lane_probability = torch.softmax(outputs , dim = 1)[: , 1]
                predictions = (lane_probability > lane_prob_threshold).to(torch.int64)

            valid = masks != 255
            for class_id in range(classes):
                predc = (predictions == class_id) & valid
                truec = (masks == class_id) & valid
                inter[class_id] += (predc & truec).sum()
                union[class_id] += (predc | truec).sum()

            correct += (predictions[valid] == masks[valid]).sum().item()
            total += valid.sum().item()

    IoU = torch.where(union > 0 , inter / union , torch.tensor(float("nan") , device = device))
    return (total_loss / len(loader.dataset) , correct / total , IoU , torch.nanmean(IoU))

loss_set = []
miou_set = []

for epoch in range(25):
    try:
        road_loss , road_IoU , road_mIoU = train_task(road_train_loader , "road" , road_criterion , 3)
        lane_loss , lane_IoU , lane_mIoU = train_task(lane_train_loader , "lane" , lane_criterion , 2)
        avg_loss = (road_loss + lane_loss) / 2
        train_mIoU = torch.nanmean(torch.stack((road_mIoU , lane_mIoU)))
        loss_set.append(avg_loss)
        miou_set.append(train_mIoU.item())

        print("\nepoch:" , epoch + 1 , "loss =" , avg_loss)
        print("train road-head background IoU:" , road_IoU[0].item())
        print("train road IoU:" , road_IoU[1].item())
        print("train car IoU:" , road_IoU[2].item())
        print("train lane-head background IoU:" , lane_IoU[0].item())
        print("train lane IoU:" , lane_IoU[1].item())
        print("train mIoU:" , train_mIoU.item())
    except KeyboardInterrupt:
        break

torch.save(model.state_dict() , "pc_model_v2.pth")
print("\nmodel saved: pc_model_v2.pth")

road_val_loss , road_accuracy , road_IoU , road_mIoU = evaluate_task(road_val_loader , "road" , road_criterion , 3)
lane_val_loss , lane_accuracy , lane_IoU , lane_mIoU = evaluate_task(lane_val_loader , "lane" , lane_criterion , 2)
avg_val_loss = (road_val_loss + lane_val_loss) / 2
mIoU = torch.nanmean(torch.stack((road_mIoU , lane_mIoU)))

print("\nroad/car validation loss:" , road_val_loss)
print("road/car validation accuracy:" , road_accuracy)
print("validation road-head background IoU:" , road_IoU[0].item())
print("validation road IoU:" , road_IoU[1].item())
print("validation car IoU:" , road_IoU[2].item())
print("lane validation loss:" , lane_val_loss)
print("lane validation accuracy:" , lane_accuracy)
print("validation lane-head background IoU:" , lane_IoU[0].item())
print("validation lane IoU:" , lane_IoU[1].item())
print("validation combined mIoU:" , mIoU.item())

epochs = range(1 , len(loss_set) + 1)
figure , axes = plt.subplots(1 , 2 , figsize = (12 , 5))

axes[0].plot(epochs , loss_set , color = "blue" , label = "Train Loss")
axes[0].axhline(avg_val_loss , color = "red" , linestyle = "--" , label = "Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Pixel Classifier V2 Loss Curve")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(epochs , miou_set , color = "green" , label = "Train mIoU")
axes[1].axhline(mIoU.item() , color = "orange" , linestyle = "--" , label = "Validation mIoU")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("mIoU")
axes[1].set_title("Pixel Classifier V2 mIoU Curve")
axes[1].grid(True)
axes[1].legend()

figure.tight_layout()
figure.savefig("figure.png" , dpi = 300 , bbox_inches = "tight")
plt.close(figure)
print("\nfigure saved: figure.png")

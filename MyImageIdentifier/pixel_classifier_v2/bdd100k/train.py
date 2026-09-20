import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as tud
from pathlib import Path

from Models.pc_v2 import UNet

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent.parent / "datasets"

from MyImageIdentifier.pixel_classifier_v2.bdd100k.pc_v2_bdd100k_ds import BDDLaneDataset

road_train_dataset = BDDLaneDataset(DATASETS_DIR / "processed_pc_ds_bdd100k/train_pairs.csv")
road_val_dataset = BDDLaneDataset(DATASETS_DIR / "processed_pc_ds_bdd100k/val_pairs.csv")
full_lane_train_dataset = BDDLaneDataset(DATASETS_DIR / "processed_pc_lane_ds_bdd100k/train_pairs.csv")
lane_val_dataset = BDDLaneDataset(DATASETS_DIR / "processed_pc_lane_ds_bdd100k/val_pairs.csv")

lane_train_sample_size = min(30000 , len(full_lane_train_dataset))
lane_sample_generator = torch.Generator().manual_seed(42)
lane_train_indices = torch.randperm(len(full_lane_train_dataset) , generator = lane_sample_generator)[ : lane_train_sample_size].tolist()
lane_train_dataset = tud.Subset(full_lane_train_dataset , lane_train_indices)

batch_size = 16
road_train_loader = tud.DataLoader(road_train_dataset , batch_size = batch_size , shuffle = True)
road_val_loader = tud.DataLoader(road_val_dataset , batch_size = batch_size , shuffle = False)
lane_train_loader = tud.DataLoader(lane_train_dataset , batch_size = batch_size , shuffle = True)
lane_val_loader = tud.DataLoader(lane_val_dataset , batch_size = batch_size , shuffle = False)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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

    for batch_index , (images , masks) in enumerate(loader):
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

        if batch_index == 0 or (batch_index + 1) % 100 == 0:
            print(task , "batch:" , batch_index + 1 , "/" , len(loader) , "loss:" , loss.item() , flush = True)
        yield #Let the other task update the shared network between batches.

    IoU = torch.where(union > 0 , inter / union , torch.tensor(float("nan") , device = device)
)
    yield total_loss / len(loader.dataset) , IoU , torch.nanmean(IoU)

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

for epoch in range(30):
    try:
        print("\nepoch:" , epoch + 1 , "training" , flush = True)
        road_training = train_task(road_train_loader , "road" , road_criterion , 3)
        lane_training = train_task(lane_train_loader , "lane" , lane_criterion , 2)
        road_steps = lane_steps = 0
        road_batches , lane_batches = len(road_train_loader) , len(lane_train_loader)
        # Interleave by progress: visit every batch once without a single-task tail.
        while road_steps < road_batches or lane_steps < lane_batches:
            if lane_steps == lane_batches or (road_steps < road_batches and road_steps / road_batches <= lane_steps / lane_batches):
                next(road_training)
                road_steps += 1
            else:
                next(lane_training)
                lane_steps += 1
        road_loss , road_IoU , road_mIoU = next(road_training)
        lane_loss , lane_IoU , lane_mIoU = next(lane_training)
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
        torch.save(model.state_dict() , SCRIPT_DIR / "pc_model_v2_bdd100k.pth")
    except KeyboardInterrupt:
        break

torch.save(model.state_dict() , SCRIPT_DIR / "pc_model_v2_bdd100k.pth")
print("\nmodel saved: pc_model_v2_bdd100k.pth")

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
axes[0].set_title("Pixel Classifier v2_bdd100k Loss Curve")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(epochs , miou_set , color = "green" , label = "Train mIoU")
axes[1].axhline(mIoU.item() , color = "orange" , linestyle = "--" , label = "Validation mIoU")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("mIoU")
axes[1].set_title("Pixel Classifier v2_bdd100k mIoU Curve")
axes[1].grid(True)
axes[1].legend()

figure.tight_layout()
figure.savefig(SCRIPT_DIR / "figure_v2_bdd100k.png" , dpi = 300 , bbox_inches = "tight")
plt.close(figure)
print("\nfigure saved: figure_v2_bdd100k.png")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as tud
from pathlib import Path

from Models.pc_v3 import UNet

SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent / "datasets"

from NetTrainer.pixel_classifier_v2.bdd100k.pc_v2_bdd100k_ds import BDDLaneDataset
from NetTrainer.pixel_classifier_v3.steering_dataset import SteeringDataset , split_steering_dataset

road_train_dataset = BDDLaneDataset(DATASETS_DIR / "processed_pc_ds/train_pairs.csv")
road_val_dataset = BDDLaneDataset(DATASETS_DIR / "processed_pc_ds/val_pairs.csv")
full_lane_train_dataset = BDDLaneDataset(DATASETS_DIR / "processed_pc_lane_ds_bdd100k/train_pairs.csv")
lane_val_dataset = BDDLaneDataset(DATASETS_DIR / "processed_pc_lane_ds_bdd100k/val_pairs.csv")

steering_target = "gameSteer"
full_steering_dataset = SteeringDataset(DATASETS_DIR / "ets2_steering" , target = steering_target)
steering_train_dataset , steering_val_dataset = split_steering_dataset(full_steering_dataset)

lane_train_sample_size = min(30000 , len(full_lane_train_dataset))
lane_sample_generator = torch.Generator().manual_seed(42)
lane_train_indices = torch.randperm(len(full_lane_train_dataset) , generator = lane_sample_generator)[ : lane_train_sample_size].tolist()
lane_train_dataset = tud.Subset(full_lane_train_dataset , lane_train_indices)

batch_size = 16
road_train_loader = tud.DataLoader(road_train_dataset , batch_size = batch_size , shuffle = True)
road_val_loader = tud.DataLoader(road_val_dataset , batch_size = batch_size , shuffle = False)
lane_train_loader = tud.DataLoader(lane_train_dataset , batch_size = batch_size , shuffle = True)
lane_val_loader = tud.DataLoader(lane_val_dataset , batch_size = batch_size , shuffle = False)
steering_train_loader = tud.DataLoader(steering_train_dataset , batch_size = batch_size , shuffle = True)
steering_val_loader = tud.DataLoader(steering_val_dataset , batch_size = batch_size , shuffle = False)

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
steering_criterion = nn.SmoothL1Loss(beta = 0.1)
optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)
lane_prob_threshold = 0.7

print("current device:" , device)
print("road/car training samples:" , len(road_train_dataset))
print("lane training samples:" , len(lane_train_dataset))
print("steering training samples:" , len(steering_train_dataset))
print("steering target:" , steering_target)

steering_split = {
    "train": [str(record.image_path) for record in steering_train_dataset.records] ,
    "val": [str(record.image_path) for record in steering_val_dataset.records]
}
pd.Series(steering_split).to_json(SCRIPT_DIR / "pc_model_v3.split.json" , indent = 2 , force_ascii = False)

def train_mask(loader , task , criterion , classes):
    model.train()
    total_loss = 0
    inter = torch.zeros(classes , dtype = torch.float64 , device = device)
    union = torch.zeros(classes , dtype = torch.float64 , device = device)

    for batch_index , (images , masks) in enumerate(loader):
        images , masks = images.to(device) , masks.to(device)
        optimizer.zero_grad()
        features = model(images)
        outputs = model.head(images , features , task) #[batch , ch , 352 , 640]
        loss = criterion(outputs , masks) if (masks != 255).any() else outputs.sum() * 0
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

    IoU = torch.where(union > 0 , inter / union , torch.tensor(float("nan") , device = device))
    yield total_loss / len(loader.dataset) , IoU , torch.nanmean(IoU)

def evaluate_mask(loader , task , criterion , classes):
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
            loss = criterion(outputs , masks) if (masks != 255).any() else outputs.sum() * 0
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
    return (total_loss / len(loader.dataset) , correct / total if total > 0 else 0 , IoU , torch.nanmean(IoU))

def train_steering(loader):
    model.train()
    total_loss = 0
    total_absolute_error = 0
    total_squared_error = 0

    for batch_index , (images , steering) in enumerate(loader):
        images , steering = images.to(device) , steering.to(device)
        optimizer.zero_grad()
        features = model(images)
        outputs = model.head(images , features , "steering") #[batch , 1]
        loss = steering_criterion(outputs , steering)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

        error = outputs.detach() - steering
        total_absolute_error += error.abs().sum().item()
        total_squared_error += error.square().sum().item()

        if batch_index == 0 or (batch_index + 1) % 100 == 0:
            print("steering batch:" , batch_index + 1 , "/" , len(loader) , "loss:" , loss.item() , flush = True)
        yield #Let road and lane update the shared network between batches.

    MAE = total_absolute_error / len(loader.dataset)
    RMSE = np.sqrt(total_squared_error / len(loader.dataset))
    yield total_loss / len(loader.dataset) , MAE , RMSE

def evaluate_steering(loader):
    model.eval()
    total_loss = 0
    total_absolute_error = 0
    total_squared_error = 0

    with torch.no_grad():
        for images , steering in loader:
            images , steering = images.to(device) , steering.to(device)
            features = model(images)
            outputs = model.head(images , features , "steering")
            loss = steering_criterion(outputs , steering)
            total_loss += loss.item() * images.size(0)

            error = outputs - steering
            total_absolute_error += error.abs().sum().item()
            total_squared_error += error.square().sum().item()

    MAE = total_absolute_error / len(loader.dataset)
    RMSE = np.sqrt(total_squared_error / len(loader.dataset))
    return total_loss / len(loader.dataset) , MAE , RMSE

loss_set = []
miou_set = []
steering_mae_set = []

for epoch in range(30):
    try:
        print("\nepoch:" , epoch + 1 , "training" , flush = True)
        road_training = train_mask(road_train_loader , "road" , road_criterion , 3)
        lane_training = train_mask(lane_train_loader , "lane" , lane_criterion , 2)
        steering_training = train_steering(steering_train_loader)
        road_steps = lane_steps = steering_steps = 0
        road_batches , lane_batches , steering_batches = len(road_train_loader) , len(lane_train_loader) , len(steering_train_loader)
        # Interleave by progress: visit every batch once without a single-task tail.
        while road_steps < road_batches or lane_steps < lane_batches or steering_steps < steering_batches:
            road_progress = road_steps / road_batches if road_steps < road_batches else float("inf")
            lane_progress = lane_steps / lane_batches if lane_steps < lane_batches else float("inf")
            steering_progress = steering_steps / steering_batches if steering_steps < steering_batches else float("inf")
            if road_progress <= lane_progress and road_progress <= steering_progress:
                next(road_training)
                road_steps += 1
            elif lane_progress <= steering_progress:
                next(lane_training)
                lane_steps += 1
            else:
                next(steering_training)
                steering_steps += 1
        road_loss , road_IoU , road_mIoU = next(road_training)
        lane_loss , lane_IoU , lane_mIoU = next(lane_training)
        steering_loss , steering_MAE , steering_RMSE = next(steering_training)
        avg_loss = (road_loss + lane_loss + steering_loss) / 3
        train_mIoU = torch.nanmean(torch.stack((road_mIoU , lane_mIoU)))
        loss_set.append(avg_loss)
        miou_set.append(train_mIoU.item())
        steering_mae_set.append(steering_MAE)

        print("\nepoch:" , epoch + 1 , "loss =" , avg_loss)
        print("train road-head background IoU:" , road_IoU[0].item())
        print("train road IoU:" , road_IoU[1].item())
        print("train car IoU:" , road_IoU[2].item())
        print("train lane-head background IoU:" , lane_IoU[0].item())
        print("train lane IoU:" , lane_IoU[1].item())
        print("train mIoU:" , train_mIoU.item())
        print("train steering loss:" , steering_loss)
        print("train steering MAE:" , steering_MAE)
        print("train steering RMSE:" , steering_RMSE)
        torch.save({
            "model_state_dict": model.state_dict() ,
            "format_version": 3 , "image_size": [640 , 352] ,
            "trained_tasks": ["road" , "lane" , "steering"] ,
            "steering_target": steering_target , "steering_positive": "left"
        } , SCRIPT_DIR / "pc_model_v3.pth")
    except KeyboardInterrupt:
        break

torch.save({
    "model_state_dict": model.state_dict() ,
    "format_version": 3 , "image_size": [640 , 352] ,
    "trained_tasks": ["road" , "lane" , "steering"] ,
    "steering_target": steering_target , "steering_positive": "left"
} , SCRIPT_DIR / "pc_model_v3.pth")
print("\nmodel saved: pc_model_v3.pth")

road_val_loss , road_accuracy , road_IoU , road_mIoU = evaluate_mask(road_val_loader , "road" , road_criterion , 3)
lane_val_loss , lane_accuracy , lane_IoU , lane_mIoU = evaluate_mask(lane_val_loader , "lane" , lane_criterion , 2)
steering_val_loss , steering_val_MAE , steering_val_RMSE = evaluate_steering(steering_val_loader)
avg_val_loss = (road_val_loss + lane_val_loss + steering_val_loss) / 3
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
print("steering validation loss:" , steering_val_loss)
print("steering validation MAE:" , steering_val_MAE)
print("steering validation RMSE:" , steering_val_RMSE)

epochs = range(1 , len(loss_set) + 1)
plt.switch_backend("Agg") #save figures without a GUI window
figure , axes = plt.subplots(1 , 3 , figsize = (16 , 5))

axes[0].plot(epochs , loss_set , color = "blue" , label = "Train Loss")
axes[0].axhline(avg_val_loss , color = "red" , linestyle = "--" , label = "Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Pixel Classifier v3 Loss Curve")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(epochs , miou_set , color = "green" , label = "Train mIoU")
axes[1].axhline(mIoU.item() , color = "orange" , linestyle = "--" , label = "Validation mIoU")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("mIoU")
axes[1].set_title("Pixel Classifier v3 mIoU Curve")
axes[1].grid(True)
axes[1].legend()

axes[2].plot(epochs , steering_mae_set , color = "blue" , label = "Train MAE")
axes[2].axhline(steering_val_MAE , color = "red" , linestyle = "--" , label = "Validation MAE")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("MAE")
axes[2].set_title("Pixel Classifier v3 Steering MAE Curve")
axes[2].grid(True)
axes[2].legend()

figure.tight_layout()
figure.savefig(SCRIPT_DIR / "figure_v3.png" , dpi = 300 , bbox_inches = "tight")
plt.close(figure)
print("\nfigure saved: figure_v3.png")

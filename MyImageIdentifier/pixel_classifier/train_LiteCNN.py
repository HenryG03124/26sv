import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as tud
from pathlib import Path
from MyImageIdentifier.pixel_classifier.bdd_pc_ds import BDDVehicleRoadDataset

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

train_dataset = BDDVehicleRoadDataset(DATASETS_DIR / "processed_pc_ds/train_pairs.csv")
val_dataset = BDDVehicleRoadDataset(DATASETS_DIR / "processed_pc_ds/val_pairs.csv")

train_loader = tud.DataLoader(train_dataset , batch_size = 64 , shuffle = True)
val_loader = tud.DataLoader(val_dataset , batch_size = 64 , shuffle = False)

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
    nn.ConvTranspose2d(32 , 16 , kernel_size = 2 , stride = 2),
    nn.ReLU(),
    nn.ConvTranspose2d(16 , 3 , kernel_size = 2 , stride = 2)
).to(device)

class_weights = torch.tensor([1.0 , 1.5 , 3.0] , dtype = torch.float32 , device = device)
criterion = nn.CrossEntropyLoss(weight = class_weights , ignore_index = 255)
optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)
print("current device:" , device)

loss_set = []
miou_set = []
model.train()

for epoch in range(100):
    try:
        total_loss = 0
        train_inter = torch.zeros(3 , dtype = torch.float64 , device = device)
        train_union = torch.zeros(3 , dtype = torch.float64 , device = device)

        for images , masks in train_loader:
            images , masks = images.to(device) , masks.to(device)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output , masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            predictions = output.argmax(dim = 1)

            valid = masks != 255

            for class_id in range(3):
                predc = (predictions == class_id) & valid
                truec = (masks == class_id) & valid

                train_inter[class_id] += (predc & truec).sum()
                train_union[class_id] += (predc | truec).sum()

        avg_loss = total_loss / len(train_dataset)
        train_IoU = torch.where(train_union > 0 , train_inter / train_union , torch.tensor(float("nan") , device = device))
        train_mIoU = torch.nanmean(train_IoU)

        loss_set.append(avg_loss)
        miou_set.append(train_mIoU.item())
        print("\nepoch:" , epoch + 1 , "loss =" , avg_loss)
        print("train 背景 IoU:" , train_IoU[0].item())
        print("train 道路 IoU:" , train_IoU[1].item())
        print("train 车辆 IoU:" , train_IoU[2].item())
        print("train mIoU:" , train_mIoU.item())

    except KeyboardInterrupt:
        break

torch.save(model.state_dict() , "pc_model_LiteCNN.pth")
print("\nmodel saved: pc_model_LiteCNN.pth")

model.eval()
total = 0
correct = 0
val_loss = 0
inter = torch.zeros(3 , dtype = torch.float64 , device = device)
union = torch.zeros(3 , dtype = torch.float64 , device = device)

with torch.no_grad():
    for images, masks in val_loader:
        images , masks = images.to(device) , masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        val_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim = 1)

        valid = masks != 255

        for class_id in range(3):
            predc = (predictions == class_id) & valid
            truec = (masks == class_id) & valid

            inter[class_id] += (predc & truec).sum()
            union[class_id] += (predc | truec).sum()

        correct += (predictions[valid] == masks[valid]).sum().item()
        total += valid.sum().item()

avg_val_loss = val_loss / len(val_dataset)
accuracy = correct / total
IoU = torch.where(union > 0 , inter / union , torch.tensor(float("nan") , device = device))
mIoU = torch.nanmean(IoU)

print("\nloss:", avg_val_loss)
print("accuracy:", accuracy)
print("validation 背景 IoU:" , IoU[0].item())
print("validation 道路 IoU:" , IoU[1].item())
print("validation 车辆 IoU:" , IoU[2].item())
print("validation mIoU:" , mIoU.item())

epochs = range(1 , len(loss_set) + 1)
figure , axes = plt.subplots(1 , 2 , figsize = (12 , 5))

axes[0].plot(epochs , loss_set , color = "blue" , label = "Train Loss")
axes[0].axhline(avg_val_loss , color = "red" , linestyle = "--" , label = "Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("LiteCNN Loss Curve")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(epochs , miou_set , color = "green" , label = "Train mIoU")
axes[1].axhline(mIoU.item() , color = "orange" , linestyle = "--" , label = "Validation mIoU")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("mIoU")
axes[1].set_title("LiteCNN mIoU Curve")
axes[1].grid(True)
axes[1].legend()

figure.tight_layout()
figure.savefig("figure_LiteCNN.png" , dpi = 300 , bbox_inches = "tight")
plt.close(figure)
print("\nfigure saved: figure_LiteCNN.png")

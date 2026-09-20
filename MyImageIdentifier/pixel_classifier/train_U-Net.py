import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as tud
from pathlib import Path
from MyImageIdentifier.pixel_classifier.bdd_pc_ds import BDDVehicleRoadDataset

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

train_dataset = BDDVehicleRoadDataset(DATASETS_DIR / "processed_pc_ds_bdd100k/train_pairs.csv")
val_dataset = BDDVehicleRoadDataset(DATASETS_DIR / "processed_pc_ds_bdd100k/val_pairs.csv")

train_loader = tud.DataLoader(train_dataset , batch_size = 64 , shuffle = True)
val_loader = tud.DataLoader(val_dataset , batch_size = 64 , shuffle = False)

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
    def __init__(self , classes):
        super().__init__()

        self.root = nn.Sequential(
            nn.Conv2d(classes , 64 , kernel_size = 7 , stride = 2 , padding = 3 , bias = False) , #224 * 128
            nn.BatchNorm2d(64) ,
            nn.ReLU(inplace = True)
        )

        self.pool = nn.MaxPool2d(kernel_size = 3 , stride = 2 , padding = 1) #112 * 64

        self.encoder1 = nn.Sequential(
            EncoderBlock(64 , 64 , stride = 1) ,
            EncoderBlock(64 , 64 , stride = 1)
        )

        self.encoder2 = nn.Sequential(
            EncoderBlock(64 , 128 , stride = 2) , #56 * 32
            EncoderBlock(128 , 128 , stride = 1)
        )

        self.encoder3 = nn.Sequential(
            EncoderBlock(128 , 256 , stride = 2) , #28 * 16
            EncoderBlock(256 , 256 , stride = 1)
        )

        self.encoder4 = nn.Sequential(
            EncoderBlock(256 , 512 , stride = 2) , #14 * 8
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

        self.output = nn.Sequential(
            nn.ConvTranspose2d(32 , 16 , kernel_size = 2 , stride = 2) ,
            nn.ReLU(inplace = True) ,
            nn.Conv2d(16 , classes , kernel_size = 1)
        )

    def forward(self , x):
        root = self.root(x) #224 * 128 ch = 64

        encoder1 = self.pool(root) #112 * 64 ch = 64
        encoder1 = self.encoder1(encoder1) 

        encoder2 = self.encoder2(encoder1) #56 * 32 ch = 128
        encoder3 = self.encoder3(encoder2) #28 * 16 ch = 256
        encoder4 = self.encoder4(encoder3) #14 * 8 ch = 512

        decoder4 = self.up4(encoder4) #28 * 16 ch = 256
        decoder4 = torch.cat((decoder4 , encoder3) , dim = 1) #28 * 16 ch = 512
        #join tensor in channel dimension ([batch , ch , h , w])
        decoder4 = self.decoder4(decoder4) #28 * 16 ch = 256

        decoder3 = self.up3(decoder4) #56 * 32 ch = 128
        decoder3 = torch.cat((decoder3 , encoder2) , dim = 1) #56 * 32 ch = 256
        decoder3 = self.decoder3(decoder3) #56 * 32 ch = 128

        decoder2 = self.up2(decoder3) #112 * 64 ch = 64
        decoder2 = torch.cat((decoder2 , encoder1) , dim = 1) #112 * 64 ch = 128
        decoder2 = self.decoder2(decoder2) #112 * 64 ch = 64

        decoder1 = self.up1(decoder2) #224 * 128 ch = 32
        decoder1 = torch.cat((decoder1 , root) , dim = 1) #224 * 128 ch = 96
        decoder1 = self.decoder1(decoder1) #224 * 128 ch = 32

        output = self.output(decoder1) #224 * 128 ch = classes

        return output

model = UNet(classes = 3).to(device)

class_weights = torch.tensor([1.0 , 1.5 , 3.0] , dtype = torch.float32 , device = device)
criterion = nn.CrossEntropyLoss(weight = class_weights , ignore_index = 255)
optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)
print("current device:" , device)

loss_set = []
miou_set = []
model.train()

for epoch in range(30):
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
        print("train background IoU:" , train_IoU[0].item())
        print("train road IoU:" , train_IoU[1].item())
        print("train vehicle IoU:" , train_IoU[2].item())
        print("train mIoU:" , train_mIoU.item())

    except KeyboardInterrupt:
        break

torch.save(model.state_dict() , "pc_model_U-Net.pth")
print("\nmodel saved: pc_model_U-Net.pth")

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
print("validation background IoU:" , IoU[0].item())
print("validation road IoU:" , IoU[1].item())
print("validation vehicle IoU:" , IoU[2].item())
print("validation mIoU:" , mIoU.item())

epochs = range(1 , len(loss_set) + 1)
figure , axes = plt.subplots(1 , 2 , figsize = (12 , 5))

axes[0].plot(epochs , loss_set , color = "blue" , label = "Train Loss")
axes[0].axhline(avg_val_loss , color = "red" , linestyle = "--" , label = "Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("U-Net Loss Curve")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(epochs , miou_set , color = "green" , label = "Train mIoU")
axes[1].axhline(mIoU.item() , color = "orange" , linestyle = "--" , label = "Validation mIoU")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("mIoU")
axes[1].set_title("U-Net mIoU Curve")
axes[1].grid(True)
axes[1].legend()

figure.tight_layout()
figure.savefig("figure_U-Net.png" , dpi = 300 , bbox_inches = "tight")
plt.close(figure)
print("\nfigure saved: figure_U-Net.png")

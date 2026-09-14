import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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

model = nn.Sequential(
    ResNet18_encoder ,
    detector
).to(device)

def calculate_iou(box , boxes):
    intersection_x1 = torch.maximum(box[0] , boxes[: , 0])
    intersection_y1 = torch.maximum(box[1] , boxes[: , 1])
    intersection_x2 = torch.minimum(box[2] , boxes[: , 2])
    intersection_y2 = torch.minimum(box[3] , boxes[: , 3])
    intersection = (intersection_x2 - intersection_x1).clamp(min = 0) * (intersection_y2 - intersection_y1).clamp(min = 0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[: , 2] - boxes[: , 0]) * (boxes[: , 3] - boxes[: , 1])
    return intersection / (box_area + boxes_area - intersection).clamp(min = 1e-7)

def non_maximum_suppression(boxes , scores , labels , threshold):
    kept = []

    for class_id in labels.unique():
        indices = (labels == class_id).nonzero(as_tuple = False).squeeze(1)
        order = indices[torch.argsort(scores[indices] , descending = True)]

        while len(order) > 0:
            current = order[0]
            kept.append(current.item())

            if len(order) == 1:
                break

            remaining = order[1 : ]
            iou = calculate_iou(boxes[current] , boxes[remaining])
            order = remaining[iou <= threshold]

    return torch.tensor(kept , dtype = torch.int64)

class_names = ("pedestrian" , "car")
colors = ((255 , 64 , 64) , (64 , 128 , 255))
grid_height = 16
grid_width = 28
confidence_threshold = 0.5
nms_threshold = 0.35

model.load_state_dict(torch.load("od_model_Lite.pth" , map_location = device , weights_only = True))
model.eval()

image = Image.open("test.jpg").convert("RGB")
original_width , original_height = image.size
resized_image = image.resize((448 , 256) , Image.Resampling.BILINEAR)
image_array = np.array(resized_image , dtype = np.uint8 , copy = True)
image_tensor = torch.from_numpy(image_array).permute(2 , 0 , 1).float().div(255.0)
image_tensor = image_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)[0].cpu()

    box_output = output[ : 4]
    obj = torch.sigmoid(output[4])
    probabilities = torch.softmax(output[5 : ] , dim = 0)

    class_scores , labels = probabilities.max(dim = 0)
    scores = obj * class_scores
    selected = scores >= confidence_threshold
    positions = selected.nonzero(as_tuple = False)

boxes = []
selected_scores = []
selected_labels = []

for position in positions:
    y = position[0].item()
    x = position[1].item()
    offset_x = torch.sigmoid(box_output[0 , y , x]).item()
    offset_y = torch.sigmoid(box_output[1 , y , x]).item()
    width_in_cells = torch.exp(box_output[2 , y , x].clamp(-4.0 , 4.0)).item()
    height_in_cells = torch.exp(box_output[3 , y , x].clamp(-4.0 , 4.0)).item()
    center_x = (x + offset_x) / grid_width * original_width
    center_y = (y + offset_y) / grid_height * original_height
    box_width = width_in_cells / grid_width * original_width
    box_height = height_in_cells / grid_height * original_height
    first_x = max(0.0 , center_x - box_width / 2)
    first_y = max(0.0 , center_y - box_height / 2)
    second_x = min(float(original_width) , center_x + box_width / 2)
    second_y = min(float(original_height) , center_y + box_height / 2)
    boxes.append([
        first_x , first_y , second_x , second_y
    ])
    selected_scores.append(scores[y , x].item())
    selected_labels.append(labels[y , x].item())

if boxes:
    boxes = torch.tensor(boxes , dtype = torch.float32)
    selected_scores = torch.tensor(selected_scores , dtype = torch.float32)
    selected_labels = torch.tensor(selected_labels , dtype = torch.int64)
    kept = non_maximum_suppression(boxes , selected_scores , selected_labels , nms_threshold)
    boxes = boxes[kept]
    selected_scores = selected_scores[kept]
    selected_labels = selected_labels[kept]

draw = ImageDraw.Draw(image)

if isinstance(boxes , torch.Tensor):
    for box , score , label in zip(boxes , selected_scores , selected_labels):
        class_id = label.item()
        color = colors[class_id]
        coordinates = tuple(box.tolist())
        text = f"{class_names[class_id]} {score.item():.2f}"
        draw.rectangle(coordinates , outline = color , width = 3)
        text_box = draw.textbbox((coordinates[0] , coordinates[1]) , text)
        draw.rectangle(text_box , fill = color)
        draw.text((coordinates[0] , coordinates[1]) , text , fill = (0 , 0 , 0))

image.save("test_result_Lite.png")
print("result saved: test_result_Lite.png")

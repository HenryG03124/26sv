import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as tud
from pathlib import Path
from bdd_objdetect_ds import BDDDetectionDataset, detection_collate_fn

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

full_train_dataset = BDDDetectionDataset(DATASETS_DIR / "processed_objdetect_ds/train_pairs.csv")
val_dataset = BDDDetectionDataset(DATASETS_DIR / "processed_objdetect_ds/val_pairs.csv")

train_sample_size = min(30000 , len(full_train_dataset))
sample_generator = torch.Generator().manual_seed(42)
train_indices = torch.randperm(len(full_train_dataset) , generator = sample_generator)[ : train_sample_size].tolist()
train_dataset = tud.Subset(full_train_dataset , train_indices)

train_loader = tud.DataLoader(train_dataset , batch_size = 64 , shuffle = True , collate_fn = detection_collate_fn)
val_loader = tud.DataLoader(val_dataset , batch_size = 64 , shuffle = False , collate_fn = detection_collate_fn)

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
    #256*448
    nn.Conv2d(3 , 64 , kernel_size = 7 , stride = 2 , padding = 3 , bias = False) , #128*224
    nn.BatchNorm2d(64) ,
    nn.ReLU(inplace = True) ,
    nn.MaxPool2d(kernel_size = 3 , stride = 2 , padding = 1) , #64*112
    BasicBlock(64 , 64 , stride = 1) ,
    BasicBlock(64 , 64 , stride = 1) ,

    BasicBlock(64 , 128 , stride = 2) , #32*56
    BasicBlock(128 , 128 , stride = 1) ,

    BasicBlock(128 , 256 , stride = 2) , #16*28
    BasicBlock(256 , 256 , stride = 1) ,

    BasicBlock(256 , 512 , stride = 1) ,
    BasicBlock(512 , 512 , stride = 1) ,
)

detector = nn.Sequential(
    nn.Conv2d(512 , 256 , kernel_size = 3 , padding = 1 , bias = False) ,
    nn.BatchNorm2d(256) ,
    nn.ReLU(inplace = True) ,

    nn.Conv2d(256 , 256 , kernel_size = 3 , padding = 1 , bias = False) ,
    nn.BatchNorm2d(256) ,
    nn.ReLU(inplace = True) ,

    nn.Conv2d(256 , 7 , kernel_size = 1 , stride = 1) #boxx , boxy , boxw , boxh , obj , person , car
)

model = nn.Sequential(
    ResNet18_encoder ,
    detector
).to(device)

class_names = ("pedestrian" , "car")
input_height = 256
input_width = 448
grid_height = 16
grid_width = 28
number_of_classes = 2
box_loss_weight = 5.0

# BDDDetectionDataset IDs: person = 1, car = 3.
pedestrian_label_id = 1
car_label_id = 3
class_weights = torch.tensor([1.0 , 0.30] , dtype = torch.float32 , device = device) #person , car
class_criterion = nn.CrossEntropyLoss(weight = class_weights)
obj_criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([10.0], device = device))
box_criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)
print("current device:" , device)
print("training samples:" , len(train_dataset))

def build_targets(targets):
    batch_size = len(targets)
    class_targets = torch.full((batch_size , grid_height , grid_width) , -1 , dtype = torch.int64)
    box_targets = torch.zeros((batch_size , 4 , grid_height , grid_width) , dtype = torch.float32)

    for image_id , target in enumerate(targets):
        boxes = target["boxes"]
        labels = target["labels"]
        selected = (labels == pedestrian_label_id) | (labels == car_label_id)
        boxes = boxes[selected]
        labels = labels[selected]

        if len(boxes) == 0:
            continue

        centers_x = (boxes[: , 0] + boxes[: , 2]) / 2
        centers_y = (boxes[: , 1] + boxes[: , 3]) / 2
        widths = boxes[: , 2] - boxes[: , 0]
        heights = boxes[: , 3] - boxes[: , 1]
        grid_x = centers_x / input_width * grid_width
        grid_y = centers_y / input_height * grid_height
        cell_x = grid_x.floor().long().clamp(0 , grid_width - 1)
        cell_y = grid_y.floor().long().clamp(0 , grid_height - 1)
        areas = widths * heights
        order = torch.argsort(areas)

        for object_id in order.tolist():
            x = cell_x[object_id].item()
            y = cell_y[object_id].item()
            class_targets[image_id , y , x] = (labels[object_id] == car_label_id).long()
            box_targets[image_id , 0 , y , x] = grid_x[object_id] - cell_x[object_id].float()
            box_targets[image_id , 1 , y , x] = grid_y[object_id] - cell_y[object_id].float()
            box_targets[image_id , 2 , y , x] = torch.log((widths[object_id] / input_width * grid_width).clamp(min = 1e-6))
            box_targets[image_id , 3 , y , x] = torch.log((heights[object_id] / input_height * grid_height).clamp(min = 1e-6))

    return class_targets.to(device) , box_targets.to(device)

def calculate_loss(outputs , targets):
    class_targets , box_targets = build_targets(targets)
    obj_targets = (class_targets >= 0).float()
    obj_loss = obj_criterion(outputs[: , 4] , obj_targets)
    positive = class_targets >= 0

    if positive.any():
        positive_class_predictions = outputs[: , 5 :].permute(0 , 2 , 3 , 1)[positive] #[B, 2, 16, 28] to [B, 16, 28, 2]
        positive_class_targets = class_targets[positive]
        class_loss = class_criterion(positive_class_predictions , positive_class_targets)

        box_predictions = torch.cat((torch.sigmoid(outputs[: , : 2]) , outputs[: , 2 : 4]) , dim = 1).permute(0 , 2 , 3 , 1)
        positive_box_predictions = box_predictions[positive] #[B, 4, 16, 28] to [B, 16, 28, 4]
        positive_box_targets = box_targets.permute(0 , 2 , 3 , 1)[positive]
        box_loss = box_criterion(positive_box_predictions , positive_box_targets)
    else:
        class_loss = outputs[: , 5 :].sum() * 0
        box_loss = outputs[: , : 4].sum() * 0

    loss = class_loss + obj_loss + box_loss_weight * box_loss
    return loss , class_targets , box_targets

def decode_relative_boxes(encoded_boxes , prediction):
    batch_size = encoded_boxes.size(0)
    grid_y , grid_x = torch.meshgrid(
        torch.arange(grid_height , device = encoded_boxes.device , dtype = encoded_boxes.dtype) ,
        torch.arange(grid_width , device = encoded_boxes.device , dtype = encoded_boxes.dtype) ,
        indexing = "ij"
    )

    if prediction:
        offset_x = torch.sigmoid(encoded_boxes[: , 0])
        offset_y = torch.sigmoid(encoded_boxes[: , 1])
        width_in_cells = torch.exp(encoded_boxes[: , 2].clamp(-4.0 , 4.0))
        height_in_cells = torch.exp(encoded_boxes[: , 3].clamp(-4.0 , 4.0))
    else:
        offset_x = encoded_boxes[: , 0]
        offset_y = encoded_boxes[: , 1]
        width_in_cells = torch.exp(encoded_boxes[: , 2])
        height_in_cells = torch.exp(encoded_boxes[: , 3])

    center_x = (grid_x.unsqueeze(0).expand(batch_size , -1 , -1) + offset_x) / grid_width
    center_y = (grid_y.unsqueeze(0).expand(batch_size , -1 , -1) + offset_y) / grid_height
    width = width_in_cells / grid_width
    height = height_in_cells / grid_height

    return torch.stack((
        center_x - width / 2 ,
        center_y - height / 2 ,
        center_x + width / 2 ,
        center_y + height / 2
    ) , dim = 1)

def calculate_iou(outputs , class_targets , box_targets):
    positive_positions = (class_targets >= 0).nonzero(as_tuple = False)

    if len(positive_positions) == 0:
        return torch.empty(0 , device = device) , torch.empty(0 , dtype = torch.int64 , device = device)

    box_predictions = decode_relative_boxes(outputs[: , : 4] , prediction = True).permute(0 , 2 , 3 , 1)
    decoded_box_targets = decode_relative_boxes(box_targets , prediction = False).permute(0 , 2 , 3 , 1)
    batch_ids = positive_positions[:, 0]
    predicted = box_predictions[batch_ids , positive_positions[:, 1] , positive_positions[: , 2]]
    expected = decoded_box_targets[batch_ids , positive_positions[:, 1] , positive_positions[ :, 2]]

    predicted_boxes = predicted.clamp(0 , 1)
    expected_boxes = expected.clamp(0 , 1)

    intersection_x1 = torch.maximum(predicted_boxes[: , 0] , expected_boxes[: , 0])
    intersection_y1 = torch.maximum(predicted_boxes[: , 1] , expected_boxes[: , 1])
    intersection_x2 = torch.minimum(predicted_boxes[: , 2] , expected_boxes[: , 2])
    intersection_y2 = torch.minimum(predicted_boxes[: , 3] , expected_boxes[: , 3])
    intersection = (intersection_x2 - intersection_x1).clamp(min = 0) * (intersection_y2 - intersection_y1).clamp(min = 0)
    predicted_area = (predicted_boxes[: , 2] - predicted_boxes[: , 0]) * (predicted_boxes[: , 3] - predicted_boxes[: , 1])
    expected_area = (expected_boxes[: , 2] - expected_boxes[: , 0]) * (expected_boxes[: , 3] - expected_boxes[: , 1])
    union = predicted_area + expected_area - intersection
    iou = intersection / union.clamp(min = 1e-7)
    labels = class_targets[batch_ids , positive_positions[: , 1] , positive_positions[: , 2]]
    return iou , labels

def calculate_box_iou(box , boxes): #calcute the iou between one box and the rest boxes
    intersection_x1 = torch.maximum(box[0] , boxes[: , 0])
    intersection_y1 = torch.maximum(box[1] , boxes[: , 1])
    intersection_x2 = torch.minimum(box[2] , boxes[: , 2])
    intersection_y2 = torch.minimum(box[3] , boxes[: , 3])
    intersection = (intersection_x2 - intersection_x1).clamp(min = 0) * (intersection_y2 - intersection_y1).clamp(min = 0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[: , 2] - boxes[: , 0]) * (boxes[: , 3] - boxes[: , 1])
    return intersection / (box_area + boxes_area - intersection).clamp(min = 1e-7)

def non_maximum_suppression(boxes , scores , nms_threshold):
    kept = []
    order = torch.argsort(scores , descending = True)

    while len(order) > 0:
        current = order[0]
        kept.append(current.item())

        if len(order) == 1:
            break

        remaining = order[1 : ] #throw order[1]
        iou = calculate_box_iou(boxes[current] , boxes[remaining])
        order = remaining[iou <= nms_threshold]

    return torch.tensor(kept , dtype = torch.int64 , device = boxes.device)

def decode_predictions(outputs , score_threshold , nms_threshold):
    boxes = decode_relative_boxes(outputs[: , :4] , prediction = True).permute(0 , 2 , 3 , 1).clamp(0 , 1)
    objectness = torch.sigmoid(outputs[: , 4])
    class_probabilities = torch.softmax(outputs[: , 5:] , dim = 1)
    predictions = []

    for image_id in range(outputs.size(0)):
        image_boxes = []
        image_scores = []
        image_labels = []

        for class_id in range(number_of_classes):
            scores = objectness[image_id] * class_probabilities[image_id , class_id]
            selected = scores >= score_threshold

            if not selected.any():
                continue

            selected_boxes = boxes[image_id][selected]
            selected_scores = scores[selected]
            kept = non_maximum_suppression(selected_boxes , selected_scores , nms_threshold)
            image_boxes.append(selected_boxes[kept])
            image_scores.append(selected_scores[kept])
            image_labels.append(torch.full((len(kept) ,) , class_id , dtype = torch.int64 , device = outputs.device))

        if image_boxes:
            image_boxes = torch.cat(image_boxes)
            image_scores = torch.cat(image_scores)
            image_labels = torch.cat(image_labels)
            order = torch.argsort(image_scores , descending = True)[: 300]
            predictions.append((image_boxes[order].cpu() , image_scores[order].cpu() , image_labels[order].cpu()))
        else:
            predictions.append((torch.empty((0 , 4)) , torch.empty(0) , torch.empty(0 , dtype = torch.int64)))

    return predictions

def calculate_detection_metrics(detections , ground_truth_boxes , score_threshold , iou_threshold):
    ground_truth_count = sum(len(boxes) for boxes in ground_truth_boxes.values())

    if ground_truth_count == 0:
        return float("nan") , float("nan") , float("nan")

    detections.sort(key = lambda detection: detection[0] , reverse = True)
    matched = {
        image_id: torch.zeros(len(boxes) , dtype = torch.bool)
        for image_id , boxes in ground_truth_boxes.items()
    }
    detection_scores = torch.zeros(len(detections) , dtype = torch.float64)
    true_positive = torch.zeros(len(detections) , dtype = torch.float64)
    false_positive = torch.zeros(len(detections) , dtype = torch.float64)

    for detection_id , (score , image_id , box) in enumerate(detections):
        detection_scores[detection_id] = score
        expected_boxes = ground_truth_boxes.get(image_id)

        if expected_boxes is None:
            false_positive[detection_id] = 1
            continue

        iou = calculate_box_iou(box , expected_boxes)
        maximum_iou , expected_id = iou.max(dim = 0)

        if maximum_iou >= iou_threshold and not matched[image_id][expected_id]:
            true_positive[detection_id] = 1
            matched[image_id][expected_id] = True
        else:
            false_positive[detection_id] = 1

    accumulated_true_positive = torch.cumsum(true_positive , dim = 0)
    accumulated_false_positive = torch.cumsum(false_positive , dim = 0)
    recall_curve = accumulated_true_positive / ground_truth_count
    precision_curve = accumulated_true_positive / (accumulated_true_positive + accumulated_false_positive).clamp(min = 1e-7)
    recall_curve = torch.cat((torch.tensor([0.0]) , recall_curve , torch.tensor([1.0])))
    precision_curve = torch.cat((torch.tensor([0.0]) , precision_curve , torch.tensor([0.0])))

    for point_id in range(len(precision_curve) - 2 , -1 , -1):
        precision_curve[point_id] = torch.maximum(precision_curve[point_id] , precision_curve[point_id + 1])

    changed = torch.where(recall_curve[1 : ] != recall_curve[ : -1])[0]
    average_precision = torch.sum((recall_curve[changed + 1] - recall_curve[changed]) * precision_curve[changed + 1]).item()
    selected = detection_scores >= score_threshold
    selected_true_positive = true_positive[selected].sum().item()
    selected_false_positive = false_positive[selected].sum().item()
    precision = selected_true_positive / max(selected_true_positive + selected_false_positive , 1)
    recall = selected_true_positive / ground_truth_count
    return average_precision , precision , recall

loss_set = []
miou_set = []
model.train()

for epoch in range(30):
    try:
        total_loss = 0
        train_iou_sum = torch.zeros(number_of_classes , dtype = torch.float64 , device = device)
        train_iou_count = torch.zeros(number_of_classes , dtype = torch.float64 , device = device)

        for images , targets in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss , class_targets , box_targets = calculate_loss(outputs , targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            with torch.no_grad():
                iou , labels = calculate_iou(outputs , class_targets , box_targets)
                for class_id in range(number_of_classes):
                    selected = labels == class_id
                    train_iou_sum[class_id] += iou[selected].sum().double()
                    train_iou_count[class_id] += selected.sum()

        avg_loss = total_loss / len(train_dataset)
        train_iou = torch.where(train_iou_count > 0 , train_iou_sum / train_iou_count , torch.tensor(float("nan") , device = device))
        train_miou = torch.nanmean(train_iou)

        loss_set.append(avg_loss)
        miou_set.append(train_miou.item())
        print("\nepoch:" , epoch + 1 , "loss =" , avg_loss)
        for class_id in range(number_of_classes):
            print("train" , class_names[class_id] , "IoU:" , train_iou[class_id].item())
        print("train mIoU:" , train_miou.item())

    except KeyboardInterrupt:
        break

torch.save(model.state_dict() , "od_model_Lite.pth")
print("\nmodel saved: od_model_Lite.pth")

model.eval()
val_loss = 0
val_iou_sum = torch.zeros(number_of_classes , dtype = torch.float64 , device = device)
val_iou_count = torch.zeros(number_of_classes , dtype = torch.float64 , device = device)
detections = [[] for _ in range(number_of_classes)]
ground_truth_boxes = [{} for _ in range(number_of_classes)]
box_scale = torch.tensor([input_width , input_height , input_width , input_height] , dtype = torch.float32)
image_number = 0
score_threshold = 0.05
nms_threshold = 0.35

with torch.no_grad():
    for images , targets in val_loader:
        images = images.to(device)
        outputs = model(images)
        loss , class_targets , box_targets = calculate_loss(outputs , targets)
        val_loss += loss.item() * images.size(0)
        iou , labels = calculate_iou(outputs , class_targets , box_targets)

        for class_id in range(number_of_classes):
            selected = labels == class_id
            val_iou_sum[class_id] += iou[selected].sum().double()
            val_iou_count[class_id] += selected.sum()

        predictions = decode_predictions(outputs , score_threshold , nms_threshold)

        for batch_image_id , (target , prediction) in enumerate(zip(targets , predictions)):
            image_id = image_number + batch_image_id
            target_boxes = (target["boxes"] / box_scale).clamp(0 , 1)
            target_labels = target["labels"]
            predicted_boxes , predicted_scores , predicted_labels = prediction

            for class_id , label_id in enumerate((pedestrian_label_id , car_label_id)):
                selected = target_labels == label_id

                if selected.any():
                    ground_truth_boxes[class_id][image_id] = target_boxes[selected]

                selected = predicted_labels == class_id

                for box , score in zip(predicted_boxes[selected] , predicted_scores[selected]):
                    detections[class_id].append((score.item() , image_id , box))

        image_number += len(targets)

avg_val_loss = val_loss / len(val_dataset)
val_iou = torch.where(
    val_iou_count > 0 , val_iou_sum / val_iou_count ,
    torch.tensor(float("nan") , device = device)
)
val_miou = torch.nanmean(val_iou)

print("\nvalidation loss:" , avg_val_loss)
for class_id in range(number_of_classes):
    print("validation" , class_names[class_id] , "localization IoU:" , val_iou[class_id].item())
print("validation localization mIoU:" , val_miou.item())

average_precisions = []

for class_id in range(number_of_classes):
    average_precision , precision , recall = calculate_detection_metrics(detections[class_id] , ground_truth_boxes[class_id] , score_threshold = 0.5 , iou_threshold = 0.5)
    average_precisions.append(average_precision)
    print("validation" , class_names[class_id] , "AP50:" , average_precision)
    print("validation" , class_names[class_id] , "precision@0.5:" , precision)
    print("validation" , class_names[class_id] , "recall@0.5:" , recall)

mean_average_precision = sum(average_precisions) / len(average_precisions)
print("validation mAP50:" , mean_average_precision)

epochs = range(1 , len(loss_set) + 1)
figure , axes = plt.subplots(1 , 2 , figsize = (12 , 5))

axes[0].plot(epochs , loss_set , color = "blue" , label = "Train Loss")
axes[0].axhline(avg_val_loss , color = "red" , linestyle = "--" , label = "Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("ResNet Lite Detection Loss Curve")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(epochs , miou_set , color = "green" , label = "Train mIoU")
axes[1].axhline(val_miou.item() , color = "orange" , linestyle = "--" , label = "Validation mIoU")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("mIoU")
axes[1].set_title("ResNet Lite Detection mIoU Curve")
axes[1].grid(True)
axes[1].legend()

figure.tight_layout()
figure.savefig("figure_Lite.png" , dpi = 300 , bbox_inches = "tight")
plt.close(figure)
print("\nfigure saved: figure_Lite.png")

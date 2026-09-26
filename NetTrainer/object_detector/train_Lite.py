import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as tud
from torchvision.ops import complete_box_iou_loss , nms , sigmoid_focal_loss
from pathlib import Path
from Models.od_Lite import ResNet

from NetTrainer.object_detector.bdd_objdetect_ds import BDDDetectionDataset, detection_collate_fn

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"

full_train_dataset = BDDDetectionDataset(DATASETS_DIR / "processed_objdetect_ds/train_pairs.csv")
val_dataset = BDDDetectionDataset(DATASETS_DIR / "processed_objdetect_ds/val_pairs.csv")

train_sample_size = min(50000 , len(full_train_dataset))
sample_generator = torch.Generator().manual_seed(42)
train_indices = torch.randperm(len(full_train_dataset) , generator = sample_generator)[ : train_sample_size].tolist()
train_dataset = tud.Subset(full_train_dataset , train_indices)

train_loader = tud.DataLoader(train_dataset , batch_size = 64 , shuffle = True , collate_fn = detection_collate_fn)
val_loader = tud.DataLoader(val_dataset , batch_size = 64 , shuffle = False , collate_fn = detection_collate_fn)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model = ResNet().to(device)

class_names = ("pedestrian" , "car" , "truck")
input_height = 352
input_width = 640
grid_height = 22
grid_width = 40
number_of_classes = 3
box_loss_weight = 5.0

pedestrian_label_id = 1
car_label_id = 3
truck_label_id = 4
class_weights = torch.tensor([1.0 , 0.30 , 1.0] , dtype = torch.float32 , device = device) #person , car , truck
class_criterion = nn.CrossEntropyLoss(weight = class_weights , label_smoothing = 0.05)
optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)
print("current device:" , device)
print("training samples:" , len(train_dataset))

def build_targets(targets):
    batch_size = len(targets)
    class_targets = torch.full((batch_size , grid_height , grid_width) , -1 , dtype = torch.int64) #[B , 22 , 40]
    box_targets = torch.zeros((batch_size , 4 , grid_height , grid_width) , dtype = torch.float32) #[B , 4 , 22 , 40]
    label_ids = (pedestrian_label_id , car_label_id , truck_label_id)
    scale = torch.tensor([grid_width / input_width , grid_height / input_height] , dtype = torch.float32)

    for image_id , target in enumerate(targets):
        labels = target["labels"]
        selected = (labels == pedestrian_label_id) | (labels == car_label_id) | (labels == truck_label_id)
        boxes = target["boxes"][selected] * scale.repeat(2)
        labels = labels[selected]
        centers = (boxes[: , :2] + boxes[: , 2:]) / 2
        sizes = boxes[: , 2:] - boxes[: , :2]
        cells = centers.floor().long()
        cells[: , 0].clamp_(0 , grid_width - 1)
        cells[: , 1].clamp_(0 , grid_height - 1)

        # Keep the largest object when centers share a cell.
        for object_id in torch.argsort(sizes.prod(dim = 1)).tolist():
            x , y = cells[object_id].tolist()
            class_targets[image_id , y , x] = label_ids.index(labels[object_id].item())
            box_targets[image_id , :2 , y , x] = centers[object_id] - cells[object_id].float() #x diff , y diff (in grid)
            box_targets[image_id , 2: , y , x] = torch.log(sizes[object_id].clamp(min = 1e-6)) #log(w) , log(h) (grid)

    return class_targets.to(device) , box_targets.to(device)

def calculate_loss(outputs , targets):
    class_targets , box_targets = build_targets(targets)
    positive = class_targets >= 0
    # RetinaNet focal loss , normalized by foreground cells.
    obj_loss = sigmoid_focal_loss(outputs[: , 4] , positive.float() , alpha = 0.25 , gamma = 2.0 , reduction = "none")
    obj_loss = (obj_loss.sum(dim = (1 , 2)) / positive.sum(dim = (1 , 2)).clamp(min = 1)).mean()
    class_loss = box_loss = outputs.sum() * 0

    if positive.any():
        class_predictions = outputs[: , 5:].permute(0 , 2 , 3 , 1)[positive]
        class_loss = class_criterion(class_predictions , class_targets[positive])
        predicted = decode_relative_boxes(outputs[: , :4] , prediction = True).permute(0 , 2 , 3 , 1)[positive]
        expected = decode_relative_boxes(box_targets , prediction = False).permute(0 , 2 , 3 , 1)[positive]
        # CIoU uses pixel geometry without clipping the predicted boxes.
        scale = outputs.new_tensor([input_width , input_height , input_width , input_height])
        box_loss = complete_box_iou_loss(predicted * scale , expected * scale , reduction = "mean")

    loss = class_loss + obj_loss + box_loss_weight * box_loss
    return loss , class_targets , box_targets

def decode_relative_boxes(encoded_boxes , prediction):
    grid_y , grid_x = torch.meshgrid(
        torch.arange(grid_height , device = encoded_boxes.device , dtype = encoded_boxes.dtype) ,
        torch.arange(grid_width , device = encoded_boxes.device , dtype = encoded_boxes.dtype) ,
        indexing = "ij"
    )

    grid = torch.stack((grid_x , grid_y) , dim = 0)
    scale = encoded_boxes.new_tensor([grid_width , grid_height]).view(1 , 2 , 1 , 1)
    offsets = torch.sigmoid(encoded_boxes[: , :2]) if prediction else encoded_boxes[: , :2]
    sizes = encoded_boxes[: , 2:].clamp(-4.0 , 4.0) if prediction else encoded_boxes[: , 2:]
    centers = (grid + offsets) / scale
    half_sizes = torch.exp(sizes) / scale / 2
    return torch.cat((centers - half_sizes , centers + half_sizes) , dim = 1)

def calculate_iou(outputs , class_targets , box_targets):
    positive = class_targets >= 0
    box_predictions = decode_relative_boxes(outputs[: , : 4] , prediction = True).permute(0 , 2 , 3 , 1)
    decoded_box_targets = decode_relative_boxes(box_targets , prediction = False).permute(0 , 2 , 3 , 1)
    predicted = box_predictions[positive].clamp(0 , 1)
    expected = decoded_box_targets[positive].clamp(0 , 1)
    return calculate_box_iou(predicted , expected) , class_targets[positive]

def calculate_box_iou(box , boxes): #Aligned boxes or one box against many.
    first = torch.maximum(box[... , : 2] , boxes[... , : 2])
    second = torch.minimum(box[..., 2 :] , boxes[... , 2 :])
    inter = (second - first).clamp(min = 0).prod(dim = -1)
    area = (box[..., 2 :] - box[..., :2]).prod(dim = -1)
    areas = (boxes[... , 2 :] - boxes[... , : 2]).prod(dim = -1)
    union = area + areas - inter
    return inter / union.clamp(min = 1e-7)

def non_maximum_suppression(boxes , scores , nms_threshold):
    if boxes.device.type == "mps":
        return nms(boxes.cpu() , scores.cpu() , nms_threshold).to(boxes.device)
    return nms(boxes , scores , nms_threshold)

def decode_predictions(outputs , score_threshold , nms_threshold):
    boxes = decode_relative_boxes(outputs[: , : 4] , prediction = True).permute(0 , 2 , 3 , 1).clamp(0 , 1)
    objectness = torch.sigmoid(outputs[: , 4])
    class_probabilities = torch.softmax(outputs[: , 5 :] , dim = 1)
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
    matched = {image_id: torch.zeros(len(boxes) , dtype = torch.bool) for image_id , boxes in ground_truth_boxes.items()}
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

            for class_id , label_id in enumerate((pedestrian_label_id , car_label_id , truck_label_id)):
                selected = target_labels == label_id

                if selected.any():
                    ground_truth_boxes[class_id][image_id] = target_boxes[selected]

                selected = predicted_labels == class_id

                for box , score in zip(predicted_boxes[selected] , predicted_scores[selected]):
                    detections[class_id].append((score.item() , image_id , box))

        image_number += len(targets)

avg_val_loss = val_loss / len(val_dataset)
val_iou = torch.where(val_iou_count > 0 , val_iou_sum / val_iou_count ,torch.tensor(float("nan") , device = device))
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

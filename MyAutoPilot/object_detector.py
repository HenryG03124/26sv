import numpy as np
import cv2
import torch

from Models.object_detector_Lite import ResNet

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

grid_height = 16
grid_width = 28
confidence_threshold = 0.5
nms_threshold = 0.35
car_mask_class_id = 2
car_label_id = 1

class ObjectDetectorLite():
    class_names = ("pedestrian" , "car")

    def __init__(self):
        self.model = ResNet().to(device)
        self.labels = None
        self.filtered_labels = torch.empty(0 , dtype = torch.int64)

    def load_model(self , model_path):
        self.model.load_state_dict(torch.load(model_path , map_location = device , weights_only = True))
        self.model.eval()

    def preprocess(self , frame):
        model_input = cv2.resize(frame , (448 , 256) , interpolation = cv2.INTER_LINEAR)
        model_input = cv2.cvtColor(model_input , cv2.COLOR_BGR2RGB)
        model_input = torch.from_numpy(model_input).permute(2 , 0 , 1).contiguous().to(dtype = torch.float32).div(255.0).unsqueeze(0).to(device)
        return model_input

    def predict(self , frame):
        self.labels = None
        self.filtered_labels = torch.empty(0 , dtype = torch.int64)
        model_input = self.preprocess(frame)

        with torch.no_grad():
            output = self.model(model_input)[0].cpu()

            box_output = output[ : 4]
            obj = torch.sigmoid(output[4])
            probabilities = torch.softmax(output[5 : ] , dim = 0)

            class_scores , self.labels = probabilities.max(dim = 0)
            scores = obj * class_scores
            selected = scores >= confidence_threshold
            positions = selected.nonzero(as_tuple = False)

        return positions , box_output , scores

    def non_maximum_suppression(self , boxes , scores , threshold):
        kept = []

        for class_id in self.filtered_labels.unique():
            indices = (self.filtered_labels == class_id).nonzero(as_tuple = False).squeeze(1)
            order = indices[torch.argsort(scores[indices] , descending = True)]

            while len(order) > 0:
                current = order[0]
                kept.append(current.item())

                if len(order) == 1:
                    break

                remaining = order[1 : ]
                iou = self.calculate_iou(boxes[current] , boxes[remaining])
                order = remaining[iou <= threshold]

        return torch.tensor(kept , dtype = torch.int64)

    def calculate_iou(self , box , boxes):
        intersection_x1 = torch.maximum(box[0] , boxes[: , 0])
        intersection_y1 = torch.maximum(box[1] , boxes[: , 1])
        intersection_x2 = torch.minimum(box[2] , boxes[: , 2])
        intersection_y2 = torch.minimum(box[3] , boxes[: , 3])
        intersection = (intersection_x2 - intersection_x1).clamp(min = 0) * (intersection_y2 - intersection_y1).clamp(min = 0)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[: , 2] - boxes[: , 0]) * (boxes[: , 3] - boxes[: , 1])
        return intersection / (box_area + boxes_area - intersection).clamp(min = 1e-7)

    def box_filter(self , positions , box_output , scores , original_width , original_height , mask , car_mask_ratio_threshold):
        if not 0.0 <= car_mask_ratio_threshold <= 1.0:
            raise ValueError("car_mask_ratio_threshold must be between 0.0 and 1.0")
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D array")
        if self.labels is None:
            raise RuntimeError("predict must be called before box_filter")

        boxes = []
        selected_scores = []
        selected_labels = []

        mask_height , mask_width = mask.shape

        for position in positions:
            y = position[0].item()
            x = position[1].item()
            label = self.labels[y , x].item()

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

            mask_first_x = max(0 , min(mask_width , int(first_x / original_width * mask_width)))
            mask_first_y = max(0 , min(mask_height , int(first_y / original_height * mask_height)))
            mask_second_x = max(0 , min(mask_width , int(second_x / original_width * mask_width)))
            mask_second_y = max(0 , min(mask_height , int(second_y / original_height * mask_height)))
            box_mask = mask[mask_first_y : mask_second_y , mask_first_x : mask_second_x]

            if second_x <= first_x or second_y <= first_y:
                continue
            if label == car_label_id:
                if box_mask.size == 0 or np.count_nonzero(box_mask == car_mask_class_id) / box_mask.size < car_mask_ratio_threshold:
                    continue

            boxes.append([first_x , first_y , second_x , second_y])

            selected_scores.append(scores[y , x].item())
            selected_labels.append(label)

        boxes = torch.tensor(boxes , dtype = torch.float32).reshape(-1 , 4)
        selected_scores = torch.tensor(selected_scores , dtype = torch.float32)
        self.filtered_labels = torch.tensor(selected_labels , dtype = torch.int64)
        if len(boxes) > 0:
            kept = self.non_maximum_suppression(boxes , selected_scores , nms_threshold)
            boxes = boxes[kept]
            selected_scores = selected_scores[kept]
            self.filtered_labels = self.filtered_labels[kept]

        return boxes , selected_scores

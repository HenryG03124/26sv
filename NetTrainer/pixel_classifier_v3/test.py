"""Run V3 on an RGB image, or evaluate steering labels from a collector CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0 , str(PROJECT_ROOT))

from Models.pc_v3 import UNet
from NetTrainer.pixel_classifier_v3.steering_dataset import IMAGE_SIZE , SteeringDataset , image_to_tensor

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(path , device):
    checkpoint = torch.load(path , map_location = "cpu" , weights_only = True)
    if checkpoint.get("format_version") != 3 or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a V3 checkpoint produced by train.py (including target/task metadata)")
    if checkpoint.get("image_size") != list(IMAGE_SIZE):
        raise ValueError("Checkpoint image size does not match V3 preprocessing")
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model , checkpoint


def predict_image(model , checkpoint , image_path , output_dir , device , lane_threshold):
    with Image.open(image_path) as source:
        original_size = source.size
        tensor = image_to_tensor(source).unsqueeze(0).to(device)
    tasks = set(checkpoint["trained_tasks"])
    result = {"image": str(image_path) , "trained_tasks": sorted(tasks)}
    output_dir.mkdir(parents = True , exist_ok = True)
    with torch.inference_mode():
        features = model(tensor)
        if "steering" in tasks:
            result.update(steering = model.head(tensor , features , "steering").item() ,
                          steering_target = checkpoint["steering_target"] , steering_positive = "left")
        mask = None
        if "road" in tasks:
            mask = model.head(tensor , features , "road").argmax(1).squeeze(0)
        if "lane" in tasks:
            lane = model.head(tensor , features , "lane").softmax(1)[: , 1].squeeze(0) > lane_threshold
            if mask is None:
                mask = torch.zeros_like(lane , dtype = torch.long)
            mask[lane & (mask != 2)] = 3
        if mask is not None:
            image = Image.fromarray(mask.cpu().numpy().astype(np.uint8)).convert("P")
            image.putpalette([0 , 0 , 0 , 0 , 255 , 0 , 255 , 0 , 0 , 255 , 255 , 0] + [0 , 0 , 0] * 252)
            path = output_dir / f"{image_path.stem}_mask_v3.png"
            image.resize(original_size , Image.Resampling.NEAREST).save(path)
            result["mask"] = str(path)
    destination = output_dir / f"{image_path.stem}_prediction_v3.json"
    destination.write_text(json.dumps(result , indent = 2 , allow_nan = False) , encoding = "utf-8")
    print(json.dumps(result , indent = 2 , allow_nan = False))
    return result


def evaluate_csv(model , checkpoint , source , destination , device , batch_size = 8 , split_file = None):
    if "steering" not in checkpoint["trained_tasks"]:
        raise ValueError("This checkpoint has no trained steering head")
    dataset = SteeringDataset(source , target = checkpoint["steering_target"])
    if split_file:
        paths = set(json.loads(split_file.read_text(encoding = "utf-8"))["val"])
        records = [record for record in dataset.records if str(record.image_path) in paths]
        if len(records) != len(paths):
            raise ValueError("Evaluation source does not contain every held-out image from the split file")
        dataset = SteeringDataset(target = dataset.target , records = records)
    loader = DataLoader(dataset , batch_size = batch_size , shuffle = False)
    destination.parent.mkdir(parents = True , exist_ok = True)
    absolute = squared = 0.0
    count = 0
    with destination.open("w" , encoding = "utf-8" , newline = "") as file , torch.inference_mode():
        writer = csv.writer(file)
        writer.writerow(["timestamp" , "image" , "target_column" , "target" , "prediction" , "absolute_error"])
        for images , targets in loader:
            predictions = model(images.to(device) , task = "steering").cpu()
            errors = predictions - targets
            absolute += errors.abs().sum().item()
            squared += errors.square().sum().item()
            for target , prediction , error in zip(targets[: , 0] , predictions[: , 0] , errors[: , 0]):
                record = dataset.records[count]
                writer.writerow([record.timestamp , str(record.image_path) , dataset.target , target.item() , prediction.item() , error.abs().item()])
                count += 1
    result = {"samples": count , "target": dataset.target , "mae": absolute / count , "rmse": (squared / count) ** 0.5}
    destination.with_suffix(".metrics.json").write_text(json.dumps(result , indent = 2 , allow_nan = False) , encoding = "utf-8")
    print(json.dumps(result , indent = 2 , allow_nan = False))
    return result


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--checkpoint" , type = Path , default = SCRIPT_DIR / "pc_model_v3.pth")
    parser.add_argument("--image" , type = Path)
    parser.add_argument("--csv" , type = Path , help = "Collector CSV or directory of sessions")
    parser.add_argument("--split-file" , type = Path , help = "Evaluate only val images in train.py's .split.json")
    parser.add_argument("--output-dir" , type = Path , default = SCRIPT_DIR / "predictions")
    parser.add_argument("--device" , default = device)
    parser.add_argument("--batch-size" , type = int , default = 8)
    parser.add_argument("--lane-threshold" , type = float , default = 0.7)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or not 0 <= args.lane_threshold <= 1:
        parser.error("Batch size must be positive; lane threshold must be in [0, 1]")
    if args.split_file and not args.csv:
        parser.error("--split-file requires --csv")
    model , checkpoint = load_model(args.checkpoint , args.device)
    if args.image or not args.csv:
        predict_image(model , checkpoint , args.image or SCRIPT_DIR / "test.jpg" , args.output_dir , args.device , args.lane_threshold)
    if args.csv:
        evaluate_csv(model , checkpoint , args.csv , args.output_dir / "steering_predictions.csv" , args.device , args.batch_size , args.split_file)


if __name__ == "__main__":
    main()

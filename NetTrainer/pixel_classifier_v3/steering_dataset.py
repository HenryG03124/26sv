"""Load collector CSVs without requiring road/lane segmentation masks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

IMAGE_SIZE = (640 , 352)  # PIL width, height; shared with inference
TARGETS = ("userSteer" , "gameSteer")


def image_to_tensor(image):
    image = image.convert("RGB").resize(IMAGE_SIZE , Image.Resampling.BILINEAR)
    array = np.array(image , dtype = np.uint8 , copy = True)
    return torch.from_numpy(array).permute(2 , 0 , 1).contiguous().float().div_(255)


@dataclass(frozen = True)
class SteeringRecord:
    timestamp: int
    image_path: Path
    user_steer: float
    game_steer: float
    session: str


class SteeringDataset(Dataset):
    def __init__(self , source = None , target = "userSteer" , * , records = None):
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")
        self.target = target
        if records is not None:
            self.records = list(records)
        else:
            source = Path(source).expanduser().resolve()
            csvs = sorted(source.rglob("samples.csv")) if source.is_dir() else [source]
            self.records = [record for path in csvs for record in self._read_csv(path)]
        if not self.records:
            raise ValueError(f"No steering samples found in {source}; run tools/data_collector.py first")
        paths = [record.image_path for record in self.records]
        if len(paths) != len(set(paths)):
            raise ValueError("Steering dataset contains duplicate image paths")

    @staticmethod
    def _read_csv(path):
        with path.open(encoding = "utf-8-sig" , newline = "") as file:
            reader = csv.DictReader(file)
            required = {"timestamp" , "image" , "userSteer" , "gameSteer" , "speed" , "throttle" , "brake"}
            if missing := required - set(reader.fieldnames or []):
                raise ValueError(f"Missing CSV columns {sorted(missing)}: {path}")
            timestamps = set()
            for line , row in enumerate(reader , 2):
                try:
                    timestamp = int(row["timestamp"])
                    image_text = row["image"].strip()
                    values = {key: float(row[key]) for key in required - {"timestamp" , "image"}}
                    if timestamp <= 0 or timestamp in timestamps or not image_text:
                        raise ValueError("Invalid/duplicate timestamp or empty image")
                    if not all(math.isfinite(value) for value in values.values()):
                        raise ValueError("Non-finite telemetry")
                    if any(not -1 <= values[key] <= 1 for key in TARGETS):
                        raise ValueError("Steering must be in [-1, 1]")
                    if any(not 0 <= values[key] <= 1 for key in ("throttle" , "brake")):
                        raise ValueError("Throttle/brake must be in [0, 1]")
                except (ValueError , TypeError , AttributeError) as error:
                    raise ValueError(f"Invalid sample {path}:{line}: {error}") from error
                image_path = Path(image_text)
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                image_path = image_path.resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing image at {path}:{line}: {image_path}")
                timestamps.add(timestamp)
                yield SteeringRecord(timestamp , image_path , values["userSteer"] , values["gameSteer"] , str(path.parent))

    def __len__(self):
        return len(self.records)

    def __getitem__(self , index):
        record = self.records[index]
        with Image.open(record.image_path) as image:
            tensor = image_to_tensor(image)
        value = record.user_steer if self.target == "userSteer" else record.game_steer
        return tensor , torch.tensor([value] , dtype = torch.float32)


def split_steering_dataset(dataset , val_fraction = 0.2 , seed = 42 , gap_seconds = 1.0):
    """Hold out whole sessions; for one session, hold out the tail with a gap."""
    if not 0 < val_fraction < 1 or not math.isfinite(gap_seconds) or gap_seconds < 0:
        raise ValueError("Need 0 < val_fraction < 1 and finite gap_seconds >= 0")
    sessions = sorted({record.session for record in dataset.records})
    if len(sessions) > 1:
        random.Random(seed).shuffle(sessions)
        val_count = max(1 , min(len(sessions) - 1 , round(len(sessions) * val_fraction)))
        held_out = set(sessions[:val_count])
        train = [record for record in dataset.records if record.session not in held_out]
        val = [record for record in dataset.records if record.session in held_out]
    else:
        ordered = sorted(dataset.records , key = lambda record: record.timestamp)
        count = max(1 , math.ceil(len(ordered) * val_fraction))
        val = ordered[-count:]
        cutoff = val[0].timestamp - int(gap_seconds * 1e9)
        train = [record for record in ordered[:-count] if record.timestamp < cutoff]
    if not train or not val:
        raise ValueError("Too few samples for a separate validation set and time gap; collect a longer/second session")
    return (SteeringDataset(target = dataset.target , records = train) ,
            SteeringDataset(target = dataset.target , records = val))


def ensure_disjoint(train , val):
    overlap = {record.image_path for record in train.records} & {record.image_path for record in val.records}
    if overlap:
        raise ValueError(f"Training and validation share {len(overlap)} image(s)")

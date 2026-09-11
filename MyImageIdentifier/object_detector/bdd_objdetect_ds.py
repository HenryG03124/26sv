"""BDD100K detection inputs with the same image preprocessing as the ResNet project."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

TARGET_HEIGHT = 256
TARGET_WIDTH = 448
# 0 is reserved for a detector's background class; annotation IDs are 1..10.
CLASS_NAMES = (
    'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motor', 'bike', 'traffic light', 'traffic sign',
)
CATEGORY_TO_ID = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class DetectionRecord:
    stem: str
    image_path: Path
    annotation_path: Path
    width: int
    height: int


class BDDDetectionDataset(Dataset):
    """Return image (3,256,448) and a variable-length detection target.

    Images are RGB float32 in [0,1], resized with PIL bilinear interpolation,
    exactly as in BDDVehicleRoadDataset. Boxes are float32 XYXY coordinates in
    the resized 448x256 image, not normalized coordinates. Labels are int64,
    with 0 reserved for background and object IDs 1..10.
    """

    def __init__(self, pairs_csv: str | Path):
        self.pairs_csv = Path(pairs_csv).expanduser().resolve()
        self.records: list[DetectionRecord] = []
        seen = set()
        with self.pairs_csv.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            required = {'stem', 'image_path', 'annotation_path', 'width', 'height'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f'Missing columns: {required - set(reader.fieldnames or [])}')
            for row in reader:
                stem = row['stem'].strip()
                if not stem or stem in seen:
                    raise ValueError(f'Empty or duplicate image ID: {stem!r}')
                paths = []
                for key in ('image_path', 'annotation_path'):
                    path = Path(row[key])
                    path = path if path.is_absolute() else self.pairs_csv.parent / path
                    # prepare_bdd_objdetect validates these paths while creating the
                    # manifest. Avoid per-record filesystem lookups during training
                    # startup; missing files will still fail clearly in __getitem__.
                    paths.append(path)
                width, height = int(row['width']), int(row['height'])
                if width <= 0 or height <= 0:
                    raise ValueError(f'Invalid image dimensions: {stem}')
                self.records.append(DetectionRecord(stem, *paths, width, height))
                seen.add(stem)
        if not self.records:
            raise ValueError(f'Empty dataset: {self.pairs_csv}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.image_path) as image_file:
            if image_file.size != (record.width, record.height):
                raise ValueError(f'Image dimensions changed: {record.image_path}')
            image = image_file.convert('RGB').resize(
                (TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.BILINEAR)
            array = np.array(image, dtype=np.uint8, copy=True)
        image_tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float().div_(255)
        annotation = json.loads(record.annotation_path.read_text(encoding='utf-8'))
        boxes = torch.as_tensor(annotation['boxes'], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(annotation['labels'], dtype=torch.int64)
        if labels.ndim != 1 or len(boxes) != len(labels):
            raise ValueError(f'Boxes/labels mismatch: {record.stem}')
        if not torch.isfinite(boxes).all() or not ((labels >= 1) & (labels <= len(CLASS_NAMES))).all():
            raise ValueError(f'Invalid detection target: {record.stem}')
        if len(boxes):
            if not ((boxes[:, 2:] > boxes[:, :2]).all()
                    and (boxes[:, :2] >= 0).all()
                    and (boxes[:, [0, 2]] <= record.width).all()
                    and (boxes[:, [1, 3]] <= record.height).all()):
                raise ValueError(f'Invalid box coordinates: {record.stem}')
        scale = torch.tensor([TARGET_WIDTH / record.width, TARGET_HEIGHT / record.height] * 2,
                             dtype=torch.float32)
        boxes = boxes * scale
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([index], dtype=torch.int64),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
            'orig_size': torch.tensor([record.height, record.width], dtype=torch.int64),
            'size': torch.tensor([TARGET_HEIGHT, TARGET_WIDTH], dtype=torch.int64),
        }
        return image_tensor, target


def detection_collate_fn(batch):
    """Stack fixed-size images; retain one target dict per image (no box padding)."""
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)

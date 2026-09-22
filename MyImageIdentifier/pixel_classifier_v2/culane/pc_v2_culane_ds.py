"""Load CULane image / binary boundary-mask pairs at the shared V2 input size."""

import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

class SegmentationAugment:
    """Apply mild geometry to both inputs and color changes only to the image.

    Call after resizing. Masks use nearest-neighbor sampling; newly exposed
    pixels are ignored (255), never treated as labeled background. Randomness
    comes from torch so DataLoader workers receive independent, seeded streams.
    """

    def __init__(self, flip_probability=0.5, affine_probability=0.5, color_probability=0.8):
        self.flip_probability = flip_probability
        self.affine_probability = affine_probability
        self.color_probability = color_probability

    @staticmethod
    def _uniform(low, high):
        return low + (high - low) * torch.rand(()).item()

    def __call__(self, image, mask):
        if torch.rand(()).item() < self.flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if torch.rand(()).item() < self.affine_probability:
            width, height = image.size
            angle = math.radians(self._uniform(-3.0, 3.0))
            scale = self._uniform(0.95, 1.05)
            dx = self._uniform(-0.04, 0.04) * width
            dy = self._uniform(-0.03, 0.03) * height
            cx, cy = width / 2, height / 2
            # PIL maps output coordinates back to the source image.
            a, b = math.cos(angle) / scale, math.sin(angle) / scale
            matrix = (a, b, cx - a * (cx + dx) - b * (cy + dy),
                      -b, a, cy + b * (cx + dx) - a * (cy + dy))
            image = image.transform(image.size, Image.Transform.AFFINE, matrix,
                                    resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
            mask = mask.transform(mask.size, Image.Transform.AFFINE, matrix,
                                  resample=Image.Resampling.NEAREST, fillcolor=255)

        if torch.rand(()).item() < self.color_probability:
            for enhancer, low, high in ((ImageEnhance.Brightness, 0.75, 1.25),
                                        (ImageEnhance.Contrast, 0.8, 1.2),
                                        (ImageEnhance.Color, 0.8, 1.2)):
                image = enhancer(image).enhance(self._uniform(low, high))

        return image, mask


class CULaneDataset(Dataset):
    def __init__(self , pairs_csv , augment = False):
        self.pairs_csv = Path(pairs_csv).resolve()
        self.augmentation = SegmentationAugment() if augment else None
        with self.pairs_csv.open(encoding = "utf-8-sig" , newline = "") as file:
            reader = csv.DictReader(file)
            if not {"image_path" , "mask_path"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing image_path/mask_path columns: {self.pairs_csv}")
            self.records = list(reader)
        if not self.records:
            raise ValueError(f"Empty dataset: {self.pairs_csv}")
        for row_number , record in enumerate(self.records , start = 2):
            if not record.get("image_path") or not record.get("mask_path"):
                raise ValueError(f"Empty image/mask path at row {row_number}: {self.pairs_csv}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self , index):
        record = self.records[index]
        with Image.open(self.pairs_csv.parent / record["image_path"]) as source_image , \
                Image.open(self.pairs_csv.parent / record["mask_path"]) as source_mask:
            if source_image.size != source_mask.size:
                raise ValueError(f"Image/mask dimensions differ: {record['image_path']}")
            # Preserve palette indices; converting a palette mask to L changes class IDs.
            if source_mask.mode not in ("L" , "P"):
                raise ValueError(f"Expected indexed mask: {record['mask_path']}")
            image = source_image.convert("RGB").resize((640 , 352) , Image.Resampling.BILINEAR)
            mask = source_mask.resize((640 , 352) , Image.Resampling.NEAREST)
            mask_array = np.array(mask , dtype = np.uint8 , copy = True)
            if not np.isin(mask_array , (0 , 1 , 255)).all():
                raise ValueError(f"Expected binary mask values 0/1/255: {record['mask_path']}")
            mask = Image.fromarray(mask_array)
        if self.augmentation is not None:
            image , mask = self.augmentation(image , mask)
        image = torch.from_numpy(np.array(image , dtype = np.uint8 , copy = True))
        image = image.permute(2 , 0 , 1).contiguous().float().div_(255.0)
        mask = torch.from_numpy(np.array(mask , dtype = np.uint8 , copy = True)).long()
        return image , mask

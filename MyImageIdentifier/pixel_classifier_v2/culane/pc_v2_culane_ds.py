"""Load CULane image / binary boundary-mask pairs at the shared V2 input size."""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CULaneDataset(Dataset):
    def __init__(self , pairs_csv):
        self.pairs_csv = Path(pairs_csv).resolve()
        with self.pairs_csv.open(encoding = "utf-8-sig" , newline = "") as file:
            self.records = list(csv.DictReader(file))

    def __len__(self):
        return len(self.records)

    def __getitem__(self , index):
        record = self.records[index]
        with Image.open(self.pairs_csv.parent / record["image_path"]) as image:
            image = image.convert("RGB").resize((640 , 352) , Image.Resampling.BILINEAR)
            image = torch.from_numpy(np.array(image , dtype = np.uint8 , copy = True))
            image = image.permute(2 , 0 , 1).contiguous().float().div(255.0)
        with Image.open(self.pairs_csv.parent / record["mask_path"]) as mask:
            mask = mask.resize((640 , 352) , Image.Resampling.NEAREST)
            mask = torch.from_numpy(np.array(mask , dtype = np.int64 , copy = True))
        return image , mask

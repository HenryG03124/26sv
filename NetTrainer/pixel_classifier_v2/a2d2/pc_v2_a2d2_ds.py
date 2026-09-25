"""Load image/mask pairs for the selected A2D2 task, using the V2 Dataset interface."""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class A2D2Dataset(Dataset):
    def __init__(self , pairs_csv , task):
        self.pairs_csv = Path(pairs_csv).resolve()
        self.mask_column = {"road": "road_mask_path" , "lane": "mask_path"}[task]
        with self.pairs_csv.open(encoding = "utf-8-sig" , newline = "") as file:
            self.records = list(csv.DictReader(file))

    def __len__(self):
        return len(self.records)

    def read_mask(self , path):
        with Image.open(self.pairs_csv.parent / path) as mask:
            mask = mask.resize((640 , 352) , Image.Resampling.NEAREST)
            return torch.from_numpy(np.array(mask , dtype = np.int64 , copy = True))

    def __getitem__(self , index):
        record = self.records[index]
        with Image.open(self.pairs_csv.parent / record["image_path"]) as image:
            image = image.convert("RGB").resize((640 , 352) , Image.Resampling.BILINEAR)
            image = torch.from_numpy(np.array(image , dtype = np.uint8 , copy = True))
            image = image.permute(2 , 0 , 1).contiguous().float().div(255.0)
        mask = self.read_mask(record[self.mask_column])
        return image , mask

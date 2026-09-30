"""PyTorch Dataset for the converted BDD100K lane-marking masks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# PIL resize receives (width, height), while tensor shapes use (height, width).
TARGET_HEIGHT = 352
TARGET_WIDTH = 640
TARGET_PIL_SIZE = (TARGET_WIDTH, TARGET_HEIGHT)

@dataclass(frozen=True)
class PairRecord:
    """One image/mask pair resolved to absolute local paths."""

    stem: str
    image_path: Path
    mask_path: Path


class BDDLaneDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read image/mask pairs listed in a generated CSV file.

    Returned values:
      image: float32 tensor, shape (3, 352, 640), values in [0, 1]
      mask:  int64/long tensor, shape (352, 640), values in {0, 1, 255}

    Class 0 is background and class 1 is a BDD100K lane marking. The same
    image/mask loading logic is also used for background/road/car masks.
    """

    def __init__(self, pairs_csv: str | Path) -> None:
        self.pairs_csv = Path(pairs_csv).expanduser().resolve()
        if not self.pairs_csv.is_file():
            raise FileNotFoundError(f"配对 CSV 不存在：{self.pairs_csv}")

        self.records = self._read_records()
        if not self.records:
            raise ValueError(f"配对 CSV 中没有数据：{self.pairs_csv}")

    def _resolve_csv_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.pairs_csv.parent / path
        return path.resolve()

    def _read_records(self) -> list[PairRecord]:
        records: list[PairRecord] = []
        seen_stems: set[str] = set()

        with self.pairs_csv.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            required_columns = {"stem", "image_path", "mask_path"}
            actual_columns = set(reader.fieldnames or [])
            missing_columns = required_columns - actual_columns
            if missing_columns:
                raise ValueError(
                    f"CSV 缺少列 {sorted(missing_columns)}：{self.pairs_csv}"
                )

            for line_number, row in enumerate(reader, start=2):
                stem = (row.get("stem") or "").strip()
                image_text = (row.get("image_path") or "").strip()
                mask_text = (row.get("mask_path") or "").strip()
                if not stem or not image_text or not mask_text:
                    raise ValueError(
                        f"CSV 第 {line_number} 行存在空字段：{self.pairs_csv}"
                    )
                if stem in seen_stems:
                    raise ValueError(f"CSV 第 {line_number} 行 stem 重复：{stem}")

                image_path = self._resolve_csv_path(image_text)
                mask_path = self._resolve_csv_path(mask_text)
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"CSV 第 {line_number} 行原图不存在：{image_path}"
                    )
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"CSV 第 {line_number} 行 mask 不存在：{mask_path}"
                    )

                records.append(PairRecord(stem, image_path, mask_path))
                seen_stems.add(stem)

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]

        with Image.open(record.image_path) as image_file:
            image = image_file.convert("RGB").resize(
                TARGET_PIL_SIZE, resample=Image.Resampling.BILINEAR
            )
            image_array = np.array(image, dtype=np.uint8, copy=True)

        with Image.open(record.mask_path) as mask_file:
            if mask_file.mode != "L":
                raise ValueError(
                    f"转换后的 mask 必须是单通道 L 模式："
                    f"{record.mask_path}，实际为 {mask_file.mode}"
                )
            mask = mask_file.resize(
                TARGET_PIL_SIZE, resample=Image.Resampling.NEAREST
            )
            mask_array = np.array(mask, dtype=np.uint8, copy=True)

        if image_array.shape != (TARGET_HEIGHT, TARGET_WIDTH, 3):
            raise ValueError(
                f"图片 resize 后形状错误：{record.image_path}，"
                f"实际为 {image_array.shape}"
            )
        if mask_array.shape != (TARGET_HEIGHT, TARGET_WIDTH):
            raise ValueError(
                f"mask resize 后形状错误：{record.mask_path}，"
                f"实际为 {mask_array.shape}"
            )

        image_tensor = (
            torch.from_numpy(image_array)
            .permute(2, 0, 1)
            .contiguous()
            .to(dtype=torch.float32)
            .div_(255.0)
        )
        mask_tensor = torch.from_numpy(mask_array).to(dtype=torch.long)

        return image_tensor, mask_tensor

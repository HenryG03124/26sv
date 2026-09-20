"""Prepare official CULane train/val images and binary lane-boundary masks."""

import argparse
import csv
import io
import json
import os
import shutil
import tarfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"


def read_splits(source_root):
    splits = {}
    with tarfile.open(source_root / "list.tar.gz") as archive:
        for split in ("train" , "val"):
            content = archive.extractfile("list/" + split + "_gt.txt").read().decode("utf-8")
            records = []
            for line in content.splitlines():
                if not line.strip():
                    continue
                image , label , *exists = line.split()
                image , label = image.lstrip("/") , label.lstrip("/")
                # Only relative paths from the official lists are used as output names.
                for key in (image , label):
                    if ".." in Path(key).parts or ":" in key or "\\" in key:
                        raise ValueError("Invalid CULane path: " + key)
                records.append((image , label))
            splits[split] = records
    images = [image for records in splits.values() for image , label in records]
    if len(images) != len(set(images)):
        raise ValueError("Duplicate images or overlapping train/val splits")
    return splits


def extract_images(source_root , raw_root , image_keys):
    found = set()
    for path in sorted(source_root.glob("driver*.zip")):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                # The mirror repeats the driver directory; the last three parts
                # match official image keys (driver / video / frame).
                key = "/".join(member.filename.split("/")[-3:])
                image_key = key.replace(".lines.txt" , ".jpg")
                if image_key not in image_keys:
                    continue
                target = raw_root / key
                target.parent.mkdir(parents = True , exist_ok = True)
                if not target.exists() or target.stat().st_size != member.file_size:
                    temporary = target.with_suffix(target.suffix + ".tmp")
                    with archive.open(member) as source , temporary.open("wb") as output:
                        shutil.copyfileobj(source , output)
                    temporary.replace(target)
                if key.endswith(".jpg"):
                    found.add(key)
                    if len(found) % 2000 == 0:
                        print("v2_culane images:" , len(found) , "/" , len(image_keys) , flush = True)
    missing = image_keys - found
    if missing:
        raise RuntimeError(f"Missing {len(missing)} CULane images, e.g. {next(iter(missing))}")


def convert_mask(label):
    values = np.asarray(label)
    if values.ndim != 2 or not np.isin(values , [0 , 1 , 2 , 3 , 4 , 255]).all():
        raise ValueError("Expected indexed CULane labels: 0, 1-4 or 255")
    mask = ((values >= 1) & (values <= 4)).astype(np.uint8)
    mask[values == 255] = 255
    return Image.fromarray(mask)


def prepare_mask(job):
    payload , raw_path , mask_path = job
    if not raw_path.exists():
        raw_path.parent.mkdir(parents = True , exist_ok = True)
        temporary = raw_path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(raw_path)
    if not mask_path.exists():
        mask_path.parent.mkdir(parents = True , exist_ok = True)
        with Image.open(io.BytesIO(payload)) as label:
            mask = convert_mask(label)
        temporary = mask_path.with_suffix(".tmp")
        mask.save(temporary , format = "PNG" , compress_level = 1)
        temporary.replace(mask_path)


def prepare_labels(source_root , raw_root , output_root , splits , workers):
    targets = {}
    rows = {"train": [] , "val": []}
    for split , records in splits.items():
        for image , label in records:
            stem = "__".join(Path(image).with_suffix("").parts)
            mask_key = Path(split) / "masks" / Path(image).with_suffix(".png")
            targets[label] = output_root / mask_key
            rows[split].append((stem , Path(os.path.relpath(raw_root / image , output_root)).as_posix() , mask_key.as_posix()))
    found = set()
    batch = []
    with ThreadPoolExecutor(max_workers = workers) as executor , tarfile.open(source_root / "laneseg_label_w16.tar.gz" , "r|gz") as archive:
        for member in archive:
            key = member.name.removeprefix("./")
            if not member.isfile() or key not in targets:
                continue
            payload = archive.extractfile(member).read()
            batch.append((payload , raw_root / key , targets[key]))
            found.add(key)
            if len(batch) == 64:
                list(executor.map(prepare_mask , batch))
                batch.clear()
                if len(found) % 2048 == 0:
                    print("v2_culane masks:" , len(found) , "/" , len(targets) , flush = True)
        list(executor.map(prepare_mask , batch))
    missing = targets.keys() - found
    if missing:
        raise RuntimeError(f"Missing {len(missing)} CULane labels, e.g. {next(iter(missing))}")
    return rows


def prepare_dataset(source_root , raw_root , output_root , workers = 4):
    splits = read_splits(source_root)
    raw_root.mkdir(parents = True , exist_ok = True)
    output_root.mkdir(parents = True , exist_ok = True)
    print("CULane split sizes:" , {key: len(value) for key , value in splits.items()} , flush = True)
    extract_images(source_root , raw_root , {image for records in splits.values() for image , label in records})
    rows = prepare_labels(source_root , raw_root , output_root , splits , workers)
    for split , records in rows.items():
        temporary = output_root / (split + "_pairs.csv.tmp")
        with temporary.open("w" , encoding = "utf-8-sig" , newline = "") as file:
            writer = csv.writer(file)
            writer.writerow(("stem" , "image_path" , "mask_path"))
            writer.writerows(records)
        temporary.replace(output_root / (split + "_pairs.csv"))
    (raw_root / "DATA_SOURCE.txt").write_text(
        "CULane: https://xingangpan.github.io/projects/CULane.html\n"
        "Official train/val split and laneseg_label_w16 labels. Test split is not extracted.\n"
        "Images and .lines.txt from the downloaded CULane mirror ZIPs.\n"
        "Original instance masks are retained under laneseg_label_w16/.\n"
        "Processed masks: 0=background, 1=lane boundary, 255=ignore (if present).\n"
        "Official lane IDs 1-4 are merged. No extra gap filling or paint extraction.\n"
        "Labels represent continuous 16-pixel-wide boundaries, not precise paint regions.\n"
        "Original resolution is preserved; the loader resizes to 640x352.\n" , encoding = "utf-8")
    status = {"phase": "complete" , "train_pairs": len(rows["train"]) , "val_pairs": len(rows["val"]) ,
              "mask_values": [0 , 1 , 255] , "target": "lane_boundary"}
    (output_root / "prepare_status.json").write_text(json.dumps(status , indent = 2) , encoding = "utf-8")
    print(json.dumps(status) , flush = True)
    return rows


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--source-root" , type = Path , default = Path(r"C:\Downloads\CULane"))
    parser.add_argument("--raw-root" , type = Path , default = DATASETS_DIR / "pc_lane_ds_culane")
    parser.add_argument("--output-root" , type = Path , default = DATASETS_DIR / "processed_pc_lane_ds_culane")
    parser.add_argument("--workers" , type = int , default = 4)
    args = parser.parse_args()
    prepare_dataset(args.source_root.resolve() , args.raw_root.resolve() , args.output_root.resolve() , args.workers)


if __name__ == "__main__":
    main()

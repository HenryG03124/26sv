"""Prepare paired A2D2 road/vehicle and lane masks for Pixel Classifier v2_a2d2.

Use --watch to prepare completed pairs while the A2D2 download is running.
Final train/val CSVs are published only when every expected pair is ready.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image


DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
LANE_COLORS = ((255 , 193 , 37) , (128 , 0 , 255)) # Solid line, dashed line.
IGNORE_COLORS = ((96 , 69 , 143) , (53 , 46 , 82) , (0 , 0 , 0)) # Blur, rain/dirt, void.
ROAD_COLORS = (
    (255 , 0 , 255) , (180 , 50 , 180) , (238 , 233 , 191) , # Normal, cobblestone, slow drive.
    (110 , 110 , 0) , (200 , 125 , 210) , (210 , 50 , 115) , # Bumps, painted instructions, zebra.
) + LANE_COLORS
CAR_COLORS = (
    (255 , 0 , 0) , (200 , 0 , 0) , (150 , 0 , 0) , (128 , 0 , 0) , # Cars.
    (255 , 128 , 0) , (200 , 128 , 0) , (150 , 128 , 0) , # Trucks.
    (255 , 255 , 0) , (255 , 255 , 200) , (0 , 0 , 100) , # Utility vehicles, tractor.
)


def convert_mask(label):
    colors = np.asarray(label.convert("RGB"))
    mask = np.zeros(colors.shape[:2] , dtype = np.uint8)
    for color in LANE_COLORS:
        mask[np.all(colors == color , axis = 2)] = 1
    for color in IGNORE_COLORS:
        mask[np.all(colors == color , axis = 2)] = 255
    return Image.fromarray(mask)


def convert_road_mask(label):
    colors = np.asarray(label.convert("RGB"))
    mask = np.zeros(colors.shape[:2] , dtype = np.uint8)
    for color in ROAD_COLORS:
        mask[np.all(colors == color , axis = 2)] = 1
    for color in CAR_COLORS:
        mask[np.all(colors == color , axis = 2)] = 2
    for color in IGNORE_COLORS:
        mask[np.all(colors == color , axis = 2)] = 255
    return Image.fromarray(mask)


def split_scenes(image_keys):
    scenes = sorted({Path(key).parts[1] for key in image_keys})
    val_scenes = set(random.Random(42).sample(scenes , max(1 , round(len(scenes) * 0.2))))
    return {scene: "val" if scene in val_scenes else "train" for scene in scenes}


def label_key(image_key):
    return image_key.replace("/camera/" , "/label/").replace("_camera_" , "_label_")


def prepare_pair(image_key , split , source_root , raw_root , output_root):
    annotation_key = label_key(image_key)
    for key in (image_key , annotation_key):
        target = raw_root / key
        target.parent.mkdir(parents = True , exist_ok = True)
        if not target.exists():
            os.link(source_root / key , target) # Reuse original files without duplicating storage.

    stem = Path(image_key).stem.replace("_camera_" , "_")
    mask_path = output_root / split / "masks" / (stem + ".png")
    road_mask_path = output_root / split / "road_masks" / (stem + ".png")
    for target , convert in ((mask_path , convert_mask) , (road_mask_path , convert_road_mask)):
        target.parent.mkdir(parents = True , exist_ok = True)
        if not target.exists():
            with Image.open(raw_root / annotation_key) as label:
                mask = convert(label)
            temporary = target.with_suffix(".tmp")
            mask.save(temporary , format = "PNG" , compress_level = 1)
            temporary.replace(target)

    return (stem , Path(os.path.relpath(raw_root / image_key , output_root)).as_posix() ,
            mask_path.relative_to(output_root).as_posix() , road_mask_path.relative_to(output_root).as_posix())


def write_pairs(output_root , rows):
    for split in ("train" , "val"):
        temporary = output_root / (split + "_pairs.csv.tmp")
        with temporary.open("w" , encoding = "utf-8-sig" , newline = "") as file:
            writer = csv.writer(file)
            writer.writerow(("stem" , "image_path" , "mask_path" , "road_mask_path"))
            writer.writerows(sorted(rows[split]))
        temporary.replace(output_root / (split + "_pairs.csv"))


def prepare_dataset(source_root , raw_root , output_root , watch = False):
    manifest = json.loads((source_root / "download_manifest.json").read_text(encoding = "utf-8"))
    image_keys = sorted(entry["key"] for entry in manifest["files"]
                        if "/camera/cam_front_center/" in entry["key"] and entry["key"].endswith(".png"))
    scene_splits = split_scenes(image_keys)
    raw_root.mkdir(parents = True , exist_ok = True)
    output_root.mkdir(parents = True , exist_ok = True)
    (output_root / "scene_splits.json").write_text(json.dumps(scene_splits , indent = 2) , encoding = "utf-8")
    (raw_root / "DATA_SOURCE.txt").write_text(
        "A2D2 front-center camera, camera_lidar_semantic subset\n"
        "Source: https://registry.opendata.aws/aev-a2d2/\n"
        "License: https://creativecommons.org/licenses/by-nd/4.0/\n"
        "Original images and RGB labels are hard-linked from the download directory.\n"
        "Masks: 0=background, 1=solid/dashed paint, 255=blur/rain-dirt/void.\n"
        "Arrows, zebra crossings and other painted instructions remain background.\n"
        "Road masks: 0=background, 1=road (including paint), 2=vehicle, 255=ignore.\n"
        "Vehicles merge cars, trucks, utility vehicles and tractors; ego car is excluded.\n"
        "Parking, restricted/non-drivable areas and sidewalks are not road targets.\n"
        "Original resolution is preserved. The dataset loader resizes to 640x352.\n"
        "Scene-disjoint validation split: 20% of scenes, random seed 42.\n" , encoding = "utf-8")

    rows = {"train": [] , "val": []}
    pending = image_keys
    while pending:
        remaining = []
        for key in pending:
            if not (source_root / key).is_file() or not (source_root / label_key(key)).is_file():
                remaining.append(key)
                continue
            split = scene_splits[Path(key).parts[1]]
            rows[split].append(prepare_pair(key , split , source_root , raw_root , output_root))
            prepared = len(rows["train"]) + len(rows["val"])
            if prepared % 100 == 0:
                print("V3 prepared pairs:" , prepared , "/" , len(image_keys) , flush = True)

        pending = remaining
        status = {"phase": "waiting_for_download" if pending else "writing_csv" ,
                  "expected_pairs": len(image_keys) , "train_pairs": len(rows["train"]) ,
                  "val_pairs": len(rows["val"]) , "remaining_pairs": len(pending)}
        (output_root / "prepare_status.json").write_text(json.dumps(status , indent = 2) , encoding = "utf-8")
        print(json.dumps(status) , flush = True)
        if not pending:
            break
        if not watch:
            raise RuntimeError("A2D2 download is incomplete; use --watch to wait. Training CSVs were not published.")
        download_status = source_root / "download_status.json"
        if download_status.exists():
            try:
                phase = json.loads(download_status.read_text(encoding = "utf-8"))["phase"]
            except json.JSONDecodeError: # The downloader may be updating this file.
                phase = "downloading"
            if phase in ("complete" , "incomplete"):
                raise RuntimeError("Downloader stopped with missing pairs; resume the download and rerun preparation.")
        time.sleep(30)

    write_pairs(output_root , rows)
    status["phase"] = "complete"
    (output_root / "prepare_status.json").write_text(json.dumps(status , indent = 2) , encoding = "utf-8")
    print(json.dumps(status) , flush = True)
    return rows


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--source-root" , type = Path , default = Path(r"C:\Downloads\A2D2"))
    parser.add_argument("--raw-root" , type = Path , default = DATASETS_DIR / "pc_lane_A2D2_ds")
    parser.add_argument("--output-root" , type = Path , default = DATASETS_DIR / "processed_pc_lane_A2D2_ds")
    parser.add_argument("--watch" , action = "store_true")
    args = parser.parse_args()
    prepare_dataset(args.source_root.resolve() , args.raw_root.resolve() , args.output_root.resolve() , args.watch)


if __name__ == "__main__":
    main()

"""Extend the active A2D2 manifest while preserving its current pairs."""

import argparse
import json
import random
from collections import Counter , defaultdict , deque
from pathlib import Path


def label_key(key):
    return key.replace("/camera/" , "/label/").replace("_camera_" , "_label_")


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--root" , type = Path , default = Path(r"C:\Downloads\A2D2"))
    parser.add_argument("--train-pairs" , type = int , required = True)
    parser.add_argument("--val-pairs" , type = int , required = True)
    args = parser.parse_args()

    active_path = args.root / "download_manifest.json"
    full_path = args.root / "download_manifest_full.json"
    active = json.loads(active_path.read_text(encoding = "utf-8"))
    full = json.loads(full_path.read_text(encoding = "utf-8"))
    entries = {entry["key"]: entry for entry in full["files"]}
    images = sorted(key for key in entries if "/camera/cam_front_center/" in key and key.endswith(".png"))
    scenes = sorted({Path(key).parts[1] for key in images})
    val_scenes = set(random.Random(42).sample(scenes , max(1 , round(len(scenes) * 0.2))))

    def split(key):
        return "val" if Path(key).parts[1] in val_scenes else "train"

    selected = {entry["key"] for entry in active["files"] if "/camera/cam_front_center/" in entry["key"]}
    targets = {"train": args.train_pairs , "val": args.val_pairs}
    current = Counter(split(key) for key in selected)
    if current["val"] != targets["val"]:
        raise RuntimeError("The existing validation selection would change; no files were modified.")

    generator = random.Random(42)
    for task in ("train" , "val"):
        if current[task] > targets[task]:
            raise RuntimeError("The requested target is smaller than the active selection; no files were modified.")
        groups = defaultdict(list)
        for key in images:
            if key not in selected and split(key) == task:
                groups[Path(key).parts[1]].append(key)
        for group in groups.values():
            generator.shuffle(group)
        queue = deque(sorted(groups))
        while current[task] < targets[task]:
            scene = queue.popleft()
            selected.add(groups[scene].pop())
            current[task] += 1
            if groups[scene]:
                queue.append(scene)

    selected_keys = selected | {label_key(key) for key in selected}
    files = [entry for key , entry in entries.items() if key in selected_keys or not key.endswith(".png")]
    pair_count = args.train_pairs + args.val_pairs
    assert len(selected) == pair_count
    assert all(label_key(key) in entries for key in selected)

    full.update(pairs = pair_count , full_pairs = len(images) , files = files ,
                selection = "Preserve active pairs, then seeded round-robin sampling from training scenes; "
                            f"{args.train_pairs} train + {args.val_pairs} val." ,
                selected_split_counts = dict(Counter(split(key) for key in selected)))
    temporary = active_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(full , indent = 2) , encoding = "utf-8")
    temporary.replace(active_path)
    print(json.dumps({"pairs": pair_count , "split": full["selected_split_counts"] ,
                      "GB": sum(entry["size"] for entry in files) / 1e9}))


if __name__ == "__main__":
    main()

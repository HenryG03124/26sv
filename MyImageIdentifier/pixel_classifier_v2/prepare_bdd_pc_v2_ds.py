"""Prepare binary BDD100K lane masks and train/validation CSV manifests.

The default archive paths use the verified BDD100K files in C:\\Downloads.
Both the official combined 100K image archive and separate train/val image
archives are supported. The official labels archive may contain either two
aggregated JSON arrays or one JSON file per image.

For an already extracted copy, pass --images-root, --train-labels and
--val-labels instead. A flat images directory is supported; only image names
present in the corresponding train/val JSON are selected.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


OFFICIAL_PAGE = "http://bdd-data.berkeley.edu/download.html"
OFFICIAL_IMAGES_URL = "http://128.32.162.150/bdd100k/bdd100k_images_100k.zip"
OFFICIAL_LABELS_URL = "http://128.32.162.150/bdd100k/bdd100k_labels.zip"
TRAIN_IMAGES_MIRROR_URL = (
    "https://huggingface.co/datasets/yiyi159/bdd100k-train/resolve/main/"
    "100k_images_train.zip"
)
VAL_IMAGES_MIRROR_URL = (
    "https://huggingface.co/datasets/hirundo-io/bdd100k-val/resolve/main/"
    "bdd100k_val_hirundo.zip"
)
DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Stream objects from a top-level JSON array without loading it all in RAM."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as file:
        buffer = ""
        position = 0
        started = False
        finished = False

        while not finished:
            chunk = file.read(chunk_size)
            if chunk:
                buffer += chunk

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1

                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"标签 JSON 顶层必须是数组：{path}")
                    started = True
                    position += 1
                    continue

                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1

                if position < len(buffer) and buffer[position] == "]":
                    finished = True
                    position += 1
                    break

                if position >= len(buffer):
                    break

                try:
                    item, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if not chunk:
                        raise ValueError(f"标签 JSON 不完整或格式错误：{path}")
                    break

                if not isinstance(item, dict):
                    raise ValueError(f"标签数组元素不是对象：{path}")
                yield item
                position = end

            if position:
                buffer = buffer[position:]
                position = 0

            if not chunk:
                if not finished:
                    raise ValueError(f"标签 JSON 在数组结束前意外终止：{path}")
                break


def link_or_copy(source: Path, target: Path) -> None:
    """Prefer a space-saving hard link and fall back to a normal copy."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"目标文件已存在且大小不同：{target}")
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def find_label_member(archive: zipfile.ZipFile, split: str) -> zipfile.ZipInfo:
    suffix = f"bdd100k_labels_images_{split}.json"
    matches = [
        item
        for item in archive.infolist()
        if not item.is_dir() and Path(item.filename).name.lower() == suffix
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{archive.filename} 中应有且仅有一个 {suffix}，实际找到 {len(matches)} 个"
        )
    return matches[0]


def official_item_to_legacy(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize one per-image official JSON object to the legacy array schema."""

    if isinstance(item.get("labels"), list):
        return item

    name = str(item.get("name", ""))
    if name and not Path(name).suffix:
        name += ".jpg"

    frames = item.get("frames", [])
    frame = frames[0] if isinstance(frames, list) and frames else {}
    if not isinstance(frame, dict):
        frame = {}

    lane_labels: list[dict[str, Any]] = []
    objects = frame.get("objects", [])
    if not isinstance(objects, list):
        objects = []

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        category = str(obj.get("category", ""))
        if category != "lane" and not category.startswith("lane/"):
            continue

        raw_vertices = obj.get("poly2d", [])
        vertices: list[list[float]] = []
        vertex_types: list[str] = []
        if isinstance(raw_vertices, list):
            for point in raw_vertices:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                vertices.append([float(point[0]), float(point[1])])
                point_type = str(point[2]).upper() if len(point) >= 3 else "L"
                vertex_types.append(point_type if point_type in {"C", "L"} else "L")

        if len(vertices) < 2:
            continue

        source_attributes = obj.get("attributes", {})
        if not isinstance(source_attributes, dict):
            source_attributes = {}
        lane_type = category.split("/", maxsplit=1)[1] if "/" in category else "unknown"
        lane_labels.append(
            {
                "category": "lane",
                "attributes": {
                    "laneDirection": source_attributes.get("direction", "parallel"),
                    "laneStyle": source_attributes.get("style", "solid"),
                    "laneType": lane_type,
                },
                "poly2d": [
                    {
                        "vertices": vertices,
                        "types": "".join(vertex_types),
                        "closed": False,
                    }
                ],
                "id": obj.get("id"),
            }
        )

    return {
        "name": name,
        "attributes": item.get("attributes", {}),
        "timestamp": frame.get("timestamp"),
        "labels": lane_labels,
    }


def iter_label_items(labels_source: Path, split: str) -> Iterator[dict[str, Any]]:
    """Yield normalized frames from an array JSON, JSON directory, or official ZIP."""

    if labels_source.is_dir():
        for path in sorted(labels_source.rglob("*.json")):
            with path.open("r", encoding="utf-8") as file:
                item = json.load(file)
            if isinstance(item, dict):
                yield official_item_to_legacy(item)
        return

    if labels_source.suffix.lower() == ".zip":
        with zipfile.ZipFile(labels_source) as archive:
            legacy_matches = [
                item
                for item in archive.infolist()
                if not item.is_dir()
                and Path(item.filename).name.lower()
                == f"bdd100k_labels_images_{split}.json"
            ]
            if legacy_matches:
                with archive.open(legacy_matches[0]) as binary_file:
                    with io.TextIOWrapper(binary_file, encoding="utf-8") as text_file:
                        temporary = Path(legacy_matches[0].filename)
                        data = json.load(text_file)
                if not isinstance(data, list):
                    raise ValueError(f"标签 JSON 顶层不是数组：{temporary}")
                for item in data:
                    if isinstance(item, dict):
                        yield item
                return

            members = [
                item
                for item in archive.infolist()
                if not item.is_dir()
                and Path(item.filename).suffix.lower() == ".json"
                and "100k" in {part.lower() for part in Path(item.filename).parts}
                and split in {part.lower() for part in Path(item.filename).parts}
            ]
            if not members:
                raise ValueError(f"{labels_source} 中未找到 100k/{split} JSON 标签")
            for member in sorted(members, key=lambda item: item.filename):
                with archive.open(member) as binary_file:
                    with io.TextIOWrapper(binary_file, encoding="utf-8") as text_file:
                        item = json.load(text_file)
                if isinstance(item, dict):
                    yield official_item_to_legacy(item)
        return

    with labels_source.open("r", encoding="utf-8") as file:
        first_character = next((character for character in file.read(4096) if not character.isspace()), "")
    if first_character == "[":
        yield from iter_json_array(labels_source)
    else:
        with labels_source.open("r", encoding="utf-8") as file:
            item = json.load(file)
        if not isinstance(item, dict):
            raise ValueError(f"标签 JSON 顶层不是对象或数组：{labels_source}")
        yield official_item_to_legacy(item)


def extract_official_archives(
    image_archives: list[Path], labels_zip: Path, raw_root: Path
) -> tuple[Path, Path, Path]:
    """Extract train/val images and locate labels in the official archive."""

    for images_zip in image_archives:
        if not images_zip.is_file():
            raise FileNotFoundError(f"BDD100K 图像压缩包不存在：{images_zip}")
    if not labels_zip.is_file():
        raise FileNotFoundError(f"官方标签压缩包不存在：{labels_zip}")

    images_root = raw_root / "bdd100k" / "images" / "100k"
    found_splits: set[str] = set()
    for images_zip in image_archives:
        with zipfile.ZipFile(images_zip) as archive:
            members: list[tuple[zipfile.ZipInfo, str, str]] = []
            for item in archive.infolist():
                if item.is_dir() or Path(item.filename).suffix.lower() not in {".jpg", ".jpeg"}:
                    continue
                parts = [part.lower() for part in Path(item.filename).parts]
                split = next((name for name in ("train", "val") if name in parts), None)
                if split:
                    members.append((item, split, Path(item.filename).name))
                    found_splits.add(split)

            if not members:
                raise ValueError(f"图像压缩包中未找到 train/val JPG：{images_zip}")

            for number, (item, split, filename) in enumerate(members, start=1):
                target = images_root / split / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and target.stat().st_size == item.file_size:
                    continue
                with archive.open(item) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                if number % 1000 == 0:
                    print(f"已解压图像：{number}/{len(members)} ({images_zip.name})")

    if found_splits != {"train", "val"}:
        raise ValueError(f"图像压缩包必须同时包含 train 和 val，实际找到：{sorted(found_splits)}")

    source_labels = raw_root / "bdd100k" / "labels" / "lane" / "source"
    source_labels.mkdir(parents=True, exist_ok=True)
    label_paths: dict[str, Path] = {}
    with zipfile.ZipFile(labels_zip) as archive:
        for split in ("train", "val"):
            try:
                item = find_label_member(archive, split)
            except ValueError:
                label_paths[split] = labels_zip
            else:
                target = source_labels / f"bdd100k_labels_images_{split}.json"
                if not target.exists() or target.stat().st_size != item.file_size:
                    with archive.open(item) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                label_paths[split] = target

    return images_root, label_paths["train"], label_paths["val"]


def locate_source_image(images_root: Path, split: str, filename: str) -> Path | None:
    split_path = images_root / split / filename
    if split_path.is_file():
        return split_path
    flat_path = images_root / filename
    if flat_path.is_file():
        return flat_path
    return None


def lane_only_item(item: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lane_labels = [
        label
        for label in item.get("labels", [])
        if isinstance(label, dict) and label.get("category") == "lane"
    ]
    filtered = {
        key: value
        for key, value in item.items()
        if key not in {"labels"}
    }
    filtered["labels"] = lane_labels
    return filtered, lane_labels


def interpolate_poly2d(vertices: list[Any], types: str, curve_steps: int = 24) -> list[tuple[float, float]]:
    """Flatten BDD100K cubic Bezier vertices using Scalabel's C/L convention."""

    points = [
        (float(point[0]), float(point[1]))
        for point in vertices
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    if len(points) < 2:
        return points
    if len(types) != len(points):
        types = "L" * len(points)

    result = [points[0]]
    index = 1
    while index < len(points):
        if types[index].upper() == "C" and index + 2 < len(points):
            start = result[-1]
            control_1, control_2, end = points[index : index + 3]
            for step in range(1, curve_steps + 1):
                t = step / curve_steps
                inverse = 1.0 - t
                x = (
                    inverse**3 * start[0]
                    + 3 * inverse**2 * t * control_1[0]
                    + 3 * inverse * t**2 * control_2[0]
                    + t**3 * end[0]
                )
                y = (
                    inverse**3 * start[1]
                    + 3 * inverse**2 * t * control_1[1]
                    + 3 * inverse * t**2 * control_2[1]
                    + t**3 * end[1]
                )
                result.append((x, y))
            index += 3
        else:
            result.append(points[index])
            index += 1
    return result


def draw_lane_mask(image_path: Path, lane_labels: list[dict[str, Any]], width: int) -> Image.Image:
    with Image.open(image_path) as image:
        image_size = image.size

    mask = Image.new("L", image_size, color=0)
    draw = ImageDraw.Draw(mask)
    for label in lane_labels:
        for shape in label.get("poly2d", []):
            vertices = shape.get("vertices", []) if isinstance(shape, dict) else []
            types = str(shape.get("types", "")) if isinstance(shape, dict) else ""
            points = interpolate_poly2d(vertices, types)
            if len(points) >= 2:
                draw.line(points, fill=1, width=width, joint="curve")
    return mask


def relative_csv_path(path: Path, csv_parent: Path) -> str:
    return Path(os.path.relpath(path, csv_parent)).as_posix()


def prepare_split(
    split: str,
    images_root: Path,
    labels_path: Path,
    raw_root: Path,
    output_root: Path,
    line_width: int,
) -> tuple[int, int]:
    raw_images = raw_root / "bdd100k" / "images" / "100k" / split
    raw_polygons = raw_root / "bdd100k" / "labels" / "lane" / "polygons"
    raw_polygons.mkdir(parents=True, exist_ok=True)
    filtered_labels_path = raw_polygons / f"lane_{split}.json"

    masks_dir = output_root / split / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{split}_pairs.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pair_count = 0
    lane_image_count = 0
    with (
        filtered_labels_path.open("w", encoding="utf-8", newline="\n") as filtered_file,
        csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file,
    ):
        filtered_file.write("[\n")
        first_item = True
        writer = csv.writer(csv_file)
        writer.writerow(("stem", "image_path", "mask_path"))

        for item in iter_label_items(labels_path, split):
            filename = str(item.get("name", ""))
            if not filename:
                continue
            source_image = locate_source_image(images_root, split, filename)
            if source_image is None:
                continue

            stem = Path(filename).stem
            target_image = raw_images / filename
            if source_image.resolve() != target_image.resolve():
                link_or_copy(source_image, target_image)

            filtered_item, lane_labels = lane_only_item(item)
            if lane_labels:
                lane_image_count += 1

            if not first_item:
                filtered_file.write(",\n")
            json.dump(filtered_item, filtered_file, ensure_ascii=False, separators=(",", ":"))
            first_item = False

            mask_path = masks_dir / f"{stem}.png"
            mask = draw_lane_mask(target_image, lane_labels, line_width)
            # Low PNG compression keeps preparation fast; masks remain lossless.
            mask.save(mask_path, compress_level=1)

            writer.writerow(
                (
                    stem,
                    relative_csv_path(target_image, csv_path.parent),
                    relative_csv_path(mask_path, csv_path.parent),
                )
            )
            pair_count += 1
            if pair_count % 500 == 0:
                print(f"{split}: 已处理 {pair_count} 对图像/mask")

        filtered_file.write("\n]\n")

    return pair_count, lane_image_count


def write_source_note(raw_root: Path, line_width: int) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    note = raw_root / "DATA_SOURCE.txt"
    note.write_text(
        "BDD100K lane-marking dataset\n"
        f"Official download page: {OFFICIAL_PAGE}\n"
        f"100K images: {OFFICIAL_IMAGES_URL}\n"
        f"Labels: {OFFICIAL_LABELS_URL}\n"
        "Labels SHA256: 7f1f9043c70a6ff0788a323cfb914aeced109a37b7150740d670a347c881394a\n"
        f"Train image mirror: {TRAIN_IMAGES_MIRROR_URL}\n"
        "Train image SHA256: 389dd8888ba7fb58ba479e9dc986bd62d1b796f9015b72f75bcb1262b2e34cc7\n"
        f"Validation image mirror: {VAL_IMAGES_MIRROR_URL}\n"
        "Validation image SHA256: bf54b557af0d91267b848a504e7d596d03345d8bd4e89c06f5b4aeb9d0beae30\n"
        "Classes: 0=background, 1=lane\n"
        f"Lane rasterization width at 1280x720: {line_width} px\n"
        "Only official lane polylines are rasterized. C vertices are cubic Bezier controls.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-zip", type=Path, default=Path(r"C:\Downloads\bdd100k_images_100k.zip")
    )
    parser.add_argument(
        "--train-images-zip", type=Path, default=Path(r"C:\Downloads\100k_images_train.zip")
    )
    parser.add_argument(
        "--val-images-zip", type=Path, default=Path(r"C:\Downloads\bdd100k_val_hirundo.zip")
    )
    parser.add_argument(
        "--labels-zip", type=Path, default=Path(r"C:\Downloads\bdd100k_labels.zip")
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        help="已解压的图像根目录，可为含 train/val 的目录或一个平铺 JPG 目录",
    )
    parser.add_argument("--train-labels", type=Path, help="已解压的 train JSON")
    parser.add_argument("--val-labels", type=Path, help="已解压的 val JSON")
    parser.add_argument("--raw-root", type=Path, default=DATASETS_DIR / "pc_lane_ds")
    parser.add_argument(
        "--output-root", type=Path, default=DATASETS_DIR / "processed_pc_lane_ds"
    )
    parser.add_argument("--line-width", type=int, default=8)
    args = parser.parse_args()

    extracted_values = (args.images_root, args.train_labels, args.val_labels)
    if any(extracted_values) and not all(extracted_values):
        parser.error("使用已解压数据时必须同时提供 --images-root、--train-labels、--val-labels")
    if args.line_width <= 0:
        parser.error("--line-width 必须大于 0")
    return args


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if args.images_root:
        images_root = args.images_root.expanduser().resolve()
        train_labels = args.train_labels.expanduser().resolve()
        val_labels = args.val_labels.expanduser().resolve()
        for path in (images_root, train_labels, val_labels):
            if not path.exists():
                raise FileNotFoundError(path)
    else:
        combined_images_zip = args.images_zip.expanduser().resolve()
        if combined_images_zip.is_file():
            image_archives = [combined_images_zip]
        else:
            image_archives = [
                args.train_images_zip.expanduser().resolve(),
                args.val_images_zip.expanduser().resolve(),
            ]
        images_root, train_labels, val_labels = extract_official_archives(
            image_archives,
            args.labels_zip.expanduser().resolve(),
            raw_root,
        )

    write_source_note(raw_root, args.line_width)
    total = 0
    for split, labels_path in (("train", train_labels), ("val", val_labels)):
        pairs, with_lanes = prepare_split(
            split,
            images_root,
            labels_path,
            raw_root,
            output_root,
            args.line_width,
        )
        total += pairs
        print(f"{split}: {pairs} 对，其中 {with_lanes} 张含车道线")

    if total == 0:
        raise RuntimeError("未找到任何图像/标签配对，请检查输入路径")
    print(f"完成：共 {total} 对，输出到 {output_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, zipfile.BadZipFile) as error:
        print(f"错误：{error}")
        raise SystemExit(1)

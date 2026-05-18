#!/usr/bin/env python3
"""Build a stratified multiclass ROI dataset from DIOR annotations.

This builder keeps the same core safety rule as the legacy binary builder:
splits are assigned at the source-image level before any ROI crop is written,
which prevents train/val/test leakage from the same original image.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "dataset_DIOR"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "dior_multiclass_stratified"

DEFAULT_TARGET_CLASSES = (
    "airplane",
    "ship",
    "vehicle",
    "bridge",
    "harbor",
    "storagetank",
)
OUTPUT_SIZE = (64, 64)
POSITIVE_EXPAND_RATIO = 1.4
RANDOM_SEED = 42
SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class Box:
    """Single horizontal bounding box."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return max(1, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(1, self.ymax - self.ymin)

    def as_csv(self) -> str:
        return f"{self.xmin},{self.ymin},{self.xmax},{self.ymax}"


@dataclass(frozen=True)
class ImageRecord:
    """Single source image with all target-class boxes attached."""

    image_id: str
    image_path: Path
    class_boxes: dict[str, tuple[Box, ...]]

    @property
    def total_instance_count(self) -> int:
        return sum(len(boxes) for boxes in self.class_boxes.values())

    def class_count(self, class_name: str) -> int:
        return len(self.class_boxes.get(class_name, ()))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build a DIOR multiclass ROI dataset with source-image stratified splits."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=f"Path to dataset_DIOR root (default: {DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        "--output-root",
        dest="output_root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output directory (default: {OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_TARGET_CLASSES),
        help="Target DIOR classes used to build the multiclass ROI dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for split assignment (default: {RANDOM_SEED})",
    )
    return parser.parse_args()


def normalize_target_classes(class_names: list[str]) -> list[str]:
    """Normalize, deduplicate, and validate requested target classes."""
    normalized: list[str] = []
    seen: set[str] = set()
    for class_name in class_names:
        normalized_name = str(class_name).strip().lower()
        if not normalized_name or normalized_name in seen:
            continue
        seen.add(normalized_name)
        normalized.append(normalized_name)
    if not normalized:
        raise ValueError("At least one target class is required.")
    return normalized


def parse_target_boxes(
    annotation_path: Path,
    target_classes: set[str],
) -> dict[str, list[Box]]:
    """Parse all target-class boxes from one DIOR XML annotation."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    class_boxes = {class_name: [] for class_name in target_classes}

    for obj in root.findall("object"):
        class_name = str(obj.findtext("name") or "").strip().lower()
        if class_name not in target_classes:
            continue
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = int(float((bbox.findtext("xmin") or "").strip()))
            ymin = int(float((bbox.findtext("ymin") or "").strip()))
            xmax = int(float((bbox.findtext("xmax") or "").strip()))
            ymax = int(float((bbox.findtext("ymax") or "").strip()))
        except ValueError:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        class_boxes[class_name].append(Box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    return {class_name: boxes for class_name, boxes in class_boxes.items() if boxes}


def resolve_image_path(image_id: str, dataset_root: Path) -> Path | None:
    """Resolve one DIOR image path from trainval/test directories."""
    trainval_path = dataset_root / "JPEGImages-trainval" / f"{image_id}.jpg"
    test_path = dataset_root / "JPEGImages-test" / f"{image_id}.jpg"
    if trainval_path.exists():
        return trainval_path
    if test_path.exists():
        return test_path
    return None


def discover_positive_source_images(
    dataset_root: Path,
    target_classes: list[str],
) -> tuple[list[ImageRecord], dict[str, object]]:
    """Collect source images that contain at least one target-class instance."""
    annotation_dir = dataset_root / "Annotations" / "Horizontal Bounding Boxes"
    records: list[ImageRecord] = []
    per_class_instances = {class_name: 0 for class_name in target_classes}
    missing_images = 0

    for annotation_path in sorted(annotation_dir.glob("*.xml")):
        class_boxes = parse_target_boxes(annotation_path, set(target_classes))
        if not class_boxes:
            continue

        image_id = annotation_path.stem
        image_path = resolve_image_path(image_id, dataset_root)
        if image_path is None:
            print(
                f"Warning: missing image jpg for source image {image_id}",
                file=sys.stderr,
            )
            missing_images += 1
            continue

        for class_name, boxes in class_boxes.items():
            per_class_instances[class_name] += len(boxes)
        records.append(
            ImageRecord(
                image_id=image_id,
                image_path=image_path,
                class_boxes={
                    class_name: tuple(boxes)
                    for class_name, boxes in class_boxes.items()
                },
            )
        )

    scan_stats: dict[str, object] = {
        "positive_source_images": len(records),
        "processed_images": len(records),
        "missing_images": missing_images,
        "per_class_instances": per_class_instances,
    }
    return records, scan_stats


def compute_integer_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Convert split ratios into integer split quotas."""
    raw_targets = {split: total * ratios[split] for split in SPLIT_ORDER}
    integer_targets = {split: int(math.floor(raw_targets[split])) for split in SPLIT_ORDER}
    remainder = total - sum(integer_targets.values())
    fractional_parts = sorted(
        ((raw_targets[split] - integer_targets[split], split) for split in SPLIT_ORDER),
        reverse=True,
    )
    for _, split in fractional_parts[:remainder]:
        integer_targets[split] += 1
    return integer_targets


def choose_split_for_record(
    record: ImageRecord,
    target_classes: list[str],
    current_image_counts: dict[str, int],
    current_class_counts: dict[str, dict[str, int]],
    target_image_counts: dict[str, int],
    target_class_counts: dict[str, dict[str, float]],
    rng: random.Random,
) -> str:
    """Choose the best split for one source image."""
    available_splits = [
        split
        for split in SPLIT_ORDER
        if current_image_counts[split] < target_image_counts[split]
    ]
    if not available_splits:
        available_splits = list(SPLIT_ORDER)

    best_split = available_splits[0]
    best_score: float | None = None

    for split in available_splits:
        remaining_image_slots = max(1, target_image_counts[split] - current_image_counts[split])
        score = 0.0
        for class_name in target_classes:
            remaining_instances = (
                target_class_counts[split][class_name]
                - current_class_counts[split][class_name]
            )
            desired_instances_per_image = remaining_instances / remaining_image_slots
            score += abs(record.class_count(class_name) - desired_instances_per_image)
        score += rng.random() * 1e-8

        if best_score is None or score < best_score:
            best_score = score
            best_split = split

    return best_split


def assign_records_to_splits(
    records: list[ImageRecord],
    target_classes: list[str],
    seed: int,
) -> dict[str, list[ImageRecord]]:
    """Assign source images to train/val/test with rough class-balance preservation."""
    rng = random.Random(seed)
    ordered_records = records[:]
    rng.shuffle(ordered_records)
    ordered_records.sort(key=lambda record: record.total_instance_count, reverse=True)

    target_image_counts = compute_integer_targets(len(ordered_records), SPLIT_RATIOS)
    total_class_counts = {
        class_name: sum(record.class_count(class_name) for record in ordered_records)
        for class_name in target_classes
    }
    target_class_counts = {
        split: {
            class_name: total_class_counts[class_name] * SPLIT_RATIOS[split]
            for class_name in target_classes
        }
        for split in SPLIT_ORDER
    }
    current_image_counts = {split: 0 for split in SPLIT_ORDER}
    current_class_counts = {
        split: {class_name: 0 for class_name in target_classes}
        for split in SPLIT_ORDER
    }
    assignments = {split: [] for split in SPLIT_ORDER}

    for record in ordered_records:
        split = choose_split_for_record(
            record=record,
            target_classes=target_classes,
            current_image_counts=current_image_counts,
            current_class_counts=current_class_counts,
            target_image_counts=target_image_counts,
            target_class_counts=target_class_counts,
            rng=rng,
        )
        assignments[split].append(record)
        current_image_counts[split] += 1
        for class_name in target_classes:
            current_class_counts[split][class_name] += record.class_count(class_name)

    return assignments


def ensure_output_root(output_root: Path, target_classes: list[str]) -> None:
    """Create the output directory tree and refuse accidental overwrite."""
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists, refusing to overwrite: {output_root}")
    for split in SPLIT_ORDER:
        for class_name in target_classes:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)


def clip_box(box: Box, image_width: int, image_height: int) -> Box | None:
    """Clip a box to image bounds."""
    xmin = max(0, min(box.xmin, image_width - 1))
    ymin = max(0, min(box.ymin, image_height - 1))
    xmax = max(1, min(box.xmax, image_width))
    ymax = max(1, min(box.ymax, image_height))
    if xmax <= xmin or ymax <= ymin:
        return None
    return Box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def expand_box(box: Box, image_width: int, image_height: int) -> Box | None:
    """Expand one positive box by a fixed ratio."""
    center_x = (box.xmin + box.xmax) / 2.0
    center_y = (box.ymin + box.ymax) / 2.0
    expanded_w = max(1.0, box.width * POSITIVE_EXPAND_RATIO)
    expanded_h = max(1.0, box.height * POSITIVE_EXPAND_RATIO)
    xmin = int(round(center_x - expanded_w / 2.0))
    ymin = int(round(center_y - expanded_h / 2.0))
    xmax = int(round(center_x + expanded_w / 2.0))
    ymax = int(round(center_y + expanded_h / 2.0))
    return clip_box(Box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax), image_width, image_height)


def crop_and_resize(image, box: Box):
    """Crop one ROI and resize it to the canonical training size."""
    crop = image[box.ymin:box.ymax, box.xmin:box.xmax]
    if crop.size == 0:
        return None
    return cv2.resize(crop, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)


def save_image(image, output_path: Path) -> None:
    """Save one ROI image to disk."""
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to save image: {output_path}")


def make_output_filename(split: str, class_name: str, image_id: str, index: int) -> str:
    """Build a deterministic ROI filename."""
    return f"{split}_{class_name}_{image_id}_{index:05d}.jpg"


def build_dataset(
    assignments: dict[str, list[ImageRecord]],
    target_classes: list[str],
    output_root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, int | str]], dict[str, set[str]]]:
    """Materialize cropped ROI images for the split assignments."""
    metadata_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, int | str]] = []
    source_image_sets = {split: set() for split in SPLIT_ORDER}

    for split in SPLIT_ORDER:
        records = assignments[split]
        class_counters = {class_name: 0 for class_name in target_classes}

        for record in records:
            source_image_sets[split].add(record.image_id)
            image = cv2.imread(str(record.image_path))
            if image is None:
                print(f"Warning: failed to read image: {record.image_path}", file=sys.stderr)
                continue
            image_height, image_width = image.shape[:2]

            for class_name in target_classes:
                for box in record.class_boxes.get(class_name, ()):
                    expanded_box = expand_box(box, image_width, image_height)
                    if expanded_box is None:
                        continue
                    roi = crop_and_resize(image, expanded_box)
                    if roi is None:
                        continue

                    file_index = class_counters[class_name]
                    filename = make_output_filename(split, class_name, record.image_id, file_index)
                    output_path = output_root / split / class_name / filename
                    save_image(roi, output_path)
                    metadata_rows.append(
                        {
                            "filename": filename,
                            "split": split,
                            "label": class_name,
                            "source_image": f"{record.image_id}.jpg",
                            "bbox": expanded_box.as_csv(),
                            "crop_type": class_name,
                        }
                    )
                    class_counters[class_name] += 1

        summary_row: dict[str, int | str] = {
            "split": split,
            "source_image_count": len(source_image_sets[split]),
        }
        for class_name in target_classes:
            summary_row[f"{class_name}_count"] = class_counters[class_name]
        summary_rows.append(summary_row)

    return metadata_rows, summary_rows, source_image_sets


def write_metadata_csv(output_root: Path, metadata_rows: list[dict[str, str]]) -> Path:
    """Write ROI-level metadata.csv."""
    metadata_path = output_root / "metadata.csv"
    fieldnames = ["filename", "split", "label", "source_image", "bbox", "crop_type"]
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
    return metadata_path


def write_split_summary_csv(
    output_root: Path,
    summary_rows: list[dict[str, int | str]],
    target_classes: list[str],
) -> Path:
    """Write split-level summary counts with dynamic class columns."""
    summary_path = output_root / "split_summary.csv"
    fieldnames = ["split", "source_image_count"] + [f"{class_name}_count" for class_name in target_classes]
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path


def write_label_map_json(output_root: Path, target_classes: list[str]) -> Path:
    """Write the canonical label order used by the training scripts."""
    label_map_path = output_root / "label_map.json"
    label_map = {class_name: index for index, class_name in enumerate(target_classes)}
    with label_map_path.open("w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=2, ensure_ascii=False)
    return label_map_path


def validate_required_paths(dataset_root: Path) -> None:
    """Validate the basic DIOR directory layout required by the builder."""
    required_dirs = [
        dataset_root,
        dataset_root / "Annotations" / "Horizontal Bounding Boxes",
        dataset_root / "JPEGImages-trainval",
        dataset_root / "JPEGImages-test",
    ]
    for path in required_dirs:
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Required directory not found: {path}")


def validate_no_overlap(source_image_sets: dict[str, set[str]]) -> None:
    """Ensure the split assignments do not reuse one source image across splits."""
    for left_index, left_split in enumerate(SPLIT_ORDER):
        for right_split in SPLIT_ORDER[left_index + 1 :]:
            overlap = source_image_sets[left_split] & source_image_sets[right_split]
            if overlap:
                raise RuntimeError(
                    f"Source image leakage detected between {left_split} and {right_split}: "
                    f"{sorted(list(overlap))[:5]}"
                )


def count_output_images(output_root: Path, target_classes: list[str]) -> int:
    """Count all saved ROI JPEGs in the output directory."""
    total = 0
    for split in SPLIT_ORDER:
        for class_name in target_classes:
            total += len(list((output_root / split / class_name).glob("*.jpg")))
    return total


def print_summary(
    target_classes: list[str],
    scan_stats: dict[str, object],
    summary_rows: list[dict[str, int | str]],
    metadata_path: Path,
    split_summary_path: Path,
    label_map_path: Path,
    output_root: Path,
) -> None:
    """Print the final builder summary."""
    print("DIOR multiclass ROI dataset build finished.")
    print(f"Output root: {output_root}")
    print(f"Metadata: {metadata_path}")
    print(f"Split summary: {split_summary_path}")
    print(f"Label map: {label_map_path}")
    print()
    print(f"Target classes: {', '.join(target_classes)}")
    print(
        f"Positive source images: {scan_stats['positive_source_images']}, "
        f"missing images: {scan_stats['missing_images']}"
    )
    print(f"Per-class instances: {scan_stats['per_class_instances']}")
    print()
    for row in summary_rows:
        counts = ", ".join(
            f"{class_name}={row[f'{class_name}_count']}"
            for class_name in target_classes
        )
        print(
            f"[{row['split']}] {counts}, source_images={row['source_image_count']}"
        )
    print()
    print("Validation checks passed:")
    print("- source_image sets are disjoint across train/val/test")
    print("- metadata row count matches total number of saved ROI images")


def main() -> int:
    """Script entry point."""
    args = parse_args()
    try:
        target_classes = normalize_target_classes(args.classes)
        validate_required_paths(args.dataset_root)
        ensure_output_root(args.output_root, target_classes)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    records, scan_stats = discover_positive_source_images(args.dataset_root, target_classes)
    if not records:
        print("Error: no source images were found for the requested target classes.", file=sys.stderr)
        return 1

    assignments = assign_records_to_splits(records, target_classes, seed=args.seed)
    metadata_rows, summary_rows, source_image_sets = build_dataset(
        assignments=assignments,
        target_classes=target_classes,
        output_root=args.output_root,
    )
    metadata_path = write_metadata_csv(args.output_root, metadata_rows)
    split_summary_path = write_split_summary_csv(args.output_root, summary_rows, target_classes)
    label_map_path = write_label_map_json(args.output_root, target_classes)

    try:
        validate_no_overlap(source_image_sets)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    total_saved_images = count_output_images(args.output_root, target_classes)
    if total_saved_images != len(metadata_rows):
        print(
            "Error: metadata row count does not match number of saved ROI images "
            f"({len(metadata_rows)} vs {total_saved_images})",
            file=sys.stderr,
        )
        return 1

    print_summary(
        target_classes=target_classes,
        scan_stats=scan_stats,
        summary_rows=summary_rows,
        metadata_path=metadata_path,
        split_summary_path=split_summary_path,
        label_map_path=label_map_path,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""ROI dataset utilities for DIOR low-shot multiclass experiments.

The processed ROI dataset is stored as cropped JPEG files organized by
``split/class_name/*.jpg``. This module is used by the current paper workflow
for multiclass remote-sensing ROI classification.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2

    CV2_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    cv2 = None  # type: ignore[assignment]
    CV2_IMPORT_ERROR = exc

try:
    import numpy as np

    NUMPY_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    np = None  # type: ignore[assignment]
    NUMPY_IMPORT_ERROR = exc

try:
    import torch
    from torch.utils.data import Dataset

    TORCH_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    TORCH_IMPORT_ERROR = exc


LABEL_MAP = {"background": 0, "airplane": 1}
SPLIT_NAMES = ("train", "val", "test")
SUPPORTED_AUGMENT_PRESETS = ("none", "train_light")
SUPPORTED_NORMALIZATION_PRESETS = ("none", "imagenet")
BUILD_COMMAND_HINT = (
    ".venv/bin/python scripts/build_dior_multiclass_stratified.py"
)
LABEL_MAP_FILENAME = "label_map.json"


@dataclass(frozen=True)
class ROISample:
    """Single ROI image record."""

    path: Path
    label: int
    split: str
    class_name: str


def _legacy_binary_order(class_names: set[str]) -> list[str] | None:
    """Preserve the historical binary label order when possible."""
    if class_names == {"background", "airplane"}:
        return ["background", "airplane"]
    return None


def resolve_label_map(dataset_root: Path) -> dict[str, int]:
    """Resolve the class-to-index mapping for one processed ROI dataset."""
    label_map_path = dataset_root / LABEL_MAP_FILENAME
    if label_map_path.exists() and label_map_path.is_file():
        with label_map_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError(f"Invalid label map JSON: {label_map_path}")
        normalized = {str(key): int(value) for key, value in payload.items()}
        ordered_pairs = sorted(normalized.items(), key=lambda item: item[1])
        for expected_index, (_class_name, index) in enumerate(ordered_pairs):
            if index != expected_index:
                raise RuntimeError(
                    f"Label map indices must be contiguous and zero-based: {label_map_path}"
                )
        return dict(ordered_pairs)

    discovered_classes: set[str] = set()
    for split in SPLIT_NAMES:
        split_dir = dataset_root / split
        if not split_dir.exists() or not split_dir.is_dir():
            continue
        for child in split_dir.iterdir():
            if child.is_dir():
                discovered_classes.add(child.name)

    if not discovered_classes:
        raise RuntimeError(
            f"No class directories were found under processed dataset root: {dataset_root}"
        )

    legacy_order = _legacy_binary_order(discovered_classes)
    if legacy_order is not None:
        return {class_name: index for index, class_name in enumerate(legacy_order)}

    ordered_classes = sorted(discovered_classes)
    return {class_name: index for index, class_name in enumerate(ordered_classes)}


def resolve_class_names(dataset_root: Path) -> list[str]:
    """Return dataset class names ordered by their integer labels."""
    return list(resolve_label_map(dataset_root).keys())


def get_num_classes(dataset_root: Path) -> int:
    """Return the number of semantic classes in the processed dataset."""
    return len(resolve_label_map(dataset_root))


def resolve_split_sample_caps(
    *,
    max_samples_per_class: int | None = None,
    split_max_samples_per_class: dict[str, int | None] | None = None,
) -> dict[str, int | None]:
    """Resolve per-split sample caps with backward-compatible defaults."""
    resolved = {
        split: None if max_samples_per_class is None else int(max_samples_per_class)
        for split in SPLIT_NAMES
    }
    if split_max_samples_per_class is not None:
        for split, value in split_max_samples_per_class.items():
            if split not in SPLIT_NAMES:
                raise ValueError(
                    f"Unsupported split in split_max_samples_per_class: {split!r}"
                )
            resolved[split] = None if value is None else int(value)

    for split, value in resolved.items():
        if value is not None and value < 1:
            raise ValueError(
                f"{split} max_samples_per_class must be at least 1 when provided; got {value}."
            )
    return resolved


def ensure_processed_dataset(dataset_root: Path) -> None:
    """Validate that the processed ROI dataset exists and is structurally usable."""
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(
            "Processed ROI dataset not found. "
            f"Please run {BUILD_COMMAND_HINT} first."
        )

    label_map = resolve_label_map(dataset_root)
    for split in SPLIT_NAMES:
        split_dir = dataset_root / split
        if not split_dir.exists() or not split_dir.is_dir():
            raise FileNotFoundError(
                f"Missing processed split directory: {split_dir}. Run {BUILD_COMMAND_HINT} first."
            )
        for class_name in label_map:
            class_dir = split_dir / class_name
            if not class_dir.exists() or not class_dir.is_dir():
                raise FileNotFoundError(
                    "Missing processed split directory: "
                    f"{class_dir}. Run {BUILD_COMMAND_HINT} first."
                )


def _parse_split_summary_counts(
    split_summary_path: Path,
    class_names: list[str],
) -> dict[str, dict[str, int]] | None:
    """Parse split summary rows with dynamic ``<class>_count`` columns."""
    if not split_summary_path.exists() or not split_summary_path.is_file():
        return None

    summary_counts = {
        split: {class_name: 0 for class_name in class_names}
        for split in SPLIT_NAMES
    }
    for split in SPLIT_NAMES:
        summary_counts[split]["source_image_count"] = 0

    with split_summary_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            split = str(row.get("split", "")).strip()
            if split not in summary_counts:
                continue
            for class_name in class_names:
                column_name = f"{class_name}_count"
                summary_counts[split][class_name] = int(row.get(column_name, 0) or 0)
            summary_counts[split]["source_image_count"] = int(
                row.get("source_image_count", 0) or 0
            )
    return summary_counts


def collect_dataset_health_report(
    dataset_root: Path,
    max_samples_per_class: int | None = None,
    split_max_samples_per_class: dict[str, int | None] | None = None,
) -> dict[str, object]:
    """Collect a detailed filesystem-level health report for the processed dataset."""
    errors: list[str] = []
    warnings: list[str] = []
    dataset_exists = dataset_root.exists() and dataset_root.is_dir()
    resolved_split_caps = resolve_split_sample_caps(
        max_samples_per_class=max_samples_per_class,
        split_max_samples_per_class=split_max_samples_per_class,
    )

    if not dataset_exists:
        return {
            "dataset_root": str(dataset_root),
            "task_label_definition": None,
            "class_names": [],
            "label_map": {},
            "build_command_hint": BUILD_COMMAND_HINT,
            "dataset_root_exists": False,
            "directory_status": {},
            "counts_actual": {},
            "counts_effective": {},
            "total_actual": 0,
            "total_effective": 0,
            "max_samples_per_class": max_samples_per_class,
            "split_max_samples_per_class": resolved_split_caps,
            "metadata_csv": {
                "path": str(dataset_root / "metadata.csv"),
                "exists": False,
                "row_count": 0,
                "counts": None,
            },
            "split_summary_csv": {
                "path": str(dataset_root / "split_summary.csv"),
                "exists": False,
                "counts": None,
            },
            "errors": [
                "Processed ROI dataset root is missing: "
                f"{dataset_root}. Run {BUILD_COMMAND_HINT} first."
            ],
            "warnings": [],
        }

    label_map = resolve_label_map(dataset_root)
    class_names = list(label_map.keys())
    counts_actual = {
        split: {class_name: 0 for class_name in class_names}
        for split in SPLIT_NAMES
    }
    counts_effective = {
        split: {class_name: 0 for class_name in class_names}
        for split in SPLIT_NAMES
    }
    directory_status = {
        split: {class_name: False for class_name in class_names}
        for split in SPLIT_NAMES
    }

    for split in SPLIT_NAMES:
        split_dir = dataset_root / split
        if not split_dir.exists() or not split_dir.is_dir():
            errors.append(f"Missing processed split directory: {split_dir}.")
            continue
        for class_name in class_names:
            class_dir = split_dir / class_name
            exists = class_dir.exists() and class_dir.is_dir()
            directory_status[split][class_name] = exists
            if not exists:
                errors.append(
                    f"Missing processed split directory for {split}/{class_name}: {class_dir}. "
                    f"Run {BUILD_COMMAND_HINT} first."
                )
                continue
            count = len(list(class_dir.glob("*.jpg")))
            counts_actual[split][class_name] = count
            split_cap = resolved_split_caps[split]
            counts_effective[split][class_name] = (
                count
                if split_cap is None
                else min(count, int(split_cap))
            )

    metadata_path = dataset_root / "metadata.csv"
    split_summary_path = dataset_root / "split_summary.csv"
    metadata_exists = metadata_path.exists() and metadata_path.is_file()
    split_summary_exists = split_summary_path.exists() and split_summary_path.is_file()
    metadata_counts: dict[str, dict[str, int]] | None = None
    metadata_row_count = 0

    if metadata_exists:
        metadata_counts = {
            split: {class_name: 0 for class_name in class_names}
            for split in SPLIT_NAMES
        }
        with metadata_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                split = str(row.get("split", "")).strip()
                label = str(row.get("label", "")).strip()
                if split not in metadata_counts or label not in label_map:
                    continue
                metadata_counts[split][label] += 1
                metadata_row_count += 1

        for split in SPLIT_NAMES:
            for class_name in class_names:
                if metadata_counts[split][class_name] != counts_actual[split][class_name]:
                    errors.append(
                        "metadata.csv count mismatch for "
                        f"{split}/{class_name}: files={counts_actual[split][class_name]}, "
                        f"metadata_rows={metadata_counts[split][class_name]}."
                    )
    else:
        warnings.append(f"metadata.csv not found: {metadata_path}")

    split_summary_counts = _parse_split_summary_counts(split_summary_path, class_names)
    if split_summary_counts is None:
        warnings.append(f"split_summary.csv not found: {split_summary_path}")
    else:
        for split in SPLIT_NAMES:
            for class_name in class_names:
                if split_summary_counts[split][class_name] != counts_actual[split][class_name]:
                    errors.append(
                        "split_summary.csv count mismatch for "
                        f"{split}/{class_name}: files={counts_actual[split][class_name]}, "
                        f"summary={split_summary_counts[split][class_name]}."
                    )

    total_actual = sum(
        counts_actual[split][class_name]
        for split in SPLIT_NAMES
        for class_name in class_names
    )
    total_effective = sum(
        counts_effective[split][class_name]
        for split in SPLIT_NAMES
        for class_name in class_names
    )

    return {
        "dataset_root": str(dataset_root),
        "task_label_definition": ", ".join(class_names),
        "class_names": class_names,
        "label_map": dict(label_map),
        "build_command_hint": BUILD_COMMAND_HINT,
        "dataset_root_exists": dataset_exists,
        "directory_status": directory_status,
        "counts_actual": counts_actual,
        "counts_effective": counts_effective,
        "total_actual": total_actual,
        "total_effective": total_effective,
        "max_samples_per_class": max_samples_per_class,
        "split_max_samples_per_class": resolved_split_caps,
        "metadata_csv": {
            "path": str(metadata_path),
            "exists": metadata_exists,
            "row_count": metadata_row_count,
            "counts": metadata_counts,
        },
        "split_summary_csv": {
            "path": str(split_summary_path),
            "exists": split_summary_exists,
            "counts": split_summary_counts,
        },
        "errors": errors,
        "warnings": warnings,
    }


def validate_processed_dataset_nonempty(
    dataset_root: Path,
    max_samples_per_class: int | None = None,
    split_max_samples_per_class: dict[str, int | None] | None = None,
) -> dict[str, object]:
    """Validate that every split/class has non-zero effective samples."""
    ensure_processed_dataset(dataset_root)
    report = collect_dataset_health_report(
        dataset_root=dataset_root,
        max_samples_per_class=max_samples_per_class,
        split_max_samples_per_class=split_max_samples_per_class,
    )

    errors = list(report["errors"])
    counts_actual = report["counts_actual"]
    counts_effective = report["counts_effective"]
    class_names = list(report["class_names"])

    for split in SPLIT_NAMES:
        for class_name in class_names:
            actual = int(counts_actual[split][class_name])  # type: ignore[index]
            effective = int(counts_effective[split][class_name])  # type: ignore[index]
            if effective <= 0:
                errors.append(
                    f"Processed dataset has no usable samples for {split}/{class_name}: "
                    f"actual_count={actual}, effective_count={effective}. "
                    f"Run {BUILD_COMMAND_HINT} first."
                )

    if errors:
        raise RuntimeError(" | ".join(errors))

    return report


def select_class_paths(
    class_paths: list[Path],
    split: str,
    class_name: str,
    max_samples_per_class: int | None,
    seed: int,
) -> list[Path]:
    """Optionally subsample a class with a deterministic seed."""
    if max_samples_per_class is None or len(class_paths) <= max_samples_per_class:
        return class_paths

    rng = random.Random(f"{seed}:{split}:{class_name}")
    chosen_paths = list(class_paths)
    rng.shuffle(chosen_paths)
    return sorted(chosen_paths[:max_samples_per_class])


def collect_roi_samples(
    dataset_root: Path,
    split: str,
    max_samples_per_class: int | None = None,
    seed: int = 42,
) -> list[ROISample]:
    """Collect ROI sample metadata for a given split."""
    ensure_processed_dataset(dataset_root)
    label_map = resolve_label_map(dataset_root)
    samples: list[ROISample] = []

    for class_name, label in label_map.items():
        class_dir = dataset_root / split / class_name
        class_paths = sorted(class_dir.glob("*.jpg"))
        class_paths = select_class_paths(
            class_paths=class_paths,
            split=split,
            class_name=class_name,
            max_samples_per_class=max_samples_per_class,
            seed=seed,
        )
        for path in class_paths:
            samples.append(
                ROISample(
                    path=path,
                    label=label,
                    split=split,
                    class_name=class_name,
                )
            )

    return samples


def count_samples_by_class(
    samples: list[ROISample],
    class_names: list[str] | None = None,
) -> dict[str, int]:
    """Count samples per semantic class."""
    ordered_class_names = (
        list(class_names)
        if class_names is not None
        else sorted({sample.class_name for sample in samples})
    )
    counts = {class_name: 0 for class_name in ordered_class_names}
    for sample in samples:
        counts[sample.class_name] = counts.get(sample.class_name, 0) + 1
    return counts


class ROIDataset(Dataset):
    """PyTorch dataset wrapper for generic ROI multiclass classification."""

    def __init__(
        self,
        samples: list[ROISample],
        image_size: int = 64,
        grayscale: bool = False,
        augment: str = "none",
        normalization: str = "none",
    ) -> None:
        if TORCH_IMPORT_ERROR is not None:
            raise ImportError(
                "PyTorch is required for ROIDataset. Please install torch first."
            ) from TORCH_IMPORT_ERROR

        self.samples = list(samples)
        self.image_size = int(image_size)
        self.grayscale = bool(grayscale)
        self.augment = normalize_augment_preset(augment)
        self.normalization = normalize_input_preset(normalization)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if CV2_IMPORT_ERROR is not None:
            raise ImportError(
                "opencv-python is required for ROIDataset image loading."
            ) from CV2_IMPORT_ERROR
        if NUMPY_IMPORT_ERROR is not None:
            raise ImportError(
                "numpy is required for ROIDataset image loading."
            ) from NUMPY_IMPORT_ERROR

        sample = self.samples[index]
        if self.grayscale:
            image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Failed to read ROI image: {sample.path}")
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
            image = image[:, :, None]
        else:
            image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read ROI image: {sample.path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
            if self.augment == "train_light":
                image = apply_train_light_augmentation(image)

        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
        if self.normalization == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=image_tensor.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=image_tensor.dtype).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        label_tensor = torch.tensor(sample.label, dtype=torch.long)
        return image_tensor, label_tensor


def normalize_augment_preset(augment: str | None) -> str:
    """Normalize and validate an augmentation preset string."""
    preset = "none" if augment is None else str(augment).strip().lower()
    if preset not in SUPPORTED_AUGMENT_PRESETS:
        raise ValueError(
            "Unsupported ROIDataset augment preset: "
            f"{augment!r}. Supported presets: {', '.join(SUPPORTED_AUGMENT_PRESETS)}."
        )
    return preset


def normalize_input_preset(normalization: str | None) -> str:
    """Normalize and validate an input normalization preset string."""
    preset = "none" if normalization is None else str(normalization).strip().lower()
    if preset not in SUPPORTED_NORMALIZATION_PRESETS:
        raise ValueError(
            "Unsupported ROIDataset normalization preset: "
            f"{normalization!r}. Supported presets: {', '.join(SUPPORTED_NORMALIZATION_PRESETS)}."
        )
    return preset


def apply_train_light_augmentation(image: np.ndarray) -> np.ndarray:
    """Apply conservative augmentation suited to ROI airplane recognition."""
    if CV2_IMPORT_ERROR is not None:
        raise ImportError("opencv-python is required for ROIDataset augmentation.") from CV2_IMPORT_ERROR
    if NUMPY_IMPORT_ERROR is not None:
        raise ImportError("numpy is required for ROIDataset augmentation.") from NUMPY_IMPORT_ERROR

    image = apply_random_resized_crop(image, scale_range=(0.92, 1.0), ratio_range=(0.95, 1.05))

    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1, :])

    angle = random.uniform(-8.0, 8.0)
    image = rotate_image(image, angle)
    image = apply_color_jitter(
        image,
        brightness=0.08,
        contrast=0.10,
        saturation=0.08,
    )
    return image


def apply_random_resized_crop(
    image: np.ndarray,
    scale_range: tuple[float, float],
    ratio_range: tuple[float, float],
) -> np.ndarray:
    """Crop a mild random window and resize back to the original square size."""
    height, width = image.shape[:2]
    area = float(height * width)

    for _ in range(8):
        target_area = random.uniform(*scale_range) * area
        aspect_ratio = random.uniform(*ratio_range)
        crop_width = int(round(np.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(np.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            x1 = random.randint(0, width - crop_width)
            y1 = random.randint(0, height - crop_height)
            cropped = image[y1 : y1 + crop_height, x1 : x1 + crop_width]
            return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    return image


def rotate_image(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate an image with reflected borders to avoid introducing hard edges."""
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def apply_color_jitter(
    image: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
) -> np.ndarray:
    """Apply mild brightness, contrast, and saturation jitter in RGB space."""
    image_float = image.astype(np.float32) / 255.0

    brightness_factor = 1.0 + random.uniform(-brightness, brightness)
    image_float = np.clip(image_float * brightness_factor, 0.0, 1.0)

    contrast_factor = 1.0 + random.uniform(-contrast, contrast)
    mean = np.mean(image_float, axis=(0, 1), keepdims=True)
    image_float = np.clip((image_float - mean) * contrast_factor + mean, 0.0, 1.0)

    hsv = cv2.cvtColor((image_float * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation_factor = 1.0 + random.uniform(-saturation, saturation)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_factor, 0.0, 255.0)
    image_float = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    return (np.clip(image_float, 0.0, 1.0) * 255.0).astype(np.uint8)

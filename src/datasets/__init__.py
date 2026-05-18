"""Dataset helpers for ROI classification.

This subpackage exposes the processed DIOR ROI dataset abstraction used by both
training and inference. The goal is to keep sample collection, deterministic
subsampling, and tensor loading rules consistent across all entry scripts.
"""

from .roi_dataset import (
    BUILD_COMMAND_HINT,
    LABEL_MAP_FILENAME,
    LABEL_MAP,
    ROIDataset,
    SUPPORTED_AUGMENT_PRESETS,
    SPLIT_NAMES,
    collect_dataset_health_report,
    collect_roi_samples,
    count_samples_by_class,
    ensure_processed_dataset,
    get_num_classes,
    resolve_class_names,
    resolve_label_map,
    resolve_split_sample_caps,
    validate_processed_dataset_nonempty,
)

__all__ = [
    "BUILD_COMMAND_HINT",
    "LABEL_MAP_FILENAME",
    "LABEL_MAP",
    "ROIDataset",
    "SPLIT_NAMES",
    "SUPPORTED_AUGMENT_PRESETS",
    "collect_dataset_health_report",
    "collect_roi_samples",
    "count_samples_by_class",
    "ensure_processed_dataset",
    "get_num_classes",
    "resolve_class_names",
    "resolve_label_map",
    "resolve_split_sample_caps",
    "validate_processed_dataset_nonempty",
]

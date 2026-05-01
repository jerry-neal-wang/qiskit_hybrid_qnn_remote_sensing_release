"""Shared utilities for DIOR low-shot experiments.

This package groups small but heavily reused helpers: reproducibility control,
metrics, JSON/report writing, and environment capture. Centralizing them avoids
drift between scripts that would otherwise each implement slightly different
artifact formats.
"""

from .experiment_artifacts import (
    build_runtime_payload,
    collect_environment_info,
    load_json,
    namespace_to_dict,
    plot_training_curves,
    save_json,
    summarize_metric_series,
)
from .metrics import compute_classification_metrics, save_confusion_matrix_figure
from .seed import set_seed
from .training_common import (
    build_class_weights,
    build_roi_dataloaders,
    build_scheduler,
    collect_split_outputs,
    compute_metrics_from_outputs,
    current_learning_rate,
    model_selection_key,
    resolve_device,
    select_best_threshold,
    trainable_parameter_names,
)

__all__ = [
    "build_runtime_payload",
    "collect_environment_info",
    "compute_classification_metrics",
    "compute_metrics_from_outputs",
    "current_learning_rate",
    "load_json",
    "model_selection_key",
    "namespace_to_dict",
    "plot_training_curves",
    "build_class_weights",
    "build_roi_dataloaders",
    "build_scheduler",
    "collect_split_outputs",
    "resolve_device",
    "save_confusion_matrix_figure",
    "save_json",
    "set_seed",
    "select_best_threshold",
    "summarize_metric_series",
    "trainable_parameter_names",
]

"""Shared training helpers for recognition mainline experiments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    from sklearn.metrics import accuracy_score, f1_score
    import torch
    from torch.utils.data import DataLoader

    from src.datasets.roi_dataset import (
        ROIDataset,
        collect_roi_samples,
        count_samples_by_class,
        resolve_label_map,
        resolve_split_sample_caps,
    )

    IMPORT_ERROR: Exception | None = None
except Exception as exc:
    accuracy_score = None  # type: ignore[assignment]
    f1_score = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    ROIDataset = None  # type: ignore[assignment]
    collect_roi_samples = None  # type: ignore[assignment]
    count_samples_by_class = None  # type: ignore[assignment]
    resolve_label_map = None  # type: ignore[assignment]
    resolve_split_sample_caps = None  # type: ignore[assignment]
    IMPORT_ERROR = exc

from .metrics import compute_classification_metrics


def resolve_device(device_arg: str):
    """Resolve the requested torch device."""
    if IMPORT_ERROR is not None:
        raise ImportError("Torch-based training helpers are unavailable.") from IMPORT_ERROR

    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_roi_dataloaders(
    *,
    dataset_root,
    image_size: int,
    batch_size: int,
    max_samples_per_class: int | None,
    split_max_samples_per_class: dict[str, int | None] | None,
    seed: int,
    train_augment: str = "none",
    input_normalization: str = "none",
) -> tuple[dict[str, DataLoader], dict[str, int], dict[str, dict[str, int]], dict[str, int]]:
    """Build deterministic train/val/test dataloaders for ROI experiments."""
    if IMPORT_ERROR is not None:
        raise ImportError("Torch-based training helpers are unavailable.") from IMPORT_ERROR

    split_loaders: dict[str, DataLoader] = {}
    split_sizes: dict[str, int] = {}
    split_class_counts: dict[str, dict[str, int]] = {}
    label_map = resolve_label_map(dataset_root)
    class_names = list(label_map.keys())
    resolved_split_caps = resolve_split_sample_caps(
        max_samples_per_class=max_samples_per_class,
        split_max_samples_per_class=split_max_samples_per_class,
    )

    for split in ("train", "val", "test"):
        samples = collect_roi_samples(
            dataset_root=dataset_root,
            split=split,
            max_samples_per_class=resolved_split_caps[split],
            seed=seed,
        )
        dataset = ROIDataset(
            samples=samples,
            image_size=image_size,
            grayscale=False,
            augment=train_augment if split == "train" else "none",
            normalization=input_normalization,
        )
        generator = torch.Generator()
        generator.manual_seed(seed)
        split_loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            generator=generator if split == "train" else None,
        )
        split_sizes[split] = len(samples)
        split_class_counts[split] = count_samples_by_class(samples, class_names=class_names)

    return split_loaders, split_sizes, split_class_counts, label_map


def build_class_weights(
    split_class_counts: dict[str, dict[str, int]],
    class_names: list[str],
    device=None,
):
    """Create inverse-frequency class weights from the training split."""
    if IMPORT_ERROR is not None:
        raise ImportError("Torch-based training helpers are unavailable.") from IMPORT_ERROR

    train_counts = [int(split_class_counts["train"].get(class_name, 0)) for class_name in class_names]
    total = sum(train_counts)
    num_classes = max(1, len(class_names))
    weights = [
        total / max(1, num_classes * class_count)
        for class_count in train_counts
    ]
    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


def collect_split_outputs(
    model,
    loader: DataLoader,
    criterion,
    device=None,
) -> dict[str, Any]:
    """Collect targets, probabilities, and loss for one split."""
    if IMPORT_ERROR is not None:
        raise ImportError("Torch-based training helpers are unavailable.") from IMPORT_ERROR

    model.eval()
    running_loss = 0.0
    sample_count = 0
    targets_list = []
    probabilities_list = []

    with torch.no_grad():
        for images, labels in loader:
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)

            batch_size = int(labels.size(0))
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size
            targets_list.append(labels.detach().cpu().numpy())
            probabilities_list.append(probabilities.detach().cpu().numpy())

    targets = np.concatenate(targets_list) if targets_list else np.empty((0,), dtype=np.int32)
    probabilities = (
        np.concatenate(probabilities_list, axis=0)
        if probabilities_list
        else np.empty((0, 0), dtype=np.float32)
    )
    positive_scores: np.ndarray | None
    if probabilities.ndim == 2 and probabilities.shape[1] == 2:
        positive_scores = probabilities[:, 1].astype(np.float32, copy=False)
    else:
        positive_scores = None
    return {
        "targets": targets,
        "probabilities": probabilities,
        "positive_scores": positive_scores,
        "loss": running_loss / max(1, sample_count),
    }


def select_best_threshold(
    targets: np.ndarray,
    positive_scores: np.ndarray | None,
) -> float | None:
    """Pick the validation threshold that maximizes F1, then accuracy."""
    if accuracy_score is None or f1_score is None:
        raise ImportError(
            "scikit-learn is required for training threshold selection."
        ) from IMPORT_ERROR

    if positive_scores is None:
        return None
    if targets.size == 0 or positive_scores.size == 0:
        return 0.5

    candidate_thresholds = np.unique(positive_scores)
    if candidate_thresholds.size > 256:
        candidate_thresholds = np.quantile(positive_scores, np.linspace(0.0, 1.0, 257))
        candidate_thresholds = np.unique(candidate_thresholds)

    best_threshold = 0.5
    best_key: tuple[float, float, float] | None = None
    for threshold in candidate_thresholds:
        predictions = (positive_scores >= threshold).astype(np.int32)
        key = (
            float(f1_score(targets, predictions, zero_division=0)),
            float(accuracy_score(targets, predictions)),
            -abs(float(threshold) - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def compute_metrics_from_outputs(
    outputs: dict[str, Any],
    class_names: list[str],
    threshold: float | None,
) -> dict[str, Any]:
    """Compute metrics for one split with a fixed decision rule."""
    probabilities = np.asarray(outputs["probabilities"])
    if (
        threshold is not None
        and outputs["positive_scores"] is not None
        and probabilities.ndim == 2
        and probabilities.shape[1] == 2
    ):
        predictions = (outputs["positive_scores"] >= threshold).astype(np.int32)
    else:
        predictions = np.argmax(probabilities, axis=1) if probabilities.size else np.empty((0,), dtype=np.int32)
    metrics = compute_classification_metrics(
        outputs["targets"],
        predictions,
        probability_scores=probabilities,
        class_names=class_names,
    )
    metrics["loss"] = outputs["loss"]
    metrics["threshold"] = threshold
    return metrics


def build_scheduler(
    optimizer,
    scheduler_name: str,
    total_epochs: int,
    warmup_epochs: int,
):
    """Create a simple epoch-wise scheduler with optional warmup."""
    if IMPORT_ERROR is not None:
        raise ImportError("Torch-based training helpers are unavailable.") from IMPORT_ERROR

    scheduler_name = str(scheduler_name).strip().lower()
    warmup_epochs = max(0, int(warmup_epochs))
    total_epochs = max(1, int(total_epochs))

    if scheduler_name == "none":
        return None

    if scheduler_name not in {"cosine", "step"}:
        raise ValueError(f"Unsupported scheduler: {scheduler_name!r}")

    def lr_lambda(epoch_index: int) -> float:
        epoch_number = epoch_index + 1
        if warmup_epochs > 0 and epoch_number <= warmup_epochs:
            return max(1e-6, epoch_number / warmup_epochs)

        if scheduler_name == "step":
            effective_epochs = max(1, total_epochs - warmup_epochs)
            step_size = max(1, math.ceil(effective_epochs / 3))
            decay_steps = max(0, epoch_number - warmup_epochs - 1) // step_size
            return 0.5 ** decay_steps

        if total_epochs <= warmup_epochs:
            return 1.0
        progress = (epoch_number - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def current_learning_rate(optimizer) -> float:
    """Return the first param-group learning rate."""
    return float(optimizer.param_groups[0]["lr"])


def model_selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    """Build the standard model-selection key."""
    return (float(metrics["f1"]), float(metrics["accuracy"]), -float(metrics["loss"]))


def trainable_parameter_names(model) -> list[str]:
    """Return names of parameters that currently require gradients."""
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            names.append(name)
    return names

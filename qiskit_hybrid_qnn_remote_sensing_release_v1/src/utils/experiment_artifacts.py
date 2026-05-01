"""Shared helpers for reproducible experiment artifact generation.

The training and reporting scripts write many JSON/PNG artifacts. This module
keeps formatting, environment capture, and simple summary statistics consistent
so downstream documentation can rely on a stable artifact structure.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def to_serializable(value: Any) -> Any:
    """Recursively convert common non-JSON values into serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact with consistent formatting."""
    # Every writer creates parent directories itself so scripts can focus on
    # experiment logic instead of defensive filesystem setup.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(to_serializable(payload), file, indent=2, ensure_ascii=False)


def load_json(input_path: Path) -> dict[str, Any] | None:
    """Load a JSON artifact when it exists."""
    if not input_path.exists() or not input_path.is_file():
        return None
    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def namespace_to_dict(args: Any) -> dict[str, Any]:
    """Convert an argparse namespace into a serializable dictionary."""
    return {key: to_serializable(value) for key, value in vars(args).items()}


def collect_environment_info(module_names: tuple[str, ...] = ()) -> dict[str, Any]:
    """Capture lightweight environment metadata for reproducibility."""
    versions: dict[str, str | None] = {}
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", None)
        except Exception:
            versions[module_name] = None

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "module_versions": versions,
    }


def build_runtime_payload(
    model_name: str,
    seed: int | None,
    runtime_seconds: float,
    epoch_runtime_seconds: list[float] | None = None,
    best_epoch: int | None = None,
    device: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standard runtime JSON payload."""
    payload: dict[str, Any] = {
        "model_name": model_name,
        "seed": seed,
        "device": device,
        "runtime_seconds": float(runtime_seconds),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if epoch_runtime_seconds:
        payload["epoch_runtime_seconds"] = [float(item) for item in epoch_runtime_seconds]
        payload["mean_epoch_runtime_seconds"] = float(mean(epoch_runtime_seconds))
    if best_epoch is not None:
        payload["best_epoch"] = int(best_epoch)
    if extra:
        payload.update(to_serializable(extra))
    return payload


def plot_training_curves(
    history: dict[str, list[float | int | None]],
    output_path: Path,
    title: str,
) -> None:
    """Render a two-panel training curve figure from a history JSON payload."""
    epochs = [int(epoch) for epoch in history.get("epoch", [])]
    train_loss = [float(value) for value in history.get("train_loss", [])]
    val_loss = [float(value) for value in history.get("val_loss", [])]
    val_accuracy = [float(value) for value in history.get("val_accuracy", [])]
    val_f1 = [float(value) for value in history.get("val_f1", [])]
    val_roc_auc = [
        np.nan if value is None else float(value)
        for value in history.get("val_roc_auc", [])
    ]

    # The figure always uses the same panel layout so multiple experiments can
    # be compared visually without re-learning the plot structure.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)

    axes[0].plot(epochs, train_loss, marker="o", label="Train loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, val_accuracy, marker="o", label="Val accuracy")
    axes[1].plot(epochs, val_f1, marker="o", label="Val F1")
    if not np.all(np.isnan(val_roc_auc)):
        axes[1].plot(epochs, val_roc_auc, marker="o", label="Val ROC-AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Validation Metrics")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def summarize_metric_series(values: list[float]) -> dict[str, float | None]:
    """Return mean and standard deviation for a metric series."""
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(stdev(values))}

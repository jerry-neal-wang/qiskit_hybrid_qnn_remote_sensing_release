"""Classification metric helpers shared by CNN and hybrid QNN scripts.

The repository keeps metric computation centralized so that training scripts,
seed sweeps, and paper summarizers all report the same field names and metric
definitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    SKLEARN_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    SKLEARN_IMPORT_ERROR = exc


def compute_classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probability_scores: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute binary- or multiclass-classification metrics."""
    if SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(
            "scikit-learn is required to compute classification metrics."
        ) from SKLEARN_IMPORT_ERROR

    if class_names is None:
        discovered_labels = sorted(
            set(int(item) for item in np.unique(targets).tolist())
            | set(int(item) for item in np.unique(predictions).tolist())
        )
        labels = discovered_labels
        class_names = [str(label) for label in labels]
    else:
        labels = list(range(len(class_names)))

    is_binary = len(labels) == 2
    average_mode = "binary" if is_binary else "macro"

    if is_binary:
        precision_value = float(
            precision_score(targets, predictions, zero_division=0, pos_label=1)
        )
        recall_value = float(
            recall_score(targets, predictions, zero_division=0, pos_label=1)
        )
        f1_value = float(
            f1_score(targets, predictions, zero_division=0, pos_label=1)
        )
    else:
        precision_value = float(
            precision_score(targets, predictions, average="macro", zero_division=0)
        )
        recall_value = float(
            recall_score(targets, predictions, average="macro", zero_division=0)
        )
        f1_value = float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        )

    macro_precision = float(
        precision_score(targets, predictions, average="macro", zero_division=0)
    )
    macro_recall = float(
        recall_score(targets, predictions, average="macro", zero_division=0)
    )
    macro_f1 = float(
        f1_score(targets, predictions, average="macro", zero_division=0)
    )
    weighted_precision = float(
        precision_score(targets, predictions, average="weighted", zero_division=0)
    )
    weighted_recall = float(
        recall_score(targets, predictions, average="weighted", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(targets, predictions, average="weighted", zero_division=0)
    )
    per_class_precision = precision_score(
        targets,
        predictions,
        average=None,
        labels=labels,
        zero_division=0,
    )
    per_class_recall = recall_score(
        targets,
        predictions,
        average=None,
        labels=labels,
        zero_division=0,
    )
    per_class_f1 = f1_score(
        targets,
        predictions,
        average=None,
        labels=labels,
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1_value,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "class_names": list(class_names),
        "average_mode": average_mode,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "per_class": {
            class_name: {
                "precision": float(per_class_precision[class_index]),
                "recall": float(per_class_recall[class_index]),
                "f1": float(per_class_f1[class_index]),
            }
            for class_index, class_name in enumerate(class_names)
        },
    }

    try:
        if probability_scores is None:
            metrics["roc_auc"] = None
        elif is_binary:
            positive_scores = (
                probability_scores[:, 1]
                if probability_scores.ndim == 2 and probability_scores.shape[1] == 2
                else probability_scores
            )
            metrics["roc_auc"] = float(roc_auc_score(targets, positive_scores))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(
                    targets,
                    probability_scores,
                    labels=labels,
                    average="macro",
                    multi_class="ovr",
                )
            )
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


def save_confusion_matrix_figure(
    confusion: list[list[int]],
    output_path: Path,
    title: str,
    labels: list[str] | tuple[str, ...] = ("background", "airplane"),
) -> None:
    """Render and save a confusion matrix plot."""
    matrix = np.asarray(confusion, dtype=np.int32)
    display_labels = list(labels)
    # Use a simple annotated heatmap because this figure is reused in training
    # outputs, summaries, and paper-preparation scripts.
    fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ticks = list(range(len(display_labels)))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(display_labels, rotation=30, ha="right")
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="black",
                fontsize=11,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

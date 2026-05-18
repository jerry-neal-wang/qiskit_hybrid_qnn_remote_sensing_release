#!/usr/bin/env python3
"""Generate per-class tables and confusion figures for one low-regime experiment root."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = (
    "baseline_cnn",
    "hybrid_control",
    "hybrid_no_entanglement",
    "hybrid_zz_real_amplitudes",
)
METHOD_LABELS = {
    "baseline_cnn": "CNN Baseline",
    "hybrid_control": "Classical Control",
    "hybrid_no_entanglement": "QNN No-Ent",
    "hybrid_zz_real_amplitudes": "QNN ZZ+RealAmp",
}
METHOD_COLORS = {
    "baseline_cnn": "#6D6D6D",
    "hybrid_control": "#2F6B9A",
    "hybrid_no_entanglement": "#2A9D8F",
    "hybrid_zz_real_amplitudes": "#E0A526",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-class low-regime analysis tables and figures."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Low-regime experiment root that contains shot_*/seed_* artifacts.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=None,
        help="Optional shot values to include. Defaults to all detected shots.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Method ids to include in tables and figures.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summary_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (values[0], 0.0)
    return (mean(values), stdev(values))


def discover_records(experiment_root: Path, methods: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shot_dir in sorted(experiment_root.glob("shot_*")):
        if not shot_dir.is_dir():
            continue
        try:
            shot = int(shot_dir.name.split("_", 1)[1])
        except Exception:
            continue
        for seed_dir in sorted(shot_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except Exception:
                continue
            for method_dir in sorted(seed_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                if method not in methods and not (method == "baseline" and "baseline_cnn" in methods):
                    continue
                if method == "baseline":
                    method_id = "baseline_cnn"
                    metrics_path = method_dir / "baseline_cnn_test_metrics.json"
                elif method == "hybrid_control":
                    method_id = "hybrid_control"
                    metrics_path = method_dir / "hybrid_control_test_metrics.json"
                else:
                    method_id = method
                    metrics_path = method_dir / "hybrid_qnn_test_metrics.json"
                if not metrics_path.exists():
                    continue
                payload = load_json(metrics_path)
                records.append(
                    {
                        "shot": shot,
                        "seed": seed,
                        "method": method_id,
                        "metrics_path": metrics_path,
                        "payload": payload,
                    }
                )
    return records


def build_per_class_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], dict[str, list[float]]] = {}
    for record in records:
        metrics = record["payload"]["metrics"]
        per_class = metrics.get("per_class", {})
        for class_name, class_metrics in per_class.items():
            key = (int(record["shot"]), str(record["method"]), str(class_name))
            grouped.setdefault(
                key,
                {"precision": [], "recall": [], "f1": []},
            )
            grouped[key]["precision"].append(float(class_metrics["precision"]))
            grouped[key]["recall"].append(float(class_metrics["recall"]))
            grouped[key]["f1"].append(float(class_metrics["f1"]))

    rows: list[dict[str, Any]] = []
    for (shot, method, class_name), metric_lists in sorted(grouped.items()):
        row: dict[str, Any] = {
            "shot": shot,
            "method": method,
            "display_name": METHOD_LABELS.get(method, method),
            "class_name": class_name,
            "num_runs": len(metric_lists["f1"]),
        }
        for metric_name, values in metric_lists.items():
            metric_mean, metric_std = summary_stats(values)
            row[f"{metric_name}_mean"] = metric_mean
            row[f"{metric_name}_std"] = metric_std
        rows.append(row)
    return rows


def build_confusion_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        metrics = record["payload"]["metrics"]
        class_names = metrics["class_names"]
        confusion = metrics["confusion_matrix"]
        for true_index, true_name in enumerate(class_names):
            row_total = sum(confusion[true_index])
            for pred_index, pred_name in enumerate(class_names):
                value = float(confusion[true_index][pred_index])
                rows.append(
                    {
                        "shot": int(record["shot"]),
                        "seed": int(record["seed"]),
                        "method": str(record["method"]),
                        "display_name": METHOD_LABELS.get(str(record["method"]), str(record["method"])),
                        "true_class": true_name,
                        "pred_class": pred_name,
                        "count": value,
                        "row_normalized": value / row_total if row_total > 0 else 0.0,
                    }
                )
    return rows


def plot_per_class_f1(
    per_class_rows: list[dict[str, Any]],
    *,
    shot: int,
    methods: list[str],
    output_path: Path,
) -> None:
    shot_rows = [row for row in per_class_rows if int(row["shot"]) == shot and row["method"] in methods]
    if not shot_rows:
        return
    configure_style()
    class_names = []
    for row in shot_rows:
        class_name = str(row["class_name"])
        if class_name not in class_names:
            class_names.append(class_name)
    figure, axis = plt.subplots(figsize=(max(7.2, 1.5 * len(class_names) * len(methods)), 4.8))
    x_positions = list(range(len(class_names)))
    width = 0.8 / max(1, len(methods))
    offsets = [
        (index - (len(methods) - 1) / 2.0) * width
        for index in range(len(methods))
    ]
    for offset, method in zip(offsets, methods):
        method_rows = {row["class_name"]: row for row in shot_rows if row["method"] == method}
        axis.bar(
            [x + offset for x in x_positions],
            [float(method_rows[class_name]["f1_mean"]) for class_name in class_names],
            width=width,
            color=METHOD_COLORS.get(method, "#555555"),
            label=METHOD_LABELS.get(method, method),
            yerr=[float(method_rows[class_name]["f1_std"]) for class_name in class_names],
            capsize=3.0,
        )
    axis.set_title(f"Per-Class F1 at Shot={shot}")
    axis.set_ylabel("F1")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(class_names)
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False, ncol=min(4, len(methods)))
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    ensure_dir(output_path.parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_mean_confusion(
    confusion_rows: list[dict[str, Any]],
    *,
    shot: int,
    methods: list[str],
    output_path: Path,
) -> None:
    shot_rows = [row for row in confusion_rows if int(row["shot"]) == shot and row["method"] in methods]
    if not shot_rows:
        return
    configure_style()
    class_names = []
    for row in shot_rows:
        true_name = str(row["true_class"])
        if true_name not in class_names:
            class_names.append(true_name)
    figure, axes = plt.subplots(
        1,
        len(methods),
        figsize=(4.4 * len(methods), 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()
    for axis, method in zip(axes_flat, methods):
        method_rows = [row for row in shot_rows if row["method"] == method]
        grid = [[0.0 for _ in class_names] for _ in class_names]
        count_grid = [[0 for _ in class_names] for _ in class_names]
        for row in method_rows:
            i = class_names.index(str(row["true_class"]))
            j = class_names.index(str(row["pred_class"]))
            grid[i][j] += float(row["row_normalized"])
            count_grid[i][j] += 1
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if count_grid[i][j] > 0:
                    grid[i][j] /= count_grid[i][j]
        image = axis.imshow(grid, vmin=0.0, vmax=1.0, cmap="Blues")
        axis.set_title(METHOD_LABELS.get(method, method))
        axis.set_xticks(range(len(class_names)))
        axis.set_yticks(range(len(class_names)))
        axis.set_xticklabels(class_names, rotation=45, ha="right")
        axis.set_yticklabels(class_names)
        axis.set_xlabel("Predicted")
        if axis is axes_flat[0]:
            axis.set_ylabel("True")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                axis.text(
                    j,
                    i,
                    f"{grid[i][j]:.2f}",
                    ha="center",
                    va="center",
                    color="#111111" if grid[i][j] < 0.6 else "white",
                    fontsize=8,
                )
    colorbar = figure.colorbar(image, ax=axes_flat, fraction=0.025, pad=0.03)
    colorbar.set_label("Row-Normalized Accuracy")
    figure.suptitle(f"Mean Confusion Matrices at Shot={shot}", y=1.02)
    ensure_dir(output_path.parent)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.expanduser().resolve()
    if not experiment_root.exists():
        print(f"Error: experiment root does not exist: {experiment_root}")
        return 1

    requested_methods = [str(method) for method in args.methods]
    method_set = set(requested_methods)
    records = discover_records(experiment_root, methods=method_set)
    if not records:
        print(f"Error: no compatible metrics discovered under {experiment_root}")
        return 1

    selected_shots = set(args.shots) if args.shots else {int(record["shot"]) for record in records}
    records = [record for record in records if int(record["shot"]) in selected_shots]
    if not records:
        print("Error: no records remained after shot filtering.")
        return 1

    per_class_rows = build_per_class_rows(records)
    confusion_rows = build_confusion_rows(records)
    output_root = experiment_root / "paper_ready" / "per_class"
    tables_dir = output_root / "tables"
    figures_dir = output_root / "figures"
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    write_csv(tables_dir / "tab_per_class_metrics.csv", per_class_rows)
    write_csv(tables_dir / "tab_confusion_cells.csv", confusion_rows)

    for shot in sorted(selected_shots):
        shot_methods = [
            method
            for method in requested_methods
            if any(int(row["shot"]) == shot and row["method"] == method for row in per_class_rows)
        ]
        plot_per_class_f1(
            per_class_rows,
            shot=shot,
            methods=shot_methods,
            output_path=figures_dir / f"fig_per_class_f1_shot{shot}.png",
        )
        plot_mean_confusion(
            confusion_rows,
            shot=shot,
            methods=shot_methods,
            output_path=figures_dir / f"fig_mean_confusion_shot{shot}.png",
        )

    print(f"Saved per-class tables to: {tables_dir}")
    print(f"Saved per-class figures to: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

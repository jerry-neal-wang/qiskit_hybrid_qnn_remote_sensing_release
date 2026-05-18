#!/usr/bin/env python3
"""Generate composite, paper-facing figures for the DIOR hybrid-QNN study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "paper_composite_figures"

TRANSPORT_MAIN_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "dior_subset_candidate_screening"
    / "transport_logistics4_shot16_bblr01_7seed"
)
TRANSPORT_SHOT32_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "dior_subset_candidate_screening"
    / "transport_logistics4_shot32_bblr01_7seed"
)
TRANSPORT_SHOT64_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "dior_subset_candidate_screening"
    / "transport_logistics4_shot64_bblr01_7seed"
)
TRANSPORT_RESNET18_ROOT = (
    PROJECT_ROOT / "artifacts" / "transport_logistics4_resnet18_shot16_bblr01_7seed"
)
URBAN_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "dior_subset_candidate_screening"
    / "urban_structural4_shot16_bblr01_7seed"
)

TRANSPORT_DATASET_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "subset_candidates" / "transport_logistics4"
)
URBAN_DATASET_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "subset_candidates" / "urban_structural4"
)

METHOD_ORDER = [
    "baseline_cnn",
    "hybrid_control",
    "hybrid_no_entanglement",
    "hybrid_zz_real_amplitudes",
]
METHOD_LABELS = {
    "baseline_cnn": "CNN baseline",
    "hybrid_control": "Classical control",
    "hybrid_no_entanglement": "QNN no-ent",
    "hybrid_zz_real_amplitudes": "QNN ZZ+RealAmp",
}
METHOD_COLORS = {
    "baseline_cnn": "#7A7A7A",
    "hybrid_control": "#356D9B",
    "hybrid_no_entanglement": "#2F9D8F",
    "hybrid_zz_real_amplitudes": "#E6AA1C",
}
QNN_METHODS = ["hybrid_no_entanglement", "hybrid_zz_real_amplitudes"]
QNN_LABELS = {
    "hybrid_no_entanglement": "No-ent",
    "hybrid_zz_real_amplitudes": "ZZ+RealAmp",
}

PANEL_LABEL_STYLE = {
    "fontsize": 13,
    "fontweight": "bold",
    "color": "#222222",
}
TEXT_DARK = "#1F2933"
TEXT_MUTED = "#5B6770"
GRID_COLOR = "#D6DBE1"
EDGE_COLOR = "#2B2B2B"


@dataclass(frozen=True)
class Condition:
    """One boundary-analysis condition."""

    condition_id: str
    label: str
    short_label: str
    subset_name: str
    backbone_name: str
    shot: int
    num_seeds: int
    root: Path


CONDITIONS = [
    Condition(
        condition_id="transport16_smallcnn",
        label="transport_logistics4 / small CNN / shot=16",
        short_label="Transport\nsmall CNN\n16-shot",
        subset_name="transport_logistics4",
        backbone_name="Small CNN",
        shot=16,
        num_seeds=7,
        root=TRANSPORT_MAIN_ROOT,
    ),
    Condition(
        condition_id="transport32_smallcnn",
        label="transport_logistics4 / small CNN / shot=32",
        short_label="Transport\nsmall CNN\n32-shot",
        subset_name="transport_logistics4",
        backbone_name="Small CNN",
        shot=32,
        num_seeds=7,
        root=TRANSPORT_SHOT32_ROOT,
    ),
    Condition(
        condition_id="transport64_smallcnn",
        label="transport_logistics4 / small CNN / shot=64",
        short_label="Transport\nsmall CNN\n64-shot",
        subset_name="transport_logistics4",
        backbone_name="Small CNN",
        shot=64,
        num_seeds=7,
        root=TRANSPORT_SHOT64_ROOT,
    ),
    Condition(
        condition_id="transport16_resnet18",
        label="transport_logistics4 / ResNet18 / shot=16",
        short_label="Transport\nResNet18\n16-shot",
        subset_name="transport_logistics4",
        backbone_name="ResNet18",
        shot=16,
        num_seeds=7,
        root=TRANSPORT_RESNET18_ROOT,
    ),
    Condition(
        condition_id="urban16_smallcnn",
        label="urban_structural4 / small CNN / shot=16",
        short_label="Urban\nsmall CNN\n16-shot",
        subset_name="urban_structural4",
        backbone_name="Small CNN",
        shot=16,
        num_seeds=7,
        root=URBAN_ROOT,
    ),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate composite SCI-style figures for the hybrid-QNN DIOR study."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output directory (default: {OUTPUT_ROOT})",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a CSV file into a list of dict rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value: str | None) -> float | None:
    """Convert one possibly empty field to float."""
    if value is None or value == "":
        return None
    return float(value)


def wrap_text(value: str, width: int) -> str:
    """Wrap one string for figure labels."""
    return textwrap.fill(value, width=width)


def format_count(value: int) -> str:
    """Format large class counts more compactly."""
    if value >= 1000:
        return f"{value / 1000.0:.1f}k"
    return str(value)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Add a bold panel label to one axis."""
    ax.text(
        -0.08,
        1.065,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        **PANEL_LABEL_STYLE,
    )


def configure_style() -> None:
    """Set a cleaner academic plotting style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.titlesize": 9.8,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "axes.edgecolor": EDGE_COLOR,
            "axes.linewidth": 0.8,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def table_dir_for_root(root: Path) -> Path:
    """Return the directory that stores paper-facing tables for one root."""
    paper_ready_tables = root / "paper_ready" / "tables"
    if paper_ready_tables.exists():
        return paper_ready_tables
    figures_tables = root / "figures" / "tables"
    if figures_tables.exists():
        return figures_tables
    raise FileNotFoundError(f"Cannot find table directory under {root}")


def per_class_table_for_root(root: Path) -> Path:
    """Return the per-class table path for one root."""
    path = root / "paper_ready" / "per_class" / "tables" / "tab_per_class_metrics.csv"
    if path.exists():
        return path
    raise FileNotFoundError(f"Cannot find per-class table under {root}")


def summary_rows_for_root(root: Path) -> list[dict[str, str]]:
    """Load summary rows for one experiment root."""
    return load_csv_rows(root / "lowshot_summary_rows.csv")


def detail_rows_for_root(root: Path) -> list[dict[str, str]]:
    """Load seed-level detail rows for one experiment root."""
    return load_csv_rows(root / "lowshot_detail_rows.csv")


def pairwise_rows_for_root(root: Path) -> list[dict[str, str]]:
    """Load paired comparison rows for one experiment root."""
    return load_csv_rows(table_dir_for_root(root) / "tab_dior_lowregime_pairwise_tests.csv")


def summary_row(rows: list[dict[str, str]], method: str, shot: int) -> dict[str, str]:
    """Return one summary row for a method and shot."""
    for row in rows:
        if row["method"] == method and int(row["shot"]) == shot:
            return row
    raise KeyError(f"Missing summary row for method={method}, shot={shot}")


def pairwise_row(rows: list[dict[str, str]], left: str, right: str, shot: int) -> dict[str, str]:
    """Return one pairwise-comparison row."""
    for row in rows:
        if (
            row["left_method"] == left
            and row["right_method"] == right
            and int(row["shot"]) == shot
        ):
            return row
    raise KeyError(f"Missing pairwise row for {left} -> {right}, shot={shot}")


def seed_scores(detail_rows: list[dict[str, str]], method: str, shot: int) -> list[tuple[int, float]]:
    """Return seed-level macro-F1 values sorted by seed."""
    scores = [
        (int(row["seed"]), float(row["macro_f1"]))
        for row in detail_rows
        if row["method"] == method and int(row["shot"]) == shot
    ]
    return sorted(scores, key=lambda item: item[0])


def seed_deltas(
    detail_rows: list[dict[str, str]],
    left_method: str,
    right_method: str,
    shot: int,
) -> list[tuple[int, float]]:
    """Return seed-paired deltas of right minus left."""
    left_scores = {seed: score for seed, score in seed_scores(detail_rows, left_method, shot)}
    right_scores = {seed: score for seed, score in seed_scores(detail_rows, right_method, shot)}
    seeds = sorted(set(left_scores) & set(right_scores))
    return [(seed, right_scores[seed] - left_scores[seed]) for seed in seeds]


def bootstrap_ci(
    values: list[float],
    *,
    seed: int = 0,
    num_samples: int = 20000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute a simple bootstrap CI for the sample mean."""
    if not values:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    sample_size = array.shape[0]
    sampled = rng.choice(array, size=(num_samples, sample_size), replace=True)
    sampled_means = sampled.mean(axis=1)
    lower = float(np.quantile(sampled_means, alpha / 2.0))
    upper = float(np.quantile(sampled_means, 1.0 - alpha / 2.0))
    prob_gt_zero = float((sampled_means > 0.0).mean())
    return (lower, upper, prob_gt_zero)


def significance_marker(p_value: float) -> str:
    """Map p-value to a compact significance marker."""
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def mean_confusion_matrix(root: Path, method: str, shot: int) -> tuple[list[str], np.ndarray]:
    """Compute the mean row-normalized confusion matrix across seeds."""
    detail_rows = detail_rows_for_root(root)
    shot_rows = [row for row in detail_rows if int(row["shot"]) == shot and row["method"] == method]
    if not shot_rows:
        raise ValueError(f"No detail rows for {root}, {method}, shot={shot}")

    matrices: list[np.ndarray] = []
    class_names: list[str] | None = None
    for row in shot_rows:
        seed = int(row["seed"])
        if method == "baseline_cnn":
            metrics_path = (
                root
                / f"shot_{shot}"
                / f"seed_{seed}"
                / "baseline"
                / "baseline_cnn_test_metrics.json"
            )
        elif method == "hybrid_control":
            metrics_path = (
                root
                / f"shot_{shot}"
                / f"seed_{seed}"
                / "hybrid_control"
                / "hybrid_control_test_metrics.json"
            )
        else:
            metrics_path = (
                root
                / f"shot_{shot}"
                / f"seed_{seed}"
                / method
                / "hybrid_qnn_test_metrics.json"
            )
        payload = load_json(metrics_path)
        metrics = payload["metrics"]
        if class_names is None:
            class_names = list(metrics["class_names"])
        confusion = np.asarray(metrics["confusion_matrix"], dtype=float)
        row_sums = confusion.sum(axis=1, keepdims=True)
        normalized = np.divide(
            confusion,
            row_sums,
            out=np.zeros_like(confusion),
            where=row_sums > 0.0,
        )
        matrices.append(normalized)

    assert class_names is not None
    return class_names, np.mean(matrices, axis=0)


def mean_history(root: Path, method: str, shot: int, metric_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute epoch-aligned mean and std for one history metric."""
    detail_rows = detail_rows_for_root(root)
    shot_rows = [row for row in detail_rows if int(row["shot"]) == shot and row["method"] == method]
    histories: list[list[float]] = []
    epoch_arrays: list[list[int]] = []
    for row in shot_rows:
        seed = int(row["seed"])
        if method == "baseline_cnn":
            history_path = (
                root
                / f"shot_{shot}"
                / f"seed_{seed}"
                / "baseline"
                / "baseline_cnn_history.json"
            )
        elif method == "hybrid_control":
            history_path = (
                root
                / f"shot_{shot}"
                / f"seed_{seed}"
                / "hybrid_control"
                / "hybrid_control_history.json"
            )
        else:
            history_path = (
                root / f"shot_{shot}" / f"seed_{seed}" / method / "hybrid_qnn_history.json"
            )
        payload = load_json(history_path)
        histories.append([float(value) for value in payload[metric_name]])
        epoch_arrays.append([int(value) for value in payload["epoch"]])

    if not histories:
        return (np.array([]), np.array([]), np.array([]))

    max_epoch = max(len(history) for history in histories)
    epoch_values = np.arange(1, max_epoch + 1)
    stacked = np.full((len(histories), max_epoch), np.nan, dtype=float)
    for row_index, history in enumerate(histories):
        stacked[row_index, : len(history)] = history
    mean_values = np.nanmean(stacked, axis=0)
    std_values = np.nanstd(stacked, axis=0, ddof=0)
    return (epoch_values, mean_values, std_values)


def sample_image_paths(dataset_root: Path, class_name: str, count: int = 2) -> list[Path]:
    """Pick evenly spaced example crops for one class."""
    candidates = sorted((dataset_root / "test" / class_name).glob("*.jpg"))
    if len(candidates) < count:
        return candidates
    if count == 1:
        return [candidates[len(candidates) // 2]]
    indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[index] for index in indices]


def split_counts(dataset_root: Path) -> dict[str, dict[str, int]]:
    """Load split counts per class from split_summary.csv."""
    rows = load_csv_rows(dataset_root / "split_summary.csv")
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        split_name = row["split"]
        counts[split_name] = {}
        for key, value in row.items():
            if key.endswith("_count") and key not in {"source_image_count"}:
                class_name = key[: -len("_count")]
                counts[split_name][class_name] = int(value)
    return counts


def draw_subset_montage(
    figure: plt.Figure,
    parent_spec: gridspec.SubplotSpec,
    *,
    dataset_root: Path,
    class_order: list[str],
    title: str,
    counts: dict[str, dict[str, int]],
    panel_label: str,
) -> None:
    """Draw one small multi-image subset montage."""
    add_label_axis = figure.add_subplot(parent_spec)
    add_label_axis.axis("off")
    add_panel_label(add_label_axis, panel_label)

    outer = gridspec.GridSpecFromSubplotSpec(
        3,
        len(class_order),
        subplot_spec=parent_spec,
        height_ratios=[0.32, 1.0, 1.0],
        wspace=0.06,
        hspace=0.08,
    )

    header_axis = figure.add_subplot(outer[0, :])
    header_axis.axis("off")
    header_axis.text(
        0.0,
        1.05,
        title,
        ha="left",
        va="bottom",
        fontsize=10.2,
        fontweight="bold",
        color=TEXT_DARK,
        transform=header_axis.transAxes,
    )
    for col_index, class_name in enumerate(class_order):
        image_paths = sample_image_paths(dataset_root, class_name, count=2)
        train_count = counts.get("train", {}).get(class_name, 0)
        test_count = counts.get("test", {}).get(class_name, 0)
        for row_index in range(2):
            axis = figure.add_subplot(outer[row_index + 1, col_index])
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_aspect("equal")
            if row_index < len(image_paths):
                image = plt.imread(image_paths[row_index])
                axis.imshow(image)
            else:
                axis.imshow(np.ones((32, 32, 3), dtype=float))
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)
                spine.set_edgecolor("#D1D7DF")
            if row_index == 0:
                axis.set_title(
                    f"{class_name}\ntr {format_count(train_count)} | te {format_count(test_count)}",
                    fontsize=7.8,
                    pad=3.5,
                )


def draw_pipeline_schematic(ax: plt.Axes) -> None:
    """Draw the study pipeline schematic."""
    ax.axis("off")

    def draw_box(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        subtitle: str,
        facecolor: str,
        edgecolor: str = "#C7CED6",
    ) -> None:
        def wrap_lines(value: str, width_chars: int) -> str:
            return "\n".join(wrap_text(line, width_chars) for line in value.splitlines())

        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            linewidth=0.9,
            edgecolor=edgecolor,
            facecolor=facecolor,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x + width * 0.06,
            y + height * 0.67,
            wrap_lines(title, 20),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=TEXT_DARK,
        )
        ax.text(
            x + width * 0.06,
            y + height * 0.31,
            wrap_lines(subtitle, 26),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.9,
            color=TEXT_MUTED,
        )

    ax.text(
        0.0,
        1.02,
        "Matched-control evaluation ladder",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.8,
        fontweight="bold",
        color=TEXT_DARK,
    )

    draw_box(
        0.02,
        0.70,
        0.20,
        0.18,
        "DIOR ROI crops",
        "transport main subset\nurban boundary subset",
        "#F5F7FA",
    )
    draw_box(
        0.29,
        0.70,
        0.20,
        0.18,
        "Low-shot split",
        "16 / 32 / 64 shots\npaired seeds 42-48",
        "#F5F7FA",
    )
    draw_box(
        0.56,
        0.70,
        0.25,
        0.18,
        "Shared representation",
        "small CNN mainline\nor ResNet18 transfer",
        "#F5F7FA",
    )
    draw_box(
        0.06,
        0.43,
        0.25,
        0.17,
        "Matched classical branch",
        "same backbone, projection,\nfusion, and optimizer",
        "#EDF4FB",
        "#9BBAD3",
    )
    draw_box(
        0.38,
        0.43,
        0.25,
        0.17,
        "No-ent QNN branch",
        "four-qubit Ry\nZ expectations",
        "#EEF8F6",
        "#8BCBC1",
    )
    draw_box(
        0.70,
        0.43,
        0.25,
        0.17,
        "ZZ+RA QNN branch",
        "ZZFeatureMap plus\nRealAmplitudes",
        "#FFF5D8",
        "#E3BE54",
    )
    draw_box(
        0.05,
        0.15,
        0.21,
        0.14,
        "QCAD",
        "positive paired gain\nunder CI and cost limits",
        "#F9FAFB",
    )
    draw_box(
        0.30,
        0.15,
        0.20,
        0.14,
        "LSEC",
        "gain per labeled ROI\nand runtime ratio",
        "#F9FAFB",
    )
    draw_box(
        0.54,
        0.15,
        0.18,
        0.14,
        "QIB",
        "class-structured\nper-class gain",
        "#F9FAFB",
    )
    draw_box(
        0.76,
        0.15,
        0.19,
        0.14,
        "Stress tests",
        "shot, backbone,\nsubset, runtime",
        "#F9FAFB",
    )

    arrow_specs = [
        ((0.22, 0.79), (0.29, 0.79)),
        ((0.49, 0.79), (0.56, 0.79)),
        ((0.685, 0.70), (0.185, 0.60)),
        ((0.685, 0.70), (0.505, 0.60)),
        ((0.685, 0.70), (0.825, 0.60)),
        ((0.185, 0.43), (0.155, 0.29)),
        ((0.505, 0.43), (0.395, 0.29)),
        ((0.825, 0.43), (0.855, 0.29)),
    ]
    for start, end in arrow_specs:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#5F6C77",
                transform=ax.transAxes,
            )
        )

    ax.plot(
        [0.05, 0.95],
        [0.355, 0.355],
        color="#B7C0CA",
        linewidth=0.9,
        transform=ax.transAxes,
    )

def draw_boundary_protocol(ax: plt.Axes) -> None:
    """Draw a compact boundary-analysis map."""
    labels = [
        "TL4 small CNN\n16-shot",
        "TL4 small CNN\n32-shot",
        "TL4 small CNN\n64-shot",
        "TL4 ResNet18\n16-shot",
        "Urban4 small CNN\n16-shot",
    ]
    no_ent = np.asarray([0.1976, 0.0094, -0.0021, -0.0011, -0.0063])
    zz_realamp = np.asarray([0.1491, 0.0018, 0.0001, 0.0051, -0.0583])
    y_positions = np.arange(len(labels), dtype=float)
    bar_height = 0.30

    ax.axvspan(0.0, 0.23, facecolor="#E7F5F1", edgecolor="none", zorder=0)
    ax.axvline(0.0, color="#3E4852", linewidth=0.9, zorder=1)
    ax.axvline(0.10, color="#6C8E82", linewidth=0.9, linestyle="--", zorder=1)
    ax.barh(
        y_positions - bar_height / 2,
        no_ent,
        height=bar_height,
        color=METHOD_COLORS["hybrid_no_entanglement"],
        alpha=0.92,
        label="No-ent",
        zorder=2,
    )
    ax.barh(
        y_positions + bar_height / 2,
        zz_realamp,
        height=bar_height,
        color=METHOD_COLORS["hybrid_zz_real_amplitudes"],
        alpha=0.92,
        label="ZZ+RA",
        zorder=2,
    )

    for row_index, (no_ent_value, zz_value) in enumerate(zip(no_ent, zz_realamp)):
        for offset, value in [(-bar_height / 2, no_ent_value), (bar_height / 2, zz_value)]:
            x_text = value + (0.006 if value >= 0 else -0.006)
            ax.text(
                x_text,
                row_index + offset,
                f"{value:+.3f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=7.5,
                color=TEXT_DARK,
            )

    ax.text(
        0.102,
        -0.62,
        "practical-gain\nthreshold",
        ha="left",
        va="top",
        fontsize=7.3,
        color="#52756A",
    )
    ax.text(
        0.198,
        -0.62,
        "main low-shot\nwindow",
        ha="right",
        va="top",
        fontsize=7.3,
        color="#52756A",
        fontweight="bold",
    )
    ax.set_title("Boundary map for stress tests", pad=8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.075, 0.23)
    ax.set_xlabel("Macro-F1 delta vs matched control")
    ax.grid(True, axis="x")
    ax.legend(frameon=False, loc="lower right")
    for spine_name in ["top", "right"]:
        ax.spines[spine_name].set_visible(False)


def plot_seed_scatter_performance(ax: plt.Axes, detail_rows: list[dict[str, str]], shot: int) -> None:
    """Plot seed-level macro-F1 scatter with mean +- std."""
    rng = np.random.default_rng(0)
    x_positions = np.arange(len(METHOD_ORDER), dtype=float)
    for index, method in enumerate(METHOD_ORDER):
        scores = [score for _, score in seed_scores(detail_rows, method, shot)]
        jitter = rng.uniform(-0.07, 0.07, size=len(scores))
        ax.scatter(
            np.full(len(scores), x_positions[index]) + jitter,
            scores,
            s=22,
            color=METHOD_COLORS[method],
            alpha=0.7,
            linewidths=0.0,
            zorder=3,
        )
        mean_value = float(np.mean(scores))
        std_value = float(np.std(scores, ddof=1))
        ax.errorbar(
            [x_positions[index]],
            [mean_value],
            yerr=[[std_value], [std_value]],
            fmt="o",
            markersize=8.5,
            color=METHOD_COLORS[method],
            capsize=4.0,
            linewidth=1.7,
            zorder=4,
        )
        ax.text(
            x_positions[index],
            mean_value + std_value + 0.02,
            f"{mean_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            color=TEXT_DARK,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([METHOD_LABELS[method] for method in METHOD_ORDER], rotation=0)
    ax.tick_params(axis="x", pad=6)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Seed-level main result at 16-shot")
    ax.set_ylim(0.05, 0.78)
    ax.grid(True, axis="y")


def plot_paired_delta_panel(
    ax: plt.Axes,
    detail_rows: list[dict[str, str]],
    pairwise_rows: list[dict[str, str]],
    shot: int,
) -> None:
    """Plot paired seed deltas and bootstrap intervals vs classical control."""
    rng = np.random.default_rng(1)
    x_positions = np.arange(len(QNN_METHODS), dtype=float)
    for index, method in enumerate(QNN_METHODS):
        deltas = [delta for _, delta in seed_deltas(detail_rows, "hybrid_control", method, shot)]
        row = pairwise_row(pairwise_rows, "hybrid_control", method, shot)
        ci_low, ci_high, _ = bootstrap_ci(deltas, seed=index + 10)
        jitter = rng.uniform(-0.06, 0.06, size=len(deltas))
        ax.scatter(
            np.full(len(deltas), x_positions[index]) + jitter,
            deltas,
            color=METHOD_COLORS[method],
            s=24,
            alpha=0.75,
            linewidths=0.0,
            zorder=3,
        )
        mean_delta = float(np.mean(deltas))
        ax.errorbar(
            [x_positions[index]],
            [mean_delta],
            yerr=[[mean_delta - ci_low], [ci_high - mean_delta]],
            fmt="o",
            color=METHOD_COLORS[method],
            markersize=8.5,
            capsize=4.0,
            linewidth=1.8,
            zorder=4,
        )
        p_value = float(row["wilcoxon_p_value"])
        marker = significance_marker(p_value)
        ax.text(
            x_positions[index],
            ci_high + 0.018,
            f"Δ={mean_delta:+.3f}\np={p_value:.3f} {marker}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=TEXT_DARK,
        )
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([QNN_LABELS[method] for method in QNN_METHODS])
    ax.set_ylabel("Macro-F1 delta vs control")
    ax.set_title("Paired gains over the matched control")
    ax.grid(True, axis="y")


def plot_training_curve_panel(ax: plt.Axes, root: Path, shot: int) -> None:
    """Plot mean validation macro-F1 trajectories."""
    for method in METHOD_ORDER:
        epochs, mean_values, std_values = mean_history(root, method, shot, "val_f1")
        if epochs.size == 0:
            continue
        ax.plot(
            epochs,
            mean_values,
            color=METHOD_COLORS[method],
            linewidth=1.8,
            label=METHOD_LABELS[method],
        )
        ax.fill_between(
            epochs,
            mean_values - std_values,
            mean_values + std_values,
            color=METHOD_COLORS[method],
            alpha=0.12,
            linewidth=0.0,
        )
    ax.axvline(2.5, color="#9CA7B2", linestyle="--", linewidth=1.0)
    ax.text(
        2.58,
        0.12,
        "hybrid backbone\nunfreezes",
        color=TEXT_MUTED,
        fontsize=7.7,
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Optimization dynamics")
    ax.set_ylim(0.08, 0.73)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, ncol=2, loc="lower right")


def plot_per_class_panel(ax: plt.Axes, rows: list[dict[str, str]], shot: int) -> None:
    """Plot per-class macro-F1 by method."""
    filtered = [row for row in rows if int(row["shot"]) == shot]
    class_order = ["airplane", "ship", "vehicle", "harbor"]
    width = 0.18
    x_positions = np.arange(len(class_order), dtype=float)
    offsets = (np.arange(len(METHOD_ORDER)) - (len(METHOD_ORDER) - 1) / 2.0) * width
    for offset, method in zip(offsets, METHOD_ORDER):
        method_rows = {row["class_name"]: row for row in filtered if row["method"] == method}
        ax.bar(
            x_positions + offset,
            [float(method_rows[class_name]["f1_mean"]) for class_name in class_order],
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        ax.errorbar(
            x_positions + offset,
            [float(method_rows[class_name]["f1_mean"]) for class_name in class_order],
            yerr=[float(method_rows[class_name]["f1_std"]) for class_name in class_order],
            fmt="none",
            ecolor=METHOD_COLORS[method],
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_order)
    ax.set_ylabel("Per-class F1")
    ax.set_ylim(0.0, 0.95)
    ax.set_title("Class-structured effect")
    ax.grid(True, axis="y")


def plot_confusion_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    class_names: list[str],
    title: str,
) -> None:
    """Plot one row-normalized confusion matrix."""
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if value > 0.52 else TEXT_DARK,
            )
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    return image


def build_boundary_records() -> list[dict[str, Any]]:
    """Collect summary and paired-comparison records across boundary conditions."""
    records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        summary_rows = summary_rows_for_root(condition.root)
        pairwise_rows = pairwise_rows_for_root(condition.root)
        detail_rows = detail_rows_for_root(condition.root)
        control_row = summary_row(summary_rows, "hybrid_control", condition.shot)
        control_runtime = float(control_row["runtime_seconds_mean"])
        for method in QNN_METHODS:
            row = pairwise_row(pairwise_rows, "hybrid_control", method, condition.shot)
            summary = summary_row(summary_rows, method, condition.shot)
            deltas = [delta for _, delta in seed_deltas(detail_rows, "hybrid_control", method, condition.shot)]
            bootstrap_seed_text = f"{condition.condition_id}:{method}"
            bootstrap_seed = sum(
                (index + 1) * ord(character)
                for index, character in enumerate(bootstrap_seed_text)
            ) % (2**32)
            ci_low, ci_high, prob_gt_zero = bootstrap_ci(
                deltas,
                seed=bootstrap_seed,
            )
            records.append(
                {
                    "condition_id": condition.condition_id,
                    "condition_label": condition.label,
                    "condition_short_label": condition.short_label,
                    "subset_name": condition.subset_name,
                    "backbone_name": condition.backbone_name,
                    "shot": condition.shot,
                    "num_seeds": condition.num_seeds,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "mean_delta": float(row["mean_delta_macro_f1"]),
                    "p_value": float(row["wilcoxon_p_value"]),
                    "macro_f1_mean": float(summary["macro_f1_mean"]),
                    "macro_f1_std": float(summary["macro_f1_std"]),
                    "runtime_seconds_mean": float(summary["runtime_seconds_mean"]),
                    "runtime_ratio_vs_control": float(summary["runtime_seconds_mean"]) / control_runtime,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "bootstrap_prob_gt_zero": prob_gt_zero,
                }
            )
    return records


def plot_boundary_heatmap(ax: plt.Axes, records: list[dict[str, Any]]) -> None:
    """Plot a delta heatmap across conditions and QNN variants."""
    row_labels = [condition.short_label for condition in CONDITIONS]
    values = np.zeros((len(CONDITIONS), len(QNN_METHODS)), dtype=float)
    annotations: list[list[str]] = []
    for condition_index, condition in enumerate(CONDITIONS):
        row_annotations: list[str] = []
        for method_index, method in enumerate(QNN_METHODS):
            record = next(
                record
                for record in records
                if record["condition_id"] == condition.condition_id and record["method"] == method
            )
            values[condition_index, method_index] = float(record["mean_delta"])
            row_annotations.append(
                f"{record['mean_delta']:+.3f}\np={record['p_value']:.3f} {significance_marker(record['p_value'])}"
            )
        annotations.append(row_annotations)

    cmap = LinearSegmentedColormap.from_list(
        "delta_map",
        ["#B74848", "#F7F7F7", "#1D8A78"],
    )
    norm = TwoSlopeNorm(vcenter=0.0, vmin=float(np.min(values)) - 0.02, vmax=float(np.max(values)) + 0.02)
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            ax.text(
                col_index,
                row_index,
                annotations[row_index][col_index],
                ha="center",
                va="center",
                fontsize=7.7,
                color=TEXT_DARK,
            )
    ax.set_xticks(np.arange(len(QNN_METHODS)))
    ax.set_xticklabels([QNN_LABELS[method] for method in QNN_METHODS])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("QNN-control deltas")
    ax.set_xlabel("Quantum head")
    ax.set_ylabel("Boundary condition")
    plt.colorbar(image, ax=ax, fraction=0.05, pad=0.03, label="Macro-F1 delta vs control")


def plot_transport_shot_trajectory(ax: plt.Axes) -> None:
    """Plot the shot trajectory for the main transport subset."""
    shot_roots = {
        16: TRANSPORT_MAIN_ROOT,
        32: TRANSPORT_SHOT32_ROOT,
        64: TRANSPORT_SHOT64_ROOT,
    }
    for method in ["hybrid_control", "hybrid_no_entanglement", "hybrid_zz_real_amplitudes"]:
        shots: list[int] = []
        means: list[float] = []
        stds: list[float] = []
        for shot, root in shot_roots.items():
            row = summary_row(summary_rows_for_root(root), method, shot)
            shots.append(shot)
            means.append(float(row["macro_f1_mean"]))
            stds.append(float(row["macro_f1_std"]))
        ax.errorbar(
            shots,
            means,
            yerr=stds,
            marker="o",
            markersize=6.0,
            linewidth=1.9,
            capsize=3.2,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

    ax.axvspan(14, 18, color="#F2F7F6", alpha=0.8, zorder=0)
    ax.text(
        16,
        0.28,
        "16-shot window",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#1D8A78",
    )
    ax.set_xlim(14, 66)
    ax.set_ylim(0.22, 0.72)
    ax.set_xticks([16, 32, 64])
    ax.set_xlabel("Shot")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Label-count trajectory")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="lower right")


def plot_bootstrap_forest(ax: plt.Axes, records: list[dict[str, Any]]) -> None:
    """Plot bootstrap confidence intervals across conditions."""
    plot_records: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for method in QNN_METHODS:
            record = next(
                record
                for record in records
                if record["condition_id"] == condition.condition_id and record["method"] == method
            )
            plot_records.append(record)

    y_positions = np.arange(len(plot_records), dtype=float)
    for y_position, record in zip(y_positions, plot_records):
        color = METHOD_COLORS[record["method"]]
        mean_delta = float(record["mean_delta"])
        ci_low = float(record["bootstrap_ci_low"])
        ci_high = float(record["bootstrap_ci_high"])
        ax.plot([ci_low, ci_high], [y_position, y_position], color=color, linewidth=2.2, solid_capstyle="round")
        ax.scatter([mean_delta], [y_position], color=color, s=40, zorder=3)

    ax.axvline(0.0, color="#444444", linewidth=1.0)
    ax.set_yticks(y_positions)
    condition_labels = {
        "transport16_smallcnn": "T16 small",
        "transport32_smallcnn": "T32 small",
        "transport64_smallcnn": "T64 small",
        "transport16_resnet18": "T16 R18",
        "urban16_smallcnn": "U16 small",
    }
    ax.set_yticklabels(
        [
            f"{condition_labels[record['condition_id']]} / {QNN_LABELS[record['method']]}"
            for record in plot_records
        ]
    )
    ax.set_xlabel("Bootstrap 95% CI of macro-F1 delta vs control")
    ax.set_title("Bootstrap intervals")
    ax.grid(True, axis="x")


def plot_efficiency_tradeoff(ax: plt.Axes, records: list[dict[str, Any]]) -> None:
    """Plot delta vs runtime multiplier across boundary conditions."""
    condition_labels = {
        "transport16_smallcnn": "trans-16",
        "transport32_smallcnn": "trans-32",
        "transport64_smallcnn": "trans-64",
        "transport16_resnet18": "R18-16",
        "urban16_smallcnn": "urban-16",
    }
    label_offsets = {
        ("transport16_smallcnn", "hybrid_no_entanglement"): (0.06, 0.006, "left"),
        ("transport16_smallcnn", "hybrid_zz_real_amplitudes"): (0.06, 0.000, "left"),
        ("transport32_smallcnn", "hybrid_no_entanglement"): (0.06, 0.012, "left"),
        ("transport32_smallcnn", "hybrid_zz_real_amplitudes"): (0.06, 0.016, "left"),
        ("transport64_smallcnn", "hybrid_no_entanglement"): (0.06, -0.014, "left"),
        ("transport64_smallcnn", "hybrid_zz_real_amplitudes"): (-0.10, -0.016, "right"),
        ("transport16_resnet18", "hybrid_no_entanglement"): (0.06, -0.017, "left"),
        ("transport16_resnet18", "hybrid_zz_real_amplitudes"): (0.06, 0.014, "left"),
        ("urban16_smallcnn", "hybrid_no_entanglement"): (0.06, -0.022, "left"),
        ("urban16_smallcnn", "hybrid_zz_real_amplitudes"): (0.06, -0.002, "left"),
    }
    for method in QNN_METHODS:
        method_records = [record for record in records if record["method"] == method]
        ax.scatter(
            [float(record["runtime_ratio_vs_control"]) for record in method_records],
            [float(record["mean_delta"]) for record in method_records],
            color=METHOD_COLORS[method],
            s=58,
            label=QNN_LABELS[method],
            zorder=3,
        )
        for record in method_records:
            delta_x, delta_y, horizontal_alignment = label_offsets[
                (record["condition_id"], record["method"])
            ]
            ax.text(
                float(record["runtime_ratio_vs_control"]) + delta_x,
                float(record["mean_delta"]) + delta_y,
                condition_labels[record["condition_id"]],
                fontsize=6.5,
                color=TEXT_DARK,
                ha=horizontal_alignment,
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5},
                clip_on=False,
            )
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.axvline(1.0, color="#9CA7B2", linewidth=1.0, linestyle="--")
    ax.text(1.02, -0.085, "control", fontsize=7.6, color=TEXT_MUTED, ha="left", va="bottom")
    ax.set_xlim(0.72, 7.82)
    ax.set_ylim(-0.092, 0.215)
    ax.set_xlabel("Runtime multiplier vs control")
    ax.set_ylabel("Macro-F1 delta vs control")
    ax.set_title("Runtime tradeoff")
    ax.grid(True)
    ax.legend(frameon=False, loc="upper right")


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    """Save one figure to PNG and PDF and return the paths."""
    ensure_dir(output_dir)
    paths = [
        output_dir / f"{stem}.png",
        output_dir / f"{stem}.pdf",
    ]
    for path in paths:
        figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return paths


def draw_figure_1(output_dir: Path) -> list[Path]:
    """Generate the study-design overview figure."""
    configure_style()
    figure = plt.figure(figsize=(14.6, 9.8))
    outer = gridspec.GridSpec(
        2,
        2,
        figure=figure,
        height_ratios=[1.0, 0.84],
        width_ratios=[1.0, 1.0],
        hspace=0.18,
        wspace=0.12,
    )

    transport_counts = split_counts(TRANSPORT_DATASET_ROOT)
    urban_counts = split_counts(URBAN_DATASET_ROOT)

    draw_subset_montage(
        figure,
        outer[0, 0],
        dataset_root=TRANSPORT_DATASET_ROOT,
        class_order=["airplane", "ship", "vehicle", "harbor"],
        title="Transport main subset",
        counts=transport_counts,
        panel_label="a",
    )
    draw_subset_montage(
        figure,
        outer[0, 1],
        dataset_root=URBAN_DATASET_ROOT,
        class_order=["vehicle", "bridge", "harbor", "storagetank"],
        title="Urban boundary subset",
        counts=urban_counts,
        panel_label="b",
    )

    axis_pipeline = figure.add_subplot(outer[1, 0])
    add_panel_label(axis_pipeline, "c")
    draw_pipeline_schematic(axis_pipeline)

    axis_protocol = figure.add_subplot(outer[1, 1])
    add_panel_label(axis_protocol, "d")
    draw_boundary_protocol(axis_protocol)

    figure.suptitle(
        "Regime-bounded study design",
        fontsize=12.5,
        fontweight="bold",
        y=0.985,
    )
    return save_figure(figure, output_dir, "fig1_study_design_overview")


def draw_figure_2(output_dir: Path) -> list[Path]:
    """Generate the main low-shot evidence composite figure."""
    configure_style()
    figure = plt.figure(figsize=(14.8, 10.4))
    outer = gridspec.GridSpec(3, 2, figure=figure, hspace=0.30, wspace=0.20)

    detail_rows = detail_rows_for_root(TRANSPORT_MAIN_ROOT)
    pairwise_rows = pairwise_rows_for_root(TRANSPORT_MAIN_ROOT)
    per_class_rows = load_csv_rows(per_class_table_for_root(TRANSPORT_MAIN_ROOT))

    axis_a = figure.add_subplot(outer[0, 0])
    add_panel_label(axis_a, "a")
    plot_seed_scatter_performance(axis_a, detail_rows, shot=16)

    axis_b = figure.add_subplot(outer[0, 1])
    add_panel_label(axis_b, "b")
    plot_paired_delta_panel(axis_b, detail_rows, pairwise_rows, shot=16)

    axis_c = figure.add_subplot(outer[1, 0])
    add_panel_label(axis_c, "c")
    plot_training_curve_panel(axis_c, TRANSPORT_MAIN_ROOT, shot=16)

    axis_d = figure.add_subplot(outer[1, 1])
    add_panel_label(axis_d, "d")
    plot_per_class_panel(axis_d, per_class_rows, shot=16)

    class_names, control_confusion = mean_confusion_matrix(TRANSPORT_MAIN_ROOT, "hybrid_control", 16)
    _, no_ent_confusion = mean_confusion_matrix(TRANSPORT_MAIN_ROOT, "hybrid_no_entanglement", 16)

    axis_e = figure.add_subplot(outer[2, 0])
    add_panel_label(axis_e, "e")
    plot_confusion_heatmap(axis_e, control_confusion, class_names, "Mean confusion: classical control")

    axis_f = figure.add_subplot(outer[2, 1])
    add_panel_label(axis_f, "f")
    image = plot_confusion_heatmap(axis_f, no_ent_confusion, class_names, "Mean confusion: QNN no-ent")
    figure.colorbar(image, ax=[axis_e, axis_f], fraction=0.014, pad=0.015, label="Row-normalized rate")

    figure.suptitle(
        "Main 16-shot evidence on transport_logistics4",
        fontsize=12.5,
        fontweight="bold",
        y=0.985,
    )
    return save_figure(figure, output_dir, "fig2_main_lowshot_evidence")


def draw_figure_3(output_dir: Path) -> list[Path]:
    """Generate the regime-boundary composite figure."""
    configure_style()
    figure = plt.figure(figsize=(14.8, 9.6))
    outer = gridspec.GridSpec(2, 2, figure=figure, hspace=0.28, wspace=0.22)

    boundary_records = build_boundary_records()

    axis_a = figure.add_subplot(outer[0, 0])
    add_panel_label(axis_a, "a")
    plot_boundary_heatmap(axis_a, boundary_records)

    axis_b = figure.add_subplot(outer[0, 1])
    add_panel_label(axis_b, "b")
    plot_transport_shot_trajectory(axis_b)

    axis_c = figure.add_subplot(outer[1, 0])
    add_panel_label(axis_c, "c")
    plot_bootstrap_forest(axis_c, boundary_records)

    axis_d = figure.add_subplot(outer[1, 1])
    add_panel_label(axis_d, "d")
    plot_efficiency_tradeoff(axis_d, boundary_records)

    figure.suptitle(
        "Boundary map of regime-limited quantum gains",
        fontsize=12.5,
        fontweight="bold",
        y=0.985,
    )
    return save_figure(figure, output_dir, "fig3_boundary_regime_map")


def write_manifest(output_dir: Path, figure_paths: list[Path]) -> Path:
    """Write a small manifest to help manuscript integration."""
    manifest_path = output_dir / "figure_manifest.md"
    content = f"""# Composite Figure Manifest

Generated figures:

- `{figure_paths[0].name}` / `{figure_paths[1].name}`: study design overview with subset montages, pipeline schematic, and boundary-analysis map.
- `{figure_paths[2].name}` / `{figure_paths[3].name}`: main 16-shot evidence with seed-level results, paired deltas, optimization curves, per-class F1, and confusion matrices.
- `{figure_paths[4].name}` / `{figure_paths[5].name}`: regime-boundary figure with cross-condition delta heatmap, shot trajectory, bootstrap forest, and runtime-vs-gain tradeoff.

Primary sources:

- `data/processed/subset_candidates/transport_logistics4`
- `data/processed/subset_candidates/urban_structural4`
- `artifacts/dior_subset_candidate_screening/transport_logistics4_shot16_bblr01_7seed`
- `artifacts/dior_subset_candidate_screening/transport_logistics4_shot32_bblr01_7seed`
- `artifacts/dior_subset_candidate_screening/transport_logistics4_shot64_bblr01_7seed`
- `artifacts/transport_logistics4_resnet18_shot16_bblr01_7seed`
- `artifacts/dior_subset_candidate_screening/urban_structural4_shot16_bblr01_7seed`
"""
    manifest_path.write_text(content, encoding="utf-8")
    return manifest_path


def main() -> None:
    """Run the composite figure pipeline."""
    args = parse_args()
    output_dir = args.output_dir
    ensure_dir(output_dir)

    figure_paths: list[Path] = []
    figure_paths.extend(draw_figure_1(output_dir))
    figure_paths.extend(draw_figure_2(output_dir))
    figure_paths.extend(draw_figure_3(output_dir))
    manifest_path = write_manifest(output_dir, figure_paths)

    print("Generated composite figures:")
    for path in figure_paths:
        print(f" - {path}")
    print(f" - {manifest_path}")


if __name__ == "__main__":
    main()

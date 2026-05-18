#!/usr/bin/env python3
"""Generate paper-ready tables and figures for the DIOR low-regime study."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.stats import wilcoxon

    SCIPY_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    wilcoxon = None  # type: ignore[assignment]
    SCIPY_IMPORT_ERROR = exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "dior_subset_candidate_screening"
    / "transport_logistics4_shot16_bblr01_7seed"
)

METHOD_ORDER = [
    "baseline_cnn",
    "hybrid_control",
    "hybrid_no_entanglement",
    "hybrid_zz_real_amplitudes",
]
METHOD_LABELS = {
    "baseline_cnn": "CNN Baseline",
    "hybrid_control": "Classical Control",
    "hybrid_no_entanglement": "QNN No-Ent",
    "hybrid_zz_real_amplitudes": "QNN ZZ+RealAmp",
}
METHOD_COLORS = {
    "baseline_cnn": "#7A7A7A",
    "hybrid_control": "#2F6B9A",
    "hybrid_no_entanglement": "#2A9D8F",
    "hybrid_zz_real_amplitudes": "#E0A526",
}
PAIRWISE_METHODS = [
    ("hybrid_control", "hybrid_no_entanglement"),
    ("hybrid_control", "hybrid_zz_real_amplitudes"),
    ("hybrid_no_entanglement", "hybrid_zz_real_amplitudes"),
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate DIOR low-regime paper tables and figures."
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help=f"Experiment artifact root (default: {DEFAULT_EXPERIMENT_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated tables/figures. Defaults to <experiment-root>/paper_ready.",
    )
    return parser.parse_args()


def configure_style() -> None:
    """Use one clean white-background academic plotting style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows into dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def require_file(path: Path) -> None:
    """Fail fast when an expected input file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def to_float(value: str | None) -> float | None:
    """Convert a CSV field into float when possible."""
    if value is None or value == "":
        return None
    return float(value)


def ensure_dir(path: Path) -> None:
    """Create a directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to CSV."""
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_mean_std(mean_value: float | None, std_value: float | None, digits: int = 4) -> str:
    """Format one mean±std cell."""
    if mean_value is None:
        return ""
    if std_value is None:
        return f"{mean_value:.{digits}f}"
    return f"{mean_value:.{digits}f} ± {std_value:.{digits}f}"


def method_rank(rows: list[dict[str, str]], shot: int, method: str) -> int:
    """Compute one method rank within a shot using macro-F1 mean."""
    shot_rows = [row for row in rows if int(row["shot"]) == shot]
    sorted_rows = sorted(
        shot_rows,
        key=lambda row: float(row["macro_f1_mean"] or 0.0),
        reverse=True,
    )
    for index, row in enumerate(sorted_rows, start=1):
        if row["method"] == method:
            return index
    raise KeyError(f"Method {method!r} not found for shot {shot}")


def build_main_results_table(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build the main paper table from aggregated summary rows."""
    rows: list[dict[str, Any]] = []
    for shot in sorted({int(row["shot"]) for row in summary_rows}):
        shot_rows = [row for row in summary_rows if int(row["shot"]) == shot]
        for method in METHOD_ORDER:
            row = next((item for item in shot_rows if item["method"] == method), None)
            if row is None:
                continue
            macro_mean = to_float(row.get("macro_f1_mean"))
            macro_std = to_float(row.get("macro_f1_std"))
            acc_mean = to_float(row.get("accuracy_mean"))
            acc_std = to_float(row.get("accuracy_std"))
            runtime_mean = to_float(row.get("runtime_seconds_mean"))
            runtime_std = to_float(row.get("runtime_seconds_std"))
            rows.append(
                {
                    "shot": shot,
                    "rank": method_rank(summary_rows, shot, method),
                    "method": method,
                    "display_name": METHOD_LABELS.get(method, method),
                    "macro_f1_mean_std": format_mean_std(macro_mean, macro_std),
                    "accuracy_mean_std": format_mean_std(acc_mean, acc_std),
                    "runtime_seconds_mean_std": format_mean_std(runtime_mean, runtime_std, digits=2),
                    "macro_f1_mean": macro_mean,
                    "macro_f1_std": macro_std,
                    "accuracy_mean": acc_mean,
                    "runtime_seconds_mean": runtime_mean,
                }
            )
    return rows


def build_pairwise_table(detail_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Build paired comparison rows from per-seed detail results."""
    if SCIPY_IMPORT_ERROR is not None:
        return []

    rows: list[dict[str, Any]] = []
    shots = sorted({int(row["shot"]) for row in detail_rows})
    seeds = sorted({int(row["seed"]) for row in detail_rows})

    for shot in shots:
        for left_method, right_method in PAIRWISE_METHODS:
            left_scores: list[float] = []
            right_scores: list[float] = []
            for seed in seeds:
                left_row = next(
                    (
                        row
                        for row in detail_rows
                        if int(row["shot"]) == shot
                        and int(row["seed"]) == seed
                        and row["method"] == left_method
                    ),
                    None,
                )
                right_row = next(
                    (
                        row
                        for row in detail_rows
                        if int(row["shot"]) == shot
                        and int(row["seed"]) == seed
                        and row["method"] == right_method
                    ),
                    None,
                )
                if left_row is None or right_row is None:
                    continue
                left_scores.append(float(left_row["macro_f1"]))
                right_scores.append(float(right_row["macro_f1"]))

            if not left_scores or len(left_scores) != len(right_scores):
                continue

            deltas = [right - left for left, right in zip(left_scores, right_scores)]
            better_count = sum(1 for delta in deltas if delta > 0)
            tie_count = sum(1 for delta in deltas if math.isclose(delta, 0.0, abs_tol=1e-12))
            worse_count = sum(1 for delta in deltas if delta < 0)

            p_value: float | None = None
            try:
                if any(not math.isclose(delta, 0.0, abs_tol=1e-12) for delta in deltas):
                    p_value = float(wilcoxon(left_scores, right_scores, alternative="two-sided").pvalue)
            except Exception:
                p_value = None

            rows.append(
                {
                    "shot": shot,
                    "left_method": left_method,
                    "left_display_name": METHOD_LABELS.get(left_method, left_method),
                    "right_method": right_method,
                    "right_display_name": METHOD_LABELS.get(right_method, right_method),
                    "num_pairs": len(deltas),
                    "mean_delta_macro_f1": float(np.mean(deltas)),
                    "std_delta_macro_f1": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                    "better_count": better_count,
                    "tie_count": tie_count,
                    "worse_count": worse_count,
                    "wilcoxon_p_value": p_value,
                }
            )
    return rows


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save one figure and close it."""
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_macro_f1_vs_shot(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    """Plot macro-F1 vs shot with mean±std error bars."""
    shots = sorted({int(row["shot"]) for row in summary_rows})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.grid(True, axis="y", alpha=0.6)

    for method in METHOD_ORDER:
        method_rows = {
            int(row["shot"]): row
            for row in summary_rows
            if row["method"] == method
        }
        x_values = [shot for shot in shots if shot in method_rows]
        y_values = [float(method_rows[shot]["macro_f1_mean"]) for shot in x_values]
        y_errors = [float(method_rows[shot]["macro_f1_std"]) for shot in x_values]
        ax.errorbar(
            x_values,
            y_values,
            yerr=y_errors,
            marker="o",
            linewidth=2.0,
            capsize=3.0,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "#333333"),
        )

    ax.set_title("Low-shot DIOR ROI classification")
    ax.set_xlabel("Train samples per class (shot)")
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(shots)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    save_figure(fig, output_path)


def plot_runtime_vs_method(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    """Plot runtime comparison with log-scale grouped bars."""
    shots = sorted({int(row["shot"]) for row in summary_rows})
    methods = [method for method in METHOD_ORDER if any(row["method"] == method for row in summary_rows)]
    x = np.arange(len(methods))
    width = 0.35 if len(shots) <= 2 else max(0.18, 0.8 / max(1, len(shots)))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for index, shot in enumerate(shots):
        shot_rows = {
            row["method"]: row
            for row in summary_rows
            if int(row["shot"]) == shot
        }
        offsets = x + (index - (len(shots) - 1) / 2) * width
        values = [float(shot_rows[method]["runtime_seconds_mean"]) for method in methods]
        ax.bar(
            offsets,
            values,
            width=width,
            label=f"{shot}-shot",
            alpha=0.9,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Runtime (seconds, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method] for method in methods], rotation=18, ha="right")
    ax.set_title("Runtime cost across low-shot settings")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.5)
    save_figure(fig, output_path)


def plot_delta_vs_control(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    """Plot macro-F1 delta against the classical control baseline."""
    shots = sorted({int(row["shot"]) for row in summary_rows})
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.grid(True, axis="y", alpha=0.6)

    control_rows = {
        int(row["shot"]): row
        for row in summary_rows
        if row["method"] == "hybrid_control"
    }
    for method in ("hybrid_no_entanglement", "hybrid_zz_real_amplitudes"):
        method_rows = {
            int(row["shot"]): row
            for row in summary_rows
            if row["method"] == method
        }
        x_values = [shot for shot in shots if shot in method_rows and shot in control_rows]
        y_values = [
            float(method_rows[shot]["macro_f1_mean"]) - float(control_rows[shot]["macro_f1_mean"])
            for shot in x_values
        ]
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS.get(method, "#333333"),
            label=f"{METHOD_LABELS[method]} - {METHOD_LABELS['hybrid_control']}",
        )

    ax.set_xlabel("Train samples per class (shot)")
    ax.set_ylabel("Macro-F1 delta vs classical control")
    ax.set_title("Where the QNN branch still helps")
    ax.set_xticks(shots)
    ax.legend(frameon=False)
    save_figure(fig, output_path)


def plot_pareto(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    """Plot runtime/performance trade-off scatter."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.grid(True, alpha=0.5)

    for row in summary_rows:
        method = row["method"]
        shot = int(row["shot"])
        runtime = float(row["runtime_seconds_mean"])
        macro_f1 = float(row["macro_f1_mean"])
        ax.scatter(
            runtime,
            macro_f1,
            s=80,
            color=METHOD_COLORS.get(method, "#333333"),
            alpha=0.9,
        )
        ax.annotate(
            f"{METHOD_LABELS.get(method, method)}\n{shot}-shot",
            (runtime, macro_f1),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Runtime (seconds, log scale)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Performance vs runtime trade-off")
    save_figure(fig, output_path)


def main() -> int:
    """Script entry point."""
    args = parse_args()
    experiment_root = args.experiment_root
    output_dir = args.output_dir or (experiment_root / "paper_ready")
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    summary_csv = experiment_root / "lowshot_summary_rows.csv"
    detail_csv = experiment_root / "lowshot_detail_rows.csv"
    require_file(summary_csv)
    require_file(detail_csv)

    configure_style()
    summary_rows = load_csv_rows(summary_csv)
    detail_rows = load_csv_rows(detail_csv)

    main_table_rows = build_main_results_table(summary_rows)
    pairwise_rows = build_pairwise_table(detail_rows)

    write_csv(tables_dir / "tab_dior_lowregime_main_results.csv", main_table_rows)
    write_csv(tables_dir / "tab_dior_lowregime_pairwise_tests.csv", pairwise_rows)

    plot_macro_f1_vs_shot(summary_rows, figures_dir / "fig_dior_lowregime_macro_f1_vs_shot.png")
    plot_runtime_vs_method(summary_rows, figures_dir / "fig_dior_lowregime_runtime_comparison.png")
    plot_delta_vs_control(summary_rows, figures_dir / "fig_dior_lowregime_delta_vs_control.png")
    plot_pareto(summary_rows, figures_dir / "fig_dior_lowregime_pareto_runtime_vs_macro_f1.png")

    print(f"Saved tables to: {tables_dir}")
    print(f"Saved figures to: {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

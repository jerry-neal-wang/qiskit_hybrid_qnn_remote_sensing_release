#!/usr/bin/env python3
"""Generate the GRSL regime-boundary evidence map and supplementary figures."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "paper_composite_figures"
EN_FIGURES = PROJECT_ROOT / "GRSL_overleaf_submission_final_package" / "figures"
CN_FIGURES = PROJECT_ROOT / "GRSL_overleaf_submission_final_CN_package" / "figures"
EN_SUPP = PROJECT_ROOT / "GRSL_overleaf_submission_final_package" / "supplementary"
CN_SUPP = PROJECT_ROOT / "GRSL_overleaf_submission_final_CN_package" / "supplementary"

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
TRANSPORT_RESNET_ROOT = (
    PROJECT_ROOT / "artifacts" / "transport_logistics4_resnet18_shot16_bblr01_7seed"
)
URBAN_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "dior_subset_candidate_screening"
    / "urban_structural4_shot16_bblr01_7seed"
)

TRANSPORT_DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "subset_candidates" / "transport_logistics4"
URBAN_DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "subset_candidates" / "urban_structural4"
BOOTSTRAP_CI_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "transport_logistics4_boundary_bundle"
    / "tables"
    / "tab_transport_logistics4_bootstrap_delta_ci.csv"
)

QNN_METHODS = ("hybrid_no_entanglement", "hybrid_zz_real_amplitudes")
METHOD_LABELS = {
    "hybrid_no_entanglement": "No-ent",
    "hybrid_zz_real_amplitudes": "ZZ+RealAmp",
}
METHOD_COLORS = {
    "hybrid_no_entanglement": "#168579",
    "hybrid_zz_real_amplitudes": "#C88A00",
}
METHOD_MARKERS = {
    "hybrid_no_entanglement": "o",
    "hybrid_zz_real_amplitudes": "D",
}
CLASS_ORDER = ("airplane", "ship", "vehicle", "harbor")


@dataclass(frozen=True)
class Regime:
    regime_id: str
    label: str
    shot: int
    root: Path


REGIMES = (
    Regime("tl4_small_16", "TL4 / small CNN / 16-shot", 16, TRANSPORT_MAIN_ROOT),
    Regime("tl4_small_32", "TL4 / small CNN / 32-shot", 32, TRANSPORT_SHOT32_ROOT),
    Regime("tl4_small_64", "TL4 / small CNN / 64-shot", 64, TRANSPORT_SHOT64_ROOT),
    Regime("tl4_resnet18_16", "TL4 / ResNet18 / 16-shot", 16, TRANSPORT_RESNET_ROOT),
    Regime("urban4_small_16", "Urban4 / small CNN / 16-shot", 16, URBAN_ROOT),
)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.7,
            "axes.titlesize": 9.4,
            "axes.labelsize": 9.1,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 7.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def table_dir_for_root(root: Path) -> Path:
    for candidate in (root / "paper_ready" / "tables", root / "figures" / "tables"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find paper table directory under {root}")


def per_class_table_for_root(root: Path) -> Path:
    path = root / "paper_ready" / "per_class" / "tables" / "tab_per_class_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find per-class table under {root}")
    return path


def summary_rows_for_root(root: Path) -> list[dict[str, str]]:
    return load_csv_rows(root / "lowshot_summary_rows.csv")


def detail_rows_for_root(root: Path) -> list[dict[str, str]]:
    return load_csv_rows(root / "lowshot_detail_rows.csv")


def pairwise_rows_for_root(root: Path) -> list[dict[str, str]]:
    return load_csv_rows(table_dir_for_root(root) / "tab_dior_lowregime_pairwise_tests.csv")


def summary_row(rows: list[dict[str, str]], method: str, shot: int) -> dict[str, str]:
    for row in rows:
        if row["method"] == method and int(row["shot"]) == shot:
            return row
    raise KeyError(f"Missing summary row for {method}, shot={shot}")


def pairwise_row(rows: list[dict[str, str]], right_method: str, shot: int) -> dict[str, str]:
    for row in rows:
        if (
            row["left_method"] == "hybrid_control"
            and row["right_method"] == right_method
            and int(row["shot"]) == shot
        ):
            return row
    raise KeyError(f"Missing control-vs-{right_method} row for shot={shot}")


def seed_scores(rows: list[dict[str, str]], method: str, shot: int) -> dict[int, float]:
    return {
        int(row["seed"]): float(row["macro_f1"])
        for row in rows
        if row["method"] == method and int(row["shot"]) == shot
    }


def seed_deltas(rows: list[dict[str, str]], method: str, shot: int) -> list[float]:
    control = seed_scores(rows, "hybrid_control", shot)
    qnn = seed_scores(rows, method, shot)
    seeds = sorted(set(control) & set(qnn))
    return [qnn[seed] - control[seed] for seed in seeds]


def bootstrap_ci(values: list[float], seed: int, samples: int = 20000) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    sampled = rng.choice(array, size=(samples, array.size), replace=True)
    means = sampled.mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def precomputed_bootstrap_ci(regime_id: str, method: str) -> tuple[float, float] | None:
    """Return fixed CIs used in the paper tables when available."""
    if regime_id not in {"tl4_small_16", "tl4_small_64"} or not BOOTSTRAP_CI_PATH.exists():
        return None
    comparison = {
        "hybrid_no_entanglement": "no_ent-control",
        "hybrid_zz_real_amplitudes": "zz-control",
    }[method]
    shot = 16 if regime_id == "tl4_small_16" else 64
    for row in load_csv_rows(BOOTSTRAP_CI_PATH):
        if int(row["shot"]) == shot and row["comparison"] == comparison:
            return (float(row["bootstrap_ci95_low"]), float(row["bootstrap_ci95_high"]))
    return None


def per_class_delta(root: Path, shot: int, method: str, class_name: str) -> float | None:
    rows = load_csv_rows(per_class_table_for_root(root))
    control_values = {
        row["class_name"]: float(row["f1_mean"])
        for row in rows
        if row["method"] == "hybrid_control" and int(row["shot"]) == shot
    }
    method_values = {
        row["class_name"]: float(row["f1_mean"])
        for row in rows
        if row["method"] == method and int(row["shot"]) == shot
    }
    if class_name not in control_values or class_name not in method_values:
        return None
    return method_values[class_name] - control_values[class_name]


def collect_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for regime in REGIMES:
        summaries = summary_rows_for_root(regime.root)
        details = detail_rows_for_root(regime.root)
        pairwise = pairwise_rows_for_root(regime.root)
        control_runtime = float(summary_row(summaries, "hybrid_control", regime.shot)["runtime_seconds_mean"])
        for method in QNN_METHODS:
            comparison = pairwise_row(pairwise, method, regime.shot)
            summary = summary_row(summaries, method, regime.shot)
            deltas = seed_deltas(details, method, regime.shot)
            seed_text = f"{regime.regime_id}:{method}"
            seed = sum((i + 1) * ord(ch) for i, ch in enumerate(seed_text)) % (2**32)
            ci_low, ci_high = bootstrap_ci(deltas, seed)
            fixed_ci = precomputed_bootstrap_ci(regime.regime_id, method)
            if fixed_ci is not None:
                ci_low, ci_high = fixed_ci
            records.append(
                {
                    "regime_id": regime.regime_id,
                    "regime_label": regime.label,
                    "shot": regime.shot,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "mean_delta": float(comparison["mean_delta_macro_f1"]),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "runtime_ratio": float(summary["runtime_seconds_mean"]) / control_runtime,
                    "p_value": float(comparison["wilcoxon_p_value"]),
                    "class_delta": {
                        class_name: per_class_delta(regime.root, regime.shot, method, class_name)
                        for class_name in CLASS_ORDER
                    },
                }
            )
    return records


def marker_size(runtime_ratio: float) -> float:
    return 34.0 + 19.0 * runtime_ratio


def write_record_table(records: list[dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "regime": record["regime_label"],
            "method": record["method_label"],
            "mean_delta_macro_f1": record["mean_delta"],
            "bootstrap_ci_low": record["ci_low"],
            "bootstrap_ci_high": record["ci_high"],
            "runtime_ratio_vs_control": record["runtime_ratio"],
            "wilcoxon_p_value": record["p_value"],
        }
        for class_name in CLASS_ORDER:
            row[f"{class_name}_delta_f1"] = record["class_delta"][class_name]
        rows.append(row)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_supplementary_table_s1(records: list[dict[str, Any]]) -> list[Path]:
    """Write the detailed comparison table referenced as Supplementary Table S1."""
    ensure_dir(EN_SUPP)
    ensure_dir(CN_SUPP)

    rows: list[dict[str, Any]] = []
    for regime in REGIMES:
        summaries = summary_rows_for_root(regime.root)
        control = summary_row(summaries, "hybrid_control", regime.shot)
        row: dict[str, Any] = {
            "regime": regime.label,
            "control_macro_f1_mean_std": f"{float(control['macro_f1_mean']):.4f} +/- {float(control['macro_f1_std']):.4f}",
        }
        for method in QNN_METHODS:
            summary = summary_row(summaries, method, regime.shot)
            record = next(
                item
                for item in records
                if item["regime_id"] == regime.regime_id and item["method"] == method
            )
            pairwise = pairwise_row(pairwise_rows_for_root(regime.root), method, regime.shot)
            prefix = METHOD_LABELS[method].lower().replace("+", "").replace("-", "_").replace(" ", "_")
            row[f"{prefix}_macro_f1_mean_std"] = f"{float(summary['macro_f1_mean']):.4f} +/- {float(summary['macro_f1_std']):.4f}"
            row[f"{prefix}_delta"] = f"{record['mean_delta']:+.4f}"
            row[f"{prefix}_bootstrap_ci95"] = f"[{record['ci_low']:+.4f}, {record['ci_high']:+.4f}]"
            row[f"{prefix}_p_value"] = f"{record['p_value']:.5f}"
            row[f"{prefix}_sign_counts_better_tie_worse"] = (
                f"{pairwise['better_count']}/{pairwise['tie_count']}/{pairwise['worse_count']}"
            )
            row[f"{prefix}_runtime_ratio"] = f"{record['runtime_ratio']:.2f}x"
        rows.append(row)

    csv_path = EN_SUPP / "supp_table_s1_full_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tex_path = EN_SUPP / "supp_table_s1_full_statistics.tex"
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write("% Supplementary Table~S1. Full boundary statistics.\n")
        handle.write("\\begin{table*}[!t]\n\\centering\n\\scriptsize\n")
        handle.write("\\caption{Supplementary Table~S1. Full boundary statistics.}\n")
        handle.write("\\begin{tabular}{p{0.18\\textwidth}p{0.13\\textwidth}p{0.29\\textwidth}p{0.29\\textwidth}}\n")
        handle.write("\\toprule\nRegime & Control & No-ent & ZZ+RA \\\\\n\\midrule\n")
        for row in rows:
            handle.write(
                f"{row['regime']} & {row['control_macro_f1_mean_std']} & "
                f"{row['no_ent_macro_f1_mean_std']}; $\\Delta={row['no_ent_delta']}$; CI {row['no_ent_bootstrap_ci95']}; "
                f"$p={row['no_ent_p_value']}$; signs {row['no_ent_sign_counts_better_tie_worse']}; "
                f"$\\rho={row['no_ent_runtime_ratio']}$ & "
                f"{row['zzrealamp_macro_f1_mean_std']}; $\\Delta={row['zzrealamp_delta']}$; CI {row['zzrealamp_bootstrap_ci95']}; "
                f"$p={row['zzrealamp_p_value']}$; signs {row['zzrealamp_sign_counts_better_tie_worse']}; "
                f"$\\rho={row['zzrealamp_runtime_ratio']}$ \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    paths = [csv_path, tex_path]
    for source in paths:
        (CN_SUPP / source.name).write_bytes(source.read_bytes())
    return paths


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    ensure_dir(output_dir)
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    figure.savefig(paths[0], bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight", dpi=600)
    plt.close(figure)
    return paths


def copy_outputs(paths: list[Path], destinations: tuple[Path, ...]) -> None:
    for destination in destinations:
        ensure_dir(destination)
        for source in paths:
            target = destination / source.name
            target.write_bytes(source.read_bytes())


def draw_regime_boundary_map(records: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    configure_style()
    figure = plt.figure(figsize=(7.45, 4.75))
    grid = figure.add_gridspec(1, 2, width_ratios=[4.85, 1.25], wspace=0.035)
    ax = figure.add_subplot(grid[0, 0])
    heat_ax = figure.add_subplot(grid[0, 1], sharey=ax)

    base_y = {regime.regime_id: len(REGIMES) - 1 - index for index, regime in enumerate(REGIMES)}
    offset = {"hybrid_no_entanglement": 0.16, "hybrid_zz_real_amplitudes": -0.16}

    main_y = base_y["tl4_small_16"]
    for axis in (ax, heat_ax):
        axis.axhspan(main_y - 0.48, main_y + 0.48, color="#EAF5F1", zorder=0)

    ax.axvline(0.0, color="#222222", linewidth=1.05, zorder=1)
    ax.axvline(0.10, color="#666666", linewidth=1.0, linestyle=(0, (3.0, 2.0)), zorder=1)
    ax.text(0.103, main_y + 0.49, "+0.10", ha="left", va="bottom", fontsize=7.1, color="#555555")

    for record in records:
        y = base_y[record["regime_id"]] + offset[record["method"]]
        color = METHOD_COLORS[record["method"]]
        marker = METHOD_MARKERS[record["method"]]
        ax.plot(
            [record["ci_low"], record["ci_high"]],
            [y, y],
            color=color,
            linewidth=2.0,
            alpha=0.9,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            [record["mean_delta"]],
            [y],
            s=marker_size(record["runtime_ratio"]),
            marker=marker,
            color=color,
            edgecolors="#1F2933",
            linewidths=0.65,
            zorder=3,
        )

    ax.annotate(
        "main QCAD\nwindow",
        xy=(0.285, main_y),
        xytext=(0.345, main_y + 0.28),
        arrowprops={"arrowstyle": "-", "color": "#27826F", "lw": 1.1},
        ha="right",
        va="center",
        fontsize=7.7,
        color="#176D5F",
    )
    ax.text(
        0.352,
        base_y["tl4_small_32"] + 0.44,
        r"estimated practical" "\n" r"boundary $\approx$ 23 shots/class",
        ha="right",
        va="center",
        fontsize=7.35,
        color="#4A5560",
    )

    ax.set_xlim(-0.19, 0.36)
    ax.set_ylim(-0.55, len(REGIMES) - 0.45)
    ax.set_yticks([base_y[regime.regime_id] for regime in REGIMES])
    ax.set_yticklabels([regime.label for regime in REGIMES])
    ax.set_xlabel("Macro-F1 delta vs matched classical control")
    ax.set_title("Regime-boundary evidence map", fontsize=10.6, fontweight="bold", pad=9)
    ax.grid(True, axis="x", color="#D9DEE4", linewidth=0.7)
    ax.grid(True, axis="y", color="#EDF0F3", linewidth=0.55)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        ax.scatter([], [], s=78, marker=METHOD_MARKERS[method], color=METHOD_COLORS[method], edgecolors="#1F2933", linewidths=0.65)
        for method in QNN_METHODS
    ]
    size_handles = [
        ax.scatter([], [], s=marker_size(ratio), marker="o", color="white", edgecolors="#5B6770", linewidths=0.8)
        for ratio in (2.0, 5.0)
    ]
    first_legend = figure.legend(
        legend_handles,
        [METHOD_LABELS[method] for method in QNN_METHODS],
        loc="lower center",
        bbox_to_anchor=(0.44, 0.025),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        borderpad=0.0,
    )
    figure.add_artist(first_legend)
    figure.legend(
        size_handles,
        ["2x runtime", "5x runtime"],
        loc="lower center",
        bbox_to_anchor=(0.71, 0.025),
        ncol=2,
        frameon=False,
        handletextpad=0.45,
        borderpad=0.0,
    )

    cmap = LinearSegmentedColormap.from_list("qib_delta", ["#B04745", "#F7F7F7", "#177F70"])
    norm = TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.30)
    heat_ax.set_xlim(0, len(CLASS_ORDER))
    heat_ax.set_ylim(ax.get_ylim())
    heat_ax.set_xticks(np.arange(len(CLASS_ORDER)) + 0.5)
    heat_ax.set_xticklabels(["A", "S", "V", "H"])
    heat_ax.xaxis.tick_top()
    heat_ax.tick_params(axis="x", length=0, pad=2)
    heat_ax.tick_params(axis="y", left=False, labelleft=False)
    heat_ax.set_title("QIB strip\nclass delta", fontsize=8.3, pad=9)
    heat_ax.set_facecolor("white")
    for spine in heat_ax.spines.values():
        spine.set_visible(False)

    for record in records:
        y = base_y[record["regime_id"]] + offset[record["method"]]
        for class_index, class_name in enumerate(CLASS_ORDER):
            value = record["class_delta"][class_name]
            if value is None:
                facecolor = "#ECEFF2"
                edgecolor = "#C6CCD3"
                label = ""
            else:
                facecolor = cmap(norm(value))
                edgecolor = "white"
                label = "+" if value > 0.035 else ("-" if value < -0.035 else "")
            heat_ax.add_patch(
                Rectangle(
                    (class_index + 0.07, y - 0.105),
                    0.86,
                    0.21,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=0.45,
                    zorder=2,
                )
            )
            if label:
                heat_ax.text(
                    class_index + 0.5,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.7,
                    color="white" if abs(value or 0.0) > 0.10 else "#1F2933",
                    zorder=3,
                )

    heat_ax.text(
        0.0,
        -0.46,
        "A=airplane  S=ship\nV=vehicle  H=harbor; gray=N/A",
        ha="left",
        va="top",
        fontsize=6.15,
        color="#4A5560",
        transform=heat_ax.transData,
    )

    figure.subplots_adjust(left=0.275, right=0.985, top=0.84, bottom=0.22)
    paths = save_figure(figure, output_dir, "fig_regime_boundary_evidence_map")
    copy_outputs(paths, (EN_FIGURES, CN_FIGURES))
    return paths


def draw_framework_schematic(output_dir: Path) -> list[Path]:
    """Draw a compact study-design and metric-framework schematic."""
    configure_style()
    figure, ax = plt.subplots(figsize=(3.55, 1.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def block(
        xy: tuple[float, float],
        width: float,
        height: float,
        text: str,
        *,
        facecolor: str = "#F5F7FA",
        edgecolor: str = "#34495E",
        fontsize: float = 6.8,
        weight: str = "normal",
    ) -> None:
        rect = Rectangle(
            xy,
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.8,
            joinstyle="round",
        )
        ax.add_patch(rect)
        ax.text(
            xy[0] + width / 2,
            xy[1] + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#1F2933",
            fontweight=weight,
        )

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 0.8,
                "color": "#53616D",
                "shrinkA": 1.5,
                "shrinkB": 1.5,
            },
        )

    block((0.03, 0.74), 0.18, 0.15, "DIOR ROI\ncrops", facecolor="#E8F1F7", weight="bold")
    block((0.28, 0.74), 0.20, 0.15, "Low-shot\nsplit", facecolor="#E8F1F7", weight="bold")
    block((0.56, 0.74), 0.39, 0.15, "Matched protocol\nsplit / backbone / optimizer", facecolor="#E8F1F7", fontsize=6.1)
    arrow((0.21, 0.815), (0.28, 0.815))
    arrow((0.48, 0.815), (0.56, 0.815))

    block((0.05, 0.46), 0.24, 0.13, "Classical\ncontrol", facecolor="#EEF3F8", fontsize=6.4)
    block((0.38, 0.46), 0.24, 0.13, "No-ent\nQNN", facecolor="#E8F5F1", fontsize=6.4)
    block((0.71, 0.46), 0.24, 0.13, "ZZ+RA\nQNN", facecolor="#FFF5DD", fontsize=6.4)
    for x in (0.17, 0.50, 0.83):
        arrow((0.755, 0.74), (x, 0.59))

    block((0.06, 0.12), 0.34, 0.16, "Metrics\nQCAD / LSEC / QIB", facecolor="#F4F0FA", fontsize=6.5, weight="bold")
    block((0.55, 0.12), 0.39, 0.16, "Stress tests\nshot / backbone / subset\nruntime", facecolor="#F7F2E8", fontsize=6.15, weight="bold")
    arrow((0.50, 0.46), (0.24, 0.28))
    arrow((0.50, 0.46), (0.745, 0.28))
    figure.tight_layout(pad=0.18)
    paths = save_figure(figure, output_dir, "fig_metric_framework_schematic")
    copy_outputs(paths, (EN_FIGURES, CN_FIGURES))
    return paths


def sample_image(dataset_root: Path, class_name: str) -> np.ndarray:
    candidates = sorted((dataset_root / "test" / class_name).glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"No test images for {class_name} in {dataset_root}")
    path = candidates[len(candidates) // 2]
    with Image.open(path) as image:
        image = image.convert("RGB").resize((120, 120))
        return np.asarray(image)


def draw_supplement_roi_examples(output_dir: Path) -> list[Path]:
    configure_style()
    tl_classes = ("airplane", "ship", "vehicle", "harbor")
    urban_classes = ("vehicle", "bridge", "harbor", "storagetank")
    figure, axes = plt.subplots(2, 4, figsize=(7.2, 3.55))
    for col, class_name in enumerate(tl_classes):
        axes[0, col].imshow(sample_image(TRANSPORT_DATASET_ROOT, class_name))
        axes[0, col].set_title(class_name, fontsize=8.4)
    for col, class_name in enumerate(urban_classes):
        axes[1, col].imshow(sample_image(URBAN_DATASET_ROOT, class_name))
        axes[1, col].set_title(class_name, fontsize=8.4)
    axes[0, 0].set_ylabel("TL4", fontsize=9.0)
    axes[1, 0].set_ylabel("Urban4", fontsize=9.0)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#FFFFFF")
    figure.suptitle("Supplementary Fig. S1. ROI examples", fontsize=10.5, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    paths = save_figure(figure, output_dir, "supp_fig_s1_roi_examples")
    copy_outputs(paths, (EN_FIGURES, CN_FIGURES))
    return paths


def mean_history(root: Path, method: str, metric_name: str = "val_f1") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = detail_rows_for_root(root)
    histories: list[list[float]] = []
    for row in rows:
        if row["method"] != method or int(row["shot"]) != 16:
            continue
        seed = int(row["seed"])
        if method == "baseline_cnn":
            path = root / "shot_16" / f"seed_{seed}" / "baseline" / "baseline_cnn_history.json"
        elif method == "hybrid_control":
            path = root / "shot_16" / f"seed_{seed}" / "hybrid_control" / "hybrid_control_history.json"
        else:
            path = root / "shot_16" / f"seed_{seed}" / method / "hybrid_qnn_history.json"
        histories.append([float(value) for value in load_json(path)[metric_name]])
    max_len = max(len(history) for history in histories)
    stacked = np.full((len(histories), max_len), np.nan)
    for row_index, history in enumerate(histories):
        stacked[row_index, : len(history)] = history
    return np.arange(1, max_len + 1), np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


def draw_supplement_optimization(output_dir: Path) -> list[Path]:
    configure_style()
    figure, ax = plt.subplots(figsize=(5.9, 3.35))
    method_order = ("baseline_cnn", "hybrid_control", "hybrid_no_entanglement", "hybrid_zz_real_amplitudes")
    labels = {
        "baseline_cnn": "CNN baseline",
        "hybrid_control": "Classical control",
        "hybrid_no_entanglement": "No-ent",
        "hybrid_zz_real_amplitudes": "ZZ+RealAmp",
    }
    colors = {
        "baseline_cnn": "#7A7A7A",
        "hybrid_control": "#356D9B",
        **METHOD_COLORS,
    }
    for method in method_order:
        epochs, mean_values, std_values = mean_history(TRANSPORT_MAIN_ROOT, method)
        ax.plot(epochs, mean_values, color=colors[method], linewidth=1.8, label=labels[method])
        ax.fill_between(epochs, mean_values - std_values, mean_values + std_values, color=colors[method], alpha=0.12, linewidth=0)
    ax.axvline(2.5, color="#808A93", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Supplementary Fig. S2. Optimization curves", fontsize=10.5, fontweight="bold")
    ax.grid(True, axis="y", color="#D9DEE4", linewidth=0.7)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    figure.tight_layout()
    paths = save_figure(figure, output_dir, "supp_fig_s2_optimization_curves")
    copy_outputs(paths, (EN_FIGURES, CN_FIGURES))
    return paths


def mean_confusion_matrix(root: Path, method: str) -> tuple[list[str], np.ndarray]:
    rows = [row for row in detail_rows_for_root(root) if row["method"] == method and int(row["shot"]) == 16]
    matrices: list[np.ndarray] = []
    names: list[str] | None = None
    for row in rows:
        seed = int(row["seed"])
        if method == "hybrid_control":
            path = root / "shot_16" / f"seed_{seed}" / "hybrid_control" / "hybrid_control_test_metrics.json"
        else:
            path = root / "shot_16" / f"seed_{seed}" / method / "hybrid_qnn_test_metrics.json"
        metrics = load_json(path)["metrics"]
        names = list(metrics["class_names"])
        matrix = np.asarray(metrics["confusion_matrix"], dtype=float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrices.append(np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0))
    if names is None:
        raise ValueError("No confusion records found")
    return names, np.mean(matrices, axis=0)


def draw_confusion(axis: plt.Axes, matrix: np.ndarray, names: list[str], title: str) -> None:
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6.4, color="white" if value > 0.52 else "#1F2933")
    axis.set_xticks(np.arange(len(names)))
    axis.set_xticklabels(names, rotation=30, ha="right", fontsize=7.0)
    axis.set_yticks(np.arange(len(names)))
    axis.set_yticklabels(names, fontsize=7.0)
    axis.set_title(title, fontsize=8.5)
    return image


def draw_supplement_per_class_confusion(output_dir: Path) -> list[Path]:
    configure_style()
    per_class_rows = load_csv_rows(per_class_table_for_root(TRANSPORT_MAIN_ROOT))
    figure = plt.figure(figsize=(7.2, 4.9))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.43, wspace=0.28)
    ax_bar = figure.add_subplot(grid[0, :])
    x = np.arange(len(CLASS_ORDER))
    width = 0.18
    methods = ("baseline_cnn", "hybrid_control", "hybrid_no_entanglement", "hybrid_zz_real_amplitudes")
    labels = {
        "baseline_cnn": "CNN",
        "hybrid_control": "Control",
        "hybrid_no_entanglement": "No-ent",
        "hybrid_zz_real_amplitudes": "ZZ+RealAmp",
    }
    colors = {
        "baseline_cnn": "#7A7A7A",
        "hybrid_control": "#356D9B",
        **METHOD_COLORS,
    }
    for index, method in enumerate(methods):
        method_rows = {row["class_name"]: row for row in per_class_rows if row["method"] == method and int(row["shot"]) == 16}
        ax_bar.bar(
            x + (index - 1.5) * width,
            [float(method_rows[class_name]["f1_mean"]) for class_name in CLASS_ORDER],
            width=width,
            color=colors[method],
            label=labels[method],
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(CLASS_ORDER)
    ax_bar.set_ylim(0, 0.95)
    ax_bar.set_ylabel("Per-class F1")
    ax_bar.set_title("Supplementary Fig. S3. Per-class F1 and confusion matrices", fontsize=10.5, fontweight="bold")
    ax_bar.grid(True, axis="y", color="#D9DEE4", linewidth=0.7)
    ax_bar.legend(frameon=False, ncol=4, loc="upper left")

    names, control_matrix = mean_confusion_matrix(TRANSPORT_MAIN_ROOT, "hybrid_control")
    _, no_ent_matrix = mean_confusion_matrix(TRANSPORT_MAIN_ROOT, "hybrid_no_entanglement")
    ax_control = figure.add_subplot(grid[1, 0])
    draw_confusion(ax_control, control_matrix, names, "Classical control")
    ax_noent = figure.add_subplot(grid[1, 1])
    image = draw_confusion(ax_noent, no_ent_matrix, names, "QNN No-ent")
    figure.colorbar(image, ax=[ax_control, ax_noent], fraction=0.03, pad=0.02, label="Row-normalized")
    paths = save_figure(figure, output_dir, "supp_fig_s3_per_class_confusion")
    copy_outputs(paths, (EN_FIGURES, CN_FIGURES))
    return paths


def main() -> None:
    records = collect_records()
    ensure_dir(OUTPUT_ROOT)
    write_record_table(records, OUTPUT_ROOT / "fig_regime_boundary_evidence_map_data.csv")
    write_supplementary_table_s1(records)
    generated: list[Path] = []
    generated.extend(draw_framework_schematic(OUTPUT_ROOT))
    generated.extend(draw_regime_boundary_map(records, OUTPUT_ROOT))
    generated.extend(draw_supplement_roi_examples(OUTPUT_ROOT))
    generated.extend(draw_supplement_optimization(OUTPUT_ROOT))
    generated.extend(draw_supplement_per_class_confusion(OUTPUT_ROOT))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()

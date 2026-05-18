#!/usr/bin/env python3
"""Generate compact assets for the <=5-page GRSL package."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import generate_regime_boundary_evidence_map as reg  # noqa: E402
import generate_sci_composite_figures as sci  # noqa: E402


PACKAGE_DIR = PROJECT_ROOT / "GRSL_overleaf_submission_5page_package_v2_figtable_slim"
FIG_DIR = PACKAGE_DIR / "figures"
SUPP_DIR = PACKAGE_DIR / "supplementary"

TEXT_DARK = "#1F2933"
TEXT_MUTED = "#5B6770"
GRID = "#D6DBE1"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.6,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.75,
            "grid.color": GRID,
            "grid.linewidth": 0.65,
            "grid.alpha": 0.85,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_png(fig: plt.Figure, stem: str) -> Path:
    ensure_dir(FIG_DIR)
    path = FIG_DIR / f"{stem}.png"
    fig.savefig(path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return path


def save_supp_png(fig: plt.Figure, stem: str) -> Path:
    ensure_dir(SUPP_DIR)
    path = SUPP_DIR / f"{stem}.png"
    fig.savefig(path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return path


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.045,
        1.04,
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.6,
        fontweight="bold",
        color=TEXT_DARK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
    )


def draw_roi_strip(
    fig: plt.Figure,
    spec: gridspec.SubplotSpec,
    *,
    title: str,
    dataset_root: Path,
    classes: list[str],
    panel_label: str,
) -> None:
    counts = sci.split_counts(dataset_root)
    outer = gridspec.GridSpecFromSubplotSpec(
        2,
        len(classes),
        subplot_spec=spec,
        height_ratios=[0.20, 1.0],
        wspace=0.08,
        hspace=0.03,
    )
    title_ax = fig.add_subplot(outer[0, :])
    title_ax.axis("off")
    add_panel_label(title_ax, panel_label)
    title_ax.text(
        0.0,
        0.4,
        title,
        ha="left",
        va="center",
        fontsize=8.9,
        fontweight="bold",
        color=TEXT_DARK,
        transform=title_ax.transAxes,
    )
    for col, class_name in enumerate(classes):
        ax = fig.add_subplot(outer[1, col])
        ax.set_xticks([])
        ax.set_yticks([])
        image_path = sci.sample_image_paths(dataset_root, class_name, count=1)[0]
        ax.imshow(plt.imread(image_path))
        ax.set_aspect("equal")
        ax.set_title(class_name, fontsize=6.0, pad=1.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.65)
            spine.set_edgecolor("#CAD2DA")


def draw_compact_ladder(ax: plt.Axes) -> None:
    ax.axis("off")
    add_panel_label(ax, "c")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        subtitle: str,
        face: str,
        edge: str = "#C7CED6",
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            linewidth=0.8,
            edgecolor=edge,
            facecolor=face,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(patch)
        if subtitle:
            ax.text(
                x + 0.05 * w,
                y + 0.65 * h,
                title,
                ha="left",
                va="center",
                fontsize=5.9,
                fontweight="bold",
                color=TEXT_DARK,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.text(
                x + 0.05 * w,
                y + 0.31 * h,
                subtitle,
                ha="left",
                va="center",
                fontsize=4.8,
                color=TEXT_MUTED,
                transform=ax.transAxes,
                clip_on=False,
            )
        else:
            ax.text(
                x + 0.5 * w,
                y + 0.5 * h,
                title,
                ha="center",
                va="center",
                fontsize=6.1,
                fontweight="bold",
                color=TEXT_DARK,
                transform=ax.transAxes,
                clip_on=False,
            )

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9.5,
                linewidth=0.95,
                color="#5F6C77",
                transform=ax.transAxes,
                clip_on=False,
            )
        )

    ax.text(
        0.0,
        1.03,
        "Matched-control evaluation ladder",
        ha="left",
        va="bottom",
        fontsize=8.9,
        fontweight="bold",
        color=TEXT_DARK,
        transform=ax.transAxes,
        clip_on=False,
    )
    bottom_y = -0.065
    bottom_h = 0.16
    bottom_top = bottom_y + bottom_h
    box(0.01, 0.62, 0.18, 0.23, "DIOR ROI\ncrops", "", "#F5F7FA")
    box(0.23, 0.62, 0.18, 0.23, "Low-shot split\n16/32/64", "", "#F5F7FA")
    box(0.45, 0.62, 0.20, 0.23, "Matched protocol\nsplit/backbone/opt.", "", "#F5F7FA")
    box(0.06, 0.25, 0.23, 0.23, "Classical\ncontrol", "", "#EDF4FB", "#9BBAD3")
    box(0.38, 0.25, 0.23, 0.23, "No-ent\nQNN", "", "#EEF8F6", "#8BCBC1")
    box(0.70, 0.25, 0.23, 0.23, "ZZ+RA\nQNN", "", "#FFF5D8", "#E3BE54")
    box(0.08, bottom_y, 0.18, bottom_h, "QCAD", "", "#F9FAFB")
    box(0.31, bottom_y, 0.18, bottom_h, "LSEC", "", "#F9FAFB")
    box(0.54, bottom_y, 0.18, bottom_h, "QIB", "", "#F9FAFB")
    box(0.77, bottom_y, 0.18, bottom_h, "Stress\ntests", "", "#F9FAFB")
    for start, end in [
        ((0.19, 0.735), (0.23, 0.735)),
        ((0.41, 0.735), (0.45, 0.735)),
        ((0.55, 0.62), (0.17, 0.48)),
        ((0.55, 0.62), (0.50, 0.48)),
        ((0.55, 0.62), (0.81, 0.48)),
        ((0.18, 0.25), (0.17, bottom_top)),
        ((0.50, 0.25), (0.40, bottom_top)),
        ((0.58, 0.25), (0.63, bottom_top)),
        ((0.82, 0.25), (0.86, bottom_top)),
    ]:
        arrow(start, end)


def draw_figure_1() -> Path:
    configure_style()
    fig = plt.figure(figsize=(7.18, 3.02))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.98, 1.02], hspace=0.22, wspace=0.14)
    draw_roi_strip(
        fig,
        grid[0, 0],
        title="TL4 main subset",
        dataset_root=sci.TRANSPORT_DATASET_ROOT,
        classes=["airplane", "ship", "vehicle", "harbor"],
        panel_label="a",
    )
    draw_roi_strip(
        fig,
        grid[0, 1],
        title="Urban4 boundary subset",
        dataset_root=sci.URBAN_DATASET_ROOT,
        classes=["vehicle", "bridge", "harbor", "storagetank"],
        panel_label="b",
    )
    ladder_ax = fig.add_subplot(grid[1, :])
    draw_compact_ladder(ladder_ax)
    return save_png(fig, "fig1_study_design_overview")


def plot_seed_panel(ax: plt.Axes) -> None:
    rows = sci.detail_rows_for_root(sci.TRANSPORT_MAIN_ROOT)
    rng = np.random.default_rng(3)
    methods = sci.METHOD_ORDER
    labels = ["CNN", "Control", "No-ent", "ZZ+RA"]
    for idx, method in enumerate(methods):
        scores = [score for _, score in sci.seed_scores(rows, method, 16)]
        jitter = rng.uniform(-0.045, 0.045, size=len(scores))
        color = sci.METHOD_COLORS[method]
        ax.scatter(np.full(len(scores), idx) + jitter, scores, s=14, color=color, alpha=0.72, linewidths=0)
        mean = float(np.mean(scores))
        std = float(np.std(scores, ddof=1))
        ax.errorbar(idx, mean, yerr=std, fmt="o", markersize=6.2, color=color, capsize=3.0, linewidth=1.35)
        ax.text(idx, mean + std + 0.015, f"{mean:.3f}", ha="center", va="bottom", fontsize=6.9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Macro-F1")
    ax.set_title("(a) Seed-level macro-F1")
    ax.set_ylim(0.05, 0.76)
    ax.grid(True, axis="y")


def plot_paired_delta_panel(ax: plt.Axes) -> None:
    rows = sci.detail_rows_for_root(sci.TRANSPORT_MAIN_ROOT)
    pairwise = sci.pairwise_rows_for_root(sci.TRANSPORT_MAIN_ROOT)
    rng = np.random.default_rng(4)
    methods = ["hybrid_no_entanglement", "hybrid_zz_real_amplitudes"]
    labels = ["No-ent", "ZZ+RA"]
    for idx, method in enumerate(methods):
        deltas = [delta for _, delta in sci.seed_deltas(rows, "hybrid_control", method, 16)]
        color = sci.METHOD_COLORS[method]
        jitter = rng.uniform(-0.035, 0.035, size=len(deltas))
        ax.scatter(np.full(len(deltas), idx) + jitter, deltas, s=14, color=color, alpha=0.72, linewidths=0)
        mean = float(np.mean(deltas))
        ci = reg.precomputed_bootstrap_ci("tl4_small_16", method)
        assert ci is not None
        p_value = float(sci.pairwise_row(pairwise, "hybrid_control", method, 16)["wilcoxon_p_value"])
        ax.errorbar(
            idx,
            mean,
            yerr=[[mean - ci[0]], [ci[1] - mean]],
            fmt="o",
            markersize=6.2,
            color=color,
            capsize=3.0,
            linewidth=1.35,
        )
        ax.text(idx, ci[1] + 0.018, f"Δ={mean:+.3f}\np={p_value:.3f}", ha="center", va="bottom", fontsize=6.8)
    ax.axhline(0.0, color="#444444", linewidth=0.9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta vs control")
    ax.set_title("(b) Paired QNN-control deltas")
    ax.set_ylim(-0.06, 0.49)
    ax.set_xlim(-0.70, 1.70)
    ax.grid(True, axis="y")


def plot_per_class_delta_panel(ax: plt.Axes) -> None:
    rows = sci.load_csv_rows(sci.per_class_table_for_root(sci.TRANSPORT_MAIN_ROOT))
    classes = ["airplane", "ship", "vehicle", "harbor"]
    control = {
        row["class_name"]: float(row["f1_mean"])
        for row in rows
        if row["method"] == "hybrid_control" and int(row["shot"]) == 16
    }
    x = np.arange(len(classes), dtype=float)
    width = 0.34
    for offset, method in [(-width / 2, "hybrid_no_entanglement"), (width / 2, "hybrid_zz_real_amplitudes")]:
        values = []
        for class_name in classes:
            match = next(
                row
                for row in rows
                if row["method"] == method and int(row["shot"]) == 16 and row["class_name"] == class_name
            )
            values.append(float(match["f1_mean"]) - control[class_name])
        ax.bar(x + offset, values, width=width, color=sci.METHOD_COLORS[method], edgecolor="white", linewidth=0.4, label=sci.QNN_LABELS[method])
    ax.axhline(0.0, color="#444444", linewidth=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Per-class F1 delta")
    ax.set_title("(c) Class-structured F1 delta")
    ax.set_ylim(-0.20, 0.36)
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="upper right")


def plot_confusion_delta_panel(ax: plt.Axes) -> None:
    classes, control = sci.mean_confusion_matrix(sci.TRANSPORT_MAIN_ROOT, "hybrid_control", 16)
    _, no_ent = sci.mean_confusion_matrix(sci.TRANSPORT_MAIN_ROOT, "hybrid_no_entanglement", 16)
    delta = no_ent - control
    cmap = LinearSegmentedColormap.from_list("conf_delta", ["#B04745", "#F7F7F7", "#177F70"])
    norm = TwoSlopeNorm(vmin=-0.35, vcenter=0.0, vmax=0.35)
    image = ax.imshow(delta, cmap=cmap, norm=norm)
    for r in range(delta.shape[0]):
        for c in range(delta.shape[1]):
            value = delta[r, c]
            ax.text(c, r, f"{value:+.2f}", ha="center", va="center", fontsize=6.8, color="white" if abs(value) > 0.22 else TEXT_DARK)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("(d) Confusion improvement: No-ent minus control")
    plt.colorbar(image, ax=ax, fraction=0.045, pad=0.025)


def draw_figure_2() -> Path:
    configure_style()
    fig = plt.figure(figsize=(7.18, 3.78))
    grid = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.28)
    plot_seed_panel(fig.add_subplot(grid[0, 0]))
    plot_paired_delta_panel(fig.add_subplot(grid[0, 1]))
    plot_per_class_delta_panel(fig.add_subplot(grid[1, 0]))
    plot_confusion_delta_panel(fig.add_subplot(grid[1, 1]))
    return save_png(fig, "fig2_main_lowshot_evidence")


def runtime_marker_size(runtime_ratio: float) -> float:
    return 42.0 + 18.0 * runtime_ratio


def draw_figure_3() -> Path:
    journal_script = PACKAGE_DIR / "fig3_journal_boundary_map.py"
    if journal_script.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("fig3_journal_boundary_map", journal_script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {journal_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
        return FIG_DIR / "fig3_journal_boundary_map.png"

    configure_style()
    plt.rcParams.update(
        {
            "font.size": 7.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.1,
            "legend.fontsize": 7.0,
        }
    )
    records = reg.collect_records()
    fig, ax = plt.subplots(figsize=(4.15, 3.05))
    row_order = [
        ("tl4_small_16", "TL4, 16-shot, small CNN"),
        ("tl4_small_32", "TL4, 32-shot, small CNN"),
        ("tl4_small_64", "TL4, 64-shot, small CNN"),
        ("tl4_resnet18_16", "TL4, 16-shot, ResNet18"),
        ("urban4_small_16", "Urban4, 16-shot, small CNN"),
    ]
    base_y = {regime_id: len(row_order) - 1 - idx for idx, (regime_id, _) in enumerate(row_order)}
    lookup = {(item["regime_id"], item["method"]): item for item in records}
    styles = {
        "hybrid_no_entanglement": {
            "label": "No-ent QNN",
            "color": "#1B7F79",
            "marker": "o",
            "offset": 0.105,
        },
        "hybrid_zz_real_amplitudes": {
            "label": "ZZ+RA QNN",
            "color": "#B07D00",
            "marker": "D",
            "offset": -0.105,
        },
    }

    for method, style in styles.items():
        x_values = []
        y_values = []
        xerr_low = []
        xerr_high = []
        for regime_id, _label in row_order:
            record = lookup[(regime_id, method)]
            mean = float(record["mean_delta"])
            ci_low = float(record["ci_low"])
            ci_high = float(record["ci_high"])
            x_values.append(mean)
            y_values.append(base_y[regime_id] + float(style["offset"]))
            xerr_low.append(mean - ci_low)
            xerr_high.append(ci_high - mean)
        ax.errorbar(
            x_values,
            y_values,
            xerr=np.vstack([xerr_low, xerr_high]),
            fmt=style["marker"],
            markersize=4.4,
            markerfacecolor=style["color"],
            markeredgecolor=TEXT_DARK,
            markeredgewidth=0.45,
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.15,
            capsize=2.2,
            capthick=1.0,
            linewidth=0.0,
            label=str(style["label"]),
            zorder=3,
        )

    ax.axvline(0.0, color="#222222", linewidth=0.75, zorder=1)
    ax.axvline(0.10, color="#777777", linewidth=0.75, linestyle=(0, (3.0, 2.0)), zorder=1)
    ax.set_yticks([base_y[key] for key, _ in row_order])
    ax.set_yticklabels([label for _, label in row_order])
    ax.set_xlim(-0.17, 0.34)
    ax.set_ylim(-0.55, len(row_order) - 0.45)
    ax.set_xticks([-0.15, 0.00, 0.10, 0.20, 0.30])
    ax.set_xlabel(r"Paired macro-F1 gain $\Delta$ vs matched control")
    ax.grid(True, axis="x", color="#E4E8ED", linewidth=0.55)
    ax.grid(False, axis="y")
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.65)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.31),
        handlelength=1.6,
        columnspacing=1.3,
        handletextpad=0.55,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.39, right=0.98, top=0.98, bottom=0.25)

    ensure_dir(FIG_DIR)
    pdf_path = FIG_DIR / "fig3_redesign.pdf"
    png_path = FIG_DIR / "fig3_redesign.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return png_path


def draw_supp_fig_s2() -> Path:
    configure_style()
    fig, ax = plt.subplots(figsize=(3.55, 2.18))
    sci.plot_training_curve_panel(ax, sci.TRANSPORT_MAIN_ROOT, shot=16)
    ax.set_title("Supplementary Fig. S2. Optimization dynamics", fontsize=8.7)
    fig.tight_layout(pad=0.3)
    return save_supp_png(fig, "supp_fig_s2_optimization_curves")


def write_supp_table_s1() -> tuple[Path, Path]:
    ensure_dir(SUPP_DIR)
    records = reg.collect_records()
    csv_rows: list[dict[str, str]] = []
    for regime in reg.REGIMES:
        summaries = reg.summary_rows_for_root(regime.root)
        control = reg.summary_row(summaries, "hybrid_control", regime.shot)
        row = {
            "regime": regime.label,
            "control_macro_f1": f"{float(control['macro_f1_mean']):.4f} +/- {float(control['macro_f1_std']):.4f}",
        }
        for method in reg.QNN_METHODS:
            summary = reg.summary_row(summaries, method, regime.shot)
            record = next(item for item in records if item["regime_id"] == regime.regime_id and item["method"] == method)
            pairwise = reg.pairwise_row(reg.pairwise_rows_for_root(regime.root), method, regime.shot)
            prefix = "no_ent" if method == "hybrid_no_entanglement" else "zz_ra"
            row[f"{prefix}_macro_f1"] = f"{float(summary['macro_f1_mean']):.4f} +/- {float(summary['macro_f1_std']):.4f}"
            row[f"{prefix}_delta"] = f"{record['mean_delta']:+.4f}"
            row[f"{prefix}_ci95"] = f"[{record['ci_low']:+.4f}, {record['ci_high']:+.4f}]"
            row[f"{prefix}_p"] = f"{record['p_value']:.5f}"
            row[f"{prefix}_signs"] = f"{pairwise['better_count']}/{pairwise['tie_count']}/{pairwise['worse_count']}"
            row[f"{prefix}_runtime"] = f"{record['runtime_ratio']:.2f}x"
        csv_rows.append(row)

    csv_path = SUPP_DIR / "supp_table_s1_full_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    tex_path = SUPP_DIR / "supp_table_s1_full_statistics.tex"
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write("% Supplementary Table S1. Full boundary statistics.\n")
        handle.write("\\begin{table*}[!t]\n\\centering\n\\scriptsize\n")
        handle.write("\\caption{Supplementary Table S1. Full boundary statistics.}\n")
        handle.write("\\begin{tabular}{p{0.18\\textwidth}p{0.13\\textwidth}p{0.29\\textwidth}p{0.29\\textwidth}}\n")
        handle.write("\\toprule\nRegime & Control & No-ent & ZZ+RA \\\\\n\\midrule\n")
        for row in csv_rows:
            handle.write(
                f"{row['regime']} & {row['control_macro_f1']} & "
                f"{row['no_ent_macro_f1']}; $\\Delta={row['no_ent_delta']}$; CI {row['no_ent_ci95']}; "
                f"$p={row['no_ent_p']}$; signs {row['no_ent_signs']}; $\\rho={row['no_ent_runtime']}$ & "
                f"{row['zz_ra_macro_f1']}; $\\Delta={row['zz_ra_delta']}$; CI {row['zz_ra_ci95']}; "
                f"$p={row['zz_ra_p']}$; signs {row['zz_ra_signs']}; $\\rho={row['zz_ra_runtime']}$ \\\\\n"
            )
        handle.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    return csv_path, tex_path


def main() -> None:
    ensure_dir(FIG_DIR)
    ensure_dir(SUPP_DIR)
    for old in list(FIG_DIR.glob("*")):
        old.unlink()
    for old in list(SUPP_DIR.glob("*")):
        old.unlink()
    outputs: list[Path] = [
        draw_figure_1(),
        draw_figure_2(),
        draw_figure_3(),
    ]
    outputs.extend(write_supp_table_s1())
    for output in outputs:
        print(output.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()

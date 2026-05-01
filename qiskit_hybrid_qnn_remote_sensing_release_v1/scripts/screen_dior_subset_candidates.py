#!/usr/bin/env python3
"""Screen semantic DIOR class subsets for low-shot hybrid QNN potential."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BUILD_SCRIPT = SCRIPT_DIR / "build_dior_multiclass_stratified.py"
RUN_SCRIPT = SCRIPT_DIR / "run_dior_multiclass_lowshot.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "dior_subset_candidate_screening"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed" / "subset_candidates"

CANDIDATE_CLASS_SETS: dict[str, tuple[str, ...]] = {
    "transport_logistics4": ("airplane", "ship", "vehicle", "harbor"),
}

METHOD_KEYS = (
    "baseline_cnn",
    "hybrid_control",
    "hybrid_no_entanglement",
    "hybrid_zz_real_amplitudes",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build and screen semantic DIOR low-shot subset candidates."
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=list(CANDIDATE_CLASS_SETS),
        choices=sorted(CANDIDATE_CLASS_SETS),
        help="Named class-subset candidates to screen.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "dataset_DIOR",
        help="Original DIOR dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Artifact root for candidate experiments (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--processed-data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Processed data root for candidate datasets (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used for build and run steps.",
    )
    parser.add_argument("--shots", type=int, nargs="+", default=[16], help="Shot counts to screen.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Seed list.")
    parser.add_argument("--baseline-epochs", type=int, default=10, help="Baseline epochs.")
    parser.add_argument("--hybrid-epochs", type=int, default=12, help="Hybrid/control epochs.")
    parser.add_argument(
        "--circuits",
        nargs="+",
        default=["no_entanglement", "zz_real_amplitudes"],
        help="QNN circuit types passed to the low-shot runner.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing datasets and experiment runs when artifacts already exist.",
    )
    parser.add_argument(
        "--train-augment",
        type=str,
        default="train_light",
        choices=("none", "train_light"),
        help="Train augmentation preset.",
    )
    parser.add_argument(
        "--val-max-samples-per-class",
        type=int,
        default=256,
        help="Training-time validation cap passed to the runner.",
    )
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=4,
        help="QNN qubit count.",
    )
    parser.add_argument(
        "--qnn-output-dim",
        type=int,
        default=4,
        help="QNN output feature count.",
    )
    return parser.parse_args()


def run_command(command: list[str], workdir: Path) -> None:
    """Run one subprocess with streaming output."""
    subprocess.run(command, cwd=workdir, check=True)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def summarize_candidate(
    candidate_name: str,
    class_names: tuple[str, ...],
    summary_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Summarize one candidate from low-shot summary CSV rows."""
    rows: list[dict[str, Any]] = []
    shots = sorted({int(row["shot"]) for row in summary_rows})
    for shot in shots:
        shot_rows = {
            row["method"]: row for row in summary_rows if int(row["shot"]) == shot
        }
        if not all(method in shot_rows for method in METHOD_KEYS):
            continue

        control_mean = float(shot_rows["hybrid_control"]["macro_f1_mean"])
        no_ent_mean = float(shot_rows["hybrid_no_entanglement"]["macro_f1_mean"])
        zz_mean = float(shot_rows["hybrid_zz_real_amplitudes"]["macro_f1_mean"])

        best_qnn_method = "hybrid_no_entanglement"
        best_qnn_mean = no_ent_mean
        if zz_mean > best_qnn_mean:
            best_qnn_method = "hybrid_zz_real_amplitudes"
            best_qnn_mean = zz_mean

        rows.append(
            {
                "candidate": candidate_name,
                "classes": " ".join(class_names),
                "num_classes": len(class_names),
                "shot": shot,
                "baseline_macro_f1_mean": float(shot_rows["baseline_cnn"]["macro_f1_mean"]),
                "control_macro_f1_mean": control_mean,
                "qnn_no_ent_macro_f1_mean": no_ent_mean,
                "qnn_zz_macro_f1_mean": zz_mean,
                "best_qnn_method": best_qnn_method,
                "best_qnn_macro_f1_mean": best_qnn_mean,
                "best_qnn_delta_vs_control": best_qnn_mean - control_mean,
                "control_runtime_mean": float(shot_rows["hybrid_control"]["runtime_seconds_mean"]),
                "qnn_no_ent_runtime_mean": float(shot_rows["hybrid_no_entanglement"]["runtime_seconds_mean"]),
                "qnn_zz_runtime_mean": float(shot_rows["hybrid_zz_real_amplitudes"]["runtime_seconds_mean"]),
            }
        )
    return rows


def main() -> None:
    """Entry point."""
    args = parse_args()
    aggregated_rows: list[dict[str, Any]] = []

    for candidate_name in args.candidates:
        class_names = CANDIDATE_CLASS_SETS[candidate_name]
        dataset_root = args.processed_data_root / candidate_name
        artifact_root = args.output_root / candidate_name
        summary_csv = artifact_root / "lowshot_summary_rows.csv"

        if not (args.skip_existing and dataset_root.exists()):
            build_command = [
                str(args.python),
                str(BUILD_SCRIPT),
                "--dataset-root",
                str(args.dataset_root),
                "--output-dir",
                str(dataset_root),
                "--classes",
                *class_names,
            ]
            run_command(build_command, PROJECT_ROOT)

        if not (args.skip_existing and summary_csv.exists()):
            run_command_list = [
                str(args.python),
                str(RUN_SCRIPT),
                "--dataset-root",
                str(dataset_root),
                "--output-dir",
                str(artifact_root),
                "--shots",
                *[str(shot) for shot in args.shots],
                "--seeds",
                *[str(seed) for seed in args.seeds],
                "--include-classical-control",
                "--circuits",
                *args.circuits,
                "--baseline-epochs",
                str(args.baseline_epochs),
                "--hybrid-epochs",
                str(args.hybrid_epochs),
                "--num-qubits",
                str(args.num_qubits),
                "--qnn-output-dim",
                str(args.qnn_output_dim),
                "--train-augment",
                args.train_augment,
                "--val-max-samples-per-class",
                str(args.val_max_samples_per_class),
            ]
            if args.skip_existing:
                run_command_list.append("--skip-existing")
            run_command(run_command_list, PROJECT_ROOT)

        aggregated_rows.extend(
            summarize_candidate(
                candidate_name=candidate_name,
                class_names=class_names,
                summary_rows=load_csv_rows(summary_csv),
            )
        )

    aggregated_rows.sort(
        key=lambda row: (
            int(row["shot"]),
            float(row["best_qnn_delta_vs_control"]),
        ),
        reverse=True,
    )
    output_csv = args.output_root / "subset_candidate_summary.csv"
    write_csv(output_csv, aggregated_rows)
    print(f"Wrote subset candidate summary to {output_csv}")


if __name__ == "__main__":
    main()

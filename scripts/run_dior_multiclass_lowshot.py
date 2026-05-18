#!/usr/bin/env python3
"""Run the DIOR multiclass low-shot baseline-vs-hybrid experiment matrix."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import BUILD_COMMAND_HINT, validate_processed_dataset_nonempty
from src.utils import load_json, save_json


DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dior_multiclass_stratified"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "dior_multiclass_lowshot"
BASELINE_SCRIPT = SCRIPT_DIR / "train_baseline_cnn.py"
CONTROL_SCRIPT = SCRIPT_DIR / "train_hybrid_control.py"
HYBRID_SCRIPT = SCRIPT_DIR / "train_hybrid_qnn.py"
DEFAULT_CIRCUITS = ("no_entanglement", "zz_real_amplitudes", "ry_ring")
METRIC_NAMES = (
    "accuracy",
    "f1",
    "macro_f1",
    "weighted_f1",
    "roc_auc",
    "runtime_seconds",
)


def default_experiment_python() -> Path:
    """Prefer the project-local unified venv when available."""
    candidate = PROJECT_ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return candidate
    return Path(sys.executable)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run DIOR multiclass low-shot baseline and hybrid Qiskit experiments."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=f"Processed multiclass ROI dataset root (default: {DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Root directory for low-shot experiment artifacts (default: {OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--baseline-python",
        type=Path,
        default=default_experiment_python(),
        help="Python executable used for baseline CNN runs.",
    )
    parser.add_argument(
        "--hybrid-python",
        type=Path,
        default=default_experiment_python(),
        help="Python executable used for hybrid QNN runs.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Per-class train-shot counts.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Seed list.",
    )
    parser.add_argument(
        "--circuits",
        type=str,
        nargs="+",
        default=list(DEFAULT_CIRCUITS),
        choices=DEFAULT_CIRCUITS,
        help="Hybrid QNN circuit families.",
    )
    parser.add_argument("--baseline-epochs", type=int, default=10, help="Epochs for baseline CNN runs.")
    parser.add_argument("--hybrid-epochs", type=int, default=12, help="Epochs for hybrid QNN runs.")
    parser.add_argument("--baseline-batch-size", type=int, default=32, help="Baseline batch size.")
    parser.add_argument("--hybrid-batch-size", type=int, default=16, help="Hybrid batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Shared learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Shared weight decay.")
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the hybrid/control backbone learning rate.",
    )
    parser.add_argument("--image-size", type=int, default=64, help="Input image size.")
    parser.add_argument("--feature-dim", type=int, default=64, help="Baseline CNN feature width.")
    parser.add_argument(
        "--backbone-type",
        type=str,
        default="small_cnn",
        choices=("small_cnn", "resnet18"),
        help="Backbone family shared by baseline and hybrid models.",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Initialize the baseline backbone from torchvision pretrained weights when supported.",
    )
    parser.add_argument(
        "--input-normalization",
        type=str,
        default="none",
        choices=("none", "imagenet"),
        help="Input normalization preset shared by baseline and hybrid models.",
    )
    parser.add_argument(
        "--backbone-feature-dim",
        type=int,
        default=64,
        help="Hybrid CNN backbone feature width.",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Shared dropout ratio.")
    parser.add_argument("--num-qubits", type=int, default=4, help="Hybrid QNN qubit count.")
    parser.add_argument(
        "--qnn-output-dim",
        type=int,
        default=4,
        help="Hybrid QNN observable/output feature count.",
    )
    parser.add_argument("--feature-map-reps", type=int, default=1, help="Hybrid feature-map repetitions.")
    parser.add_argument("--ansatz-reps", type=int, default=1, help="Hybrid ansatz repetitions.")
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="concat",
        choices=("concat", "add"),
        help="Hybrid fusion mode.",
    )
    parser.add_argument("--classical-head-weight", type=float, default=1.0, help="Hybrid residual classical weight.")
    parser.add_argument("--qnn-head-weight", type=float, default=1.0, help="Hybrid QNN branch weight.")
    parser.add_argument(
        "--include-classical-control",
        action="store_true",
        help="Also run the matched classical control branch under the same backbone/projector/fusion pipeline.",
    )
    parser.add_argument(
        "--control-branch-input-dim",
        type=int,
        default=4,
        help="Classical control branch projector output width.",
    )
    parser.add_argument(
        "--control-branch-output-dim",
        type=int,
        default=4,
        help="Classical control branch bounded output width.",
    )
    parser.add_argument(
        "--control-head-weight",
        type=float,
        default=1.0,
        help="Weight applied to the classical control branch contribution.",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=2,
        help="Frozen-backbone warm-start epochs for hybrid runs.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=("cosine", "step", "none"),
        help="Shared scheduler.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Shared scheduler warmup epochs.")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0, help="Shared gradient clip norm.")
    parser.add_argument(
        "--train-augment",
        type=str,
        default="train_light",
        choices=("none", "train_light"),
        help="Shared train-only augmentation preset.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help="Shared early-stopping patience.",
    )
    parser.add_argument(
        "--val-max-samples-per-class",
        type=int,
        default=256,
        help="Training-time per-class cap for validation split (default: 256).",
    )
    parser.add_argument(
        "--test-max-samples-per-class",
        type=int,
        default=None,
        help="Training-time per-class cap for test split.",
    )
    parser.add_argument(
        "--full-eval-at-end",
        action="store_true",
        help="Rebuild val/test without training-time caps for final reported metrics.",
    )
    parser.add_argument(
        "--no-full-eval-at-end",
        dest="full_eval_at_end",
        action="store_false",
        help="Keep final val/test evaluation on the same capped splits used during training.",
    )
    parser.add_argument(
        "--final-val-max-samples-per-class",
        type=int,
        default=None,
        help="Optional final val cap after training. Overrides --full-eval-at-end for val.",
    )
    parser.add_argument(
        "--final-test-max-samples-per-class",
        type=int,
        default=None,
        help="Optional final test cap after training. Overrides --full-eval-at-end for test.",
    )
    parser.add_argument(
        "--observable-mode",
        type=str,
        default="auto",
        choices=("auto", "single_z", "multi_z", "z_plus_zz"),
        help="Hybrid observable construction mode.",
    )
    parser.add_argument(
        "--max-zz-observables",
        type=int,
        default=None,
        help="Optional cap on ZZ observables for the hybrid head.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose metrics file already exists.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Mark all generated artifacts as smoke-only.",
    )
    parser.set_defaults(full_eval_at_end=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of rows to CSV."""
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


def summarize(values: list[float]) -> dict[str, float | None]:
    """Return mean/std summary stats."""
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(stdev(values))}


def validate_dataset(args: argparse.Namespace) -> None:
    """Validate the multiclass dataset before launching experiments."""
    try:
        validate_processed_dataset_nonempty(dataset_root=args.dataset_root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"{exc} Build the dataset first with: {BUILD_COMMAND_HINT}"
        ) from exc


def run_command(command: list[str]) -> None:
    """Run one subprocess command and fail fast on errors."""
    print(" ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def build_baseline_command(
    args: argparse.Namespace,
    *,
    shot: int,
    seed: int,
    output_dir: Path,
) -> list[str]:
    """Build one baseline CNN training command."""
    command = [
        str(args.baseline_python),
        str(BASELINE_SCRIPT),
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--epochs",
        str(args.baseline_epochs),
        "--batch-size",
        str(args.baseline_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--image-size",
        str(args.image_size),
        "--feature-dim",
        str(args.feature_dim),
        "--backbone-type",
        args.backbone_type,
        "--input-normalization",
        args.input_normalization,
        "--dropout",
        str(args.dropout),
        "--scheduler",
        args.scheduler,
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--train-augment",
        args.train_augment,
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--device",
        "cpu",
        "--experiment-name",
        f"baseline_lowshot_shot{shot}",
        "--train-max-samples-per-class",
        str(shot),
    ]
    if args.val_max_samples_per_class is not None:
        command.extend(["--val-max-samples-per-class", str(args.val_max_samples_per_class)])
    if args.test_max_samples_per_class is not None:
        command.extend(["--test-max-samples-per-class", str(args.test_max_samples_per_class)])
    if args.full_eval_at_end:
        command.append("--full-eval-at-end")
    if args.final_val_max_samples_per_class is not None:
        command.extend(["--final-val-max-samples-per-class", str(args.final_val_max_samples_per_class)])
    if args.final_test_max_samples_per_class is not None:
        command.extend(["--final-test-max-samples-per-class", str(args.final_test_max_samples_per_class)])
    if args.smoke_only:
        command.append("--smoke-only")
    if args.pretrained_backbone:
        command.append("--pretrained-backbone")
    return command


def build_hybrid_command(
    args: argparse.Namespace,
    *,
    shot: int,
    seed: int,
    circuit_type: str,
    output_dir: Path,
    checkpoint_path: Path,
) -> list[str]:
    """Build one hybrid QNN training command."""
    command = [
        str(args.hybrid_python),
        str(HYBRID_SCRIPT),
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--epochs",
        str(args.hybrid_epochs),
        "--batch-size",
        str(args.hybrid_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--backbone-lr-scale",
        str(args.backbone_lr_scale),
        "--image-size",
        str(args.image_size),
        "--backbone-feature-dim",
        str(args.backbone_feature_dim),
        "--backbone-type",
        args.backbone_type,
        "--input-normalization",
        args.input_normalization,
        "--num-qubits",
        str(args.num_qubits),
        "--qnn-output-dim",
        str(args.qnn_output_dim),
        "--feature-map-reps",
        str(args.feature_map_reps),
        "--ansatz-reps",
        str(args.ansatz_reps),
        "--circuit-type",
        circuit_type,
        "--dropout",
        str(args.dropout),
        "--fusion-mode",
        args.fusion_mode,
        "--classical-head-weight",
        str(args.classical_head_weight),
        "--qnn-head-weight",
        str(args.qnn_head_weight),
        "--freeze-backbone-epochs",
        str(args.freeze_backbone_epochs),
        "--scheduler",
        args.scheduler,
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--train-augment",
        args.train_augment,
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--observable-mode",
        args.observable_mode,
        "--pretrained-cnn-checkpoint",
        str(checkpoint_path),
        "--experiment-name",
        f"hybrid_{circuit_type}_lowshot_shot{shot}",
        "--train-max-samples-per-class",
        str(shot),
    ]
    if args.val_max_samples_per_class is not None:
        command.extend(["--val-max-samples-per-class", str(args.val_max_samples_per_class)])
    if args.test_max_samples_per_class is not None:
        command.extend(["--test-max-samples-per-class", str(args.test_max_samples_per_class)])
    if args.full_eval_at_end:
        command.append("--full-eval-at-end")
    if args.final_val_max_samples_per_class is not None:
        command.extend(["--final-val-max-samples-per-class", str(args.final_val_max_samples_per_class)])
    if args.final_test_max_samples_per_class is not None:
        command.extend(["--final-test-max-samples-per-class", str(args.final_test_max_samples_per_class)])
    if args.max_zz_observables is not None:
        command.extend(["--max-zz-observables", str(args.max_zz_observables)])
    if args.smoke_only:
        command.append("--smoke-only")
    if args.pretrained_backbone:
        command.append("--pretrained-backbone")
    return command


def build_control_command(
    args: argparse.Namespace,
    *,
    shot: int,
    seed: int,
    output_dir: Path,
    checkpoint_path: Path,
) -> list[str]:
    """Build one classical-control training command."""
    command = [
        str(args.baseline_python),
        str(CONTROL_SCRIPT),
        "--dataset-root",
        str(args.dataset_root),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--epochs",
        str(args.hybrid_epochs),
        "--batch-size",
        str(args.hybrid_batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--backbone-lr-scale",
        str(args.backbone_lr_scale),
        "--image-size",
        str(args.image_size),
        "--backbone-feature-dim",
        str(args.backbone_feature_dim),
        "--backbone-type",
        args.backbone_type,
        "--input-normalization",
        args.input_normalization,
        "--branch-input-dim",
        str(args.control_branch_input_dim),
        "--branch-output-dim",
        str(args.control_branch_output_dim),
        "--dropout",
        str(args.dropout),
        "--fusion-mode",
        args.fusion_mode,
        "--classical-head-weight",
        str(args.classical_head_weight),
        "--control-head-weight",
        str(args.control_head_weight),
        "--freeze-backbone-epochs",
        str(args.freeze_backbone_epochs),
        "--scheduler",
        args.scheduler,
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--gradient-clip-norm",
        str(args.gradient_clip_norm),
        "--train-augment",
        args.train_augment,
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--pretrained-cnn-checkpoint",
        str(checkpoint_path),
        "--experiment-name",
        f"hybrid_control_lowshot_shot{shot}",
        "--train-max-samples-per-class",
        str(shot),
    ]
    if args.val_max_samples_per_class is not None:
        command.extend(["--val-max-samples-per-class", str(args.val_max_samples_per_class)])
    if args.test_max_samples_per_class is not None:
        command.extend(["--test-max-samples-per-class", str(args.test_max_samples_per_class)])
    if args.full_eval_at_end:
        command.append("--full-eval-at-end")
    if args.final_val_max_samples_per_class is not None:
        command.extend(["--final-val-max-samples-per-class", str(args.final_val_max_samples_per_class)])
    if args.final_test_max_samples_per_class is not None:
        command.extend(["--final-test-max-samples-per-class", str(args.final_test_max_samples_per_class)])
    if args.smoke_only:
        command.append("--smoke-only")
    if args.pretrained_backbone:
        command.append("--pretrained-backbone")
    return command


def extract_result_row(
    *,
    shot: int,
    seed: int,
    method: str,
    circuit_type: str | None,
    metrics_payload: dict[str, Any],
) -> dict[str, Any]:
    """Extract a flat comparable test row from one metrics payload."""
    test_metrics = metrics_payload["results"]["test"]
    return {
        "shot": shot,
        "seed": seed,
        "method": method,
        "circuit_type": circuit_type,
        "accuracy": test_metrics["accuracy"],
        "f1": test_metrics["f1"],
        "macro_f1": test_metrics.get("macro_f1"),
        "weighted_f1": test_metrics.get("weighted_f1"),
        "roc_auc": test_metrics.get("roc_auc"),
        "runtime_seconds": metrics_payload.get("runtime_seconds"),
        "best_epoch": metrics_payload.get("best_epoch"),
        "num_classes": metrics_payload.get("num_classes"),
        "class_names": ",".join(metrics_payload.get("class_names", [])),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate seed-level rows into shot/method summaries."""
    grouped: dict[tuple[int, str, str | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["shot"]), str(row["method"]), row["circuit_type"])
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (shot, method, circuit_type), group_rows in sorted(grouped.items()):
        summary_row: dict[str, Any] = {
            "shot": shot,
            "method": method,
            "circuit_type": circuit_type,
            "num_runs": len(group_rows),
        }
        for metric_name in METRIC_NAMES:
            values = [float(row[metric_name]) for row in group_rows if row.get(metric_name) is not None]
            stats = summarize(values)
            summary_row[f"{metric_name}_mean"] = stats["mean"]
            summary_row[f"{metric_name}_std"] = stats["std"]
        summary_rows.append(summary_row)
    return summary_rows


def main() -> int:
    """Script entry point."""
    args = parse_args()

    try:
        validate_dataset(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    detail_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []

    for shot in args.shots:
        if shot < 1:
            print(f"Error: shot must be >= 1, got {shot}", file=sys.stderr)
            return 1

        for seed in args.seeds:
            seed_root = args.output_dir / f"shot_{shot}" / f"seed_{seed}"
            baseline_dir = seed_root / "baseline"
            baseline_metrics_path = baseline_dir / "baseline_cnn_metrics.json"
            baseline_checkpoint_path = baseline_dir / "baseline_cnn_best.pt"

            baseline_command = build_baseline_command(
                args,
                shot=shot,
                seed=seed,
                output_dir=baseline_dir,
            )
            if not (args.skip_existing and baseline_metrics_path.exists()):
                print(f"\n[baseline] shot={shot} seed={seed}")
                try:
                    run_command(baseline_command)
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)
                    return 1
            baseline_payload = load_json(baseline_metrics_path)
            if baseline_payload is None:
                print(f"Error: missing baseline metrics: {baseline_metrics_path}", file=sys.stderr)
                return 1
            detail_rows.append(
                extract_result_row(
                    shot=shot,
                    seed=seed,
                    method="baseline_cnn",
                    circuit_type=None,
                    metrics_payload=baseline_payload,
                )
            )
            run_records.append(
                {
                    "shot": shot,
                    "seed": seed,
                    "method": "baseline_cnn",
                    "metrics_path": str(baseline_metrics_path),
                    "checkpoint_path": str(baseline_checkpoint_path),
                }
            )

            if args.include_classical_control:
                control_dir = seed_root / "hybrid_control"
                control_metrics_path = control_dir / "hybrid_control_metrics.json"
                control_command = build_control_command(
                    args,
                    shot=shot,
                    seed=seed,
                    output_dir=control_dir,
                    checkpoint_path=baseline_checkpoint_path,
                )
                if not (args.skip_existing and control_metrics_path.exists()):
                    print(f"\n[control] shot={shot} seed={seed}")
                    try:
                        run_command(control_command)
                    except RuntimeError as exc:
                        print(str(exc), file=sys.stderr)
                        return 1
                control_payload = load_json(control_metrics_path)
                if control_payload is None:
                    print(f"Error: missing control metrics: {control_metrics_path}", file=sys.stderr)
                    return 1
                detail_rows.append(
                    extract_result_row(
                        shot=shot,
                        seed=seed,
                        method="hybrid_control",
                        circuit_type=None,
                        metrics_payload=control_payload,
                    )
                )
                run_records.append(
                    {
                        "shot": shot,
                        "seed": seed,
                        "method": "hybrid_control",
                        "metrics_path": str(control_metrics_path),
                        "checkpoint_path": str(control_dir / "hybrid_control_best.pt"),
                    }
                )

            for circuit_type in args.circuits:
                hybrid_dir = seed_root / f"hybrid_{circuit_type}"
                hybrid_metrics_path = hybrid_dir / "hybrid_qnn_metrics.json"
                hybrid_command = build_hybrid_command(
                    args,
                    shot=shot,
                    seed=seed,
                    circuit_type=circuit_type,
                    output_dir=hybrid_dir,
                    checkpoint_path=baseline_checkpoint_path,
                )
                if not (args.skip_existing and hybrid_metrics_path.exists()):
                    print(f"\n[hybrid] shot={shot} seed={seed} circuit={circuit_type}")
                    try:
                        run_command(hybrid_command)
                    except RuntimeError as exc:
                        print(str(exc), file=sys.stderr)
                        return 1
                hybrid_payload = load_json(hybrid_metrics_path)
                if hybrid_payload is None:
                    print(f"Error: missing hybrid metrics: {hybrid_metrics_path}", file=sys.stderr)
                    return 1
                detail_rows.append(
                    extract_result_row(
                        shot=shot,
                        seed=seed,
                        method=f"hybrid_{circuit_type}",
                        circuit_type=circuit_type,
                        metrics_payload=hybrid_payload,
                    )
                )
                run_records.append(
                    {
                        "shot": shot,
                        "seed": seed,
                        "method": f"hybrid_{circuit_type}",
                        "metrics_path": str(hybrid_metrics_path),
                        "checkpoint_path": str(hybrid_dir / "hybrid_qnn_best.pt"),
                    }
                )

    summary_rows = aggregate_rows(detail_rows)
    detail_csv_path = args.output_dir / "lowshot_detail_rows.csv"
    summary_csv_path = args.output_dir / "lowshot_summary_rows.csv"
    summary_json_path = args.output_dir / "lowshot_summary.json"

    write_csv(detail_csv_path, detail_rows)
    write_csv(summary_csv_path, summary_rows)
    save_json(
        summary_json_path,
        {
            "dataset_root": str(args.dataset_root),
            "output_dir": str(args.output_dir),
            "shots": args.shots,
            "seeds": args.seeds,
            "circuits": args.circuits,
            "include_classical_control": bool(args.include_classical_control),
            "control_branch_input_dim": args.control_branch_input_dim,
            "control_branch_output_dim": args.control_branch_output_dim,
            "train_sample_caps": {f"shot_{shot}": shot for shot in args.shots},
            "val_max_samples_per_class": args.val_max_samples_per_class,
            "test_max_samples_per_class": args.test_max_samples_per_class,
            "full_eval_at_end": bool(args.full_eval_at_end),
            "final_val_max_samples_per_class": args.final_val_max_samples_per_class,
            "final_test_max_samples_per_class": args.final_test_max_samples_per_class,
            "baseline_epochs": args.baseline_epochs,
            "hybrid_epochs": args.hybrid_epochs,
            "run_records": run_records,
            "detail_rows": detail_rows,
            "summary_rows": summary_rows,
        },
    )

    print(f"\nSaved detail CSV: {detail_csv_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved summary JSON: {summary_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

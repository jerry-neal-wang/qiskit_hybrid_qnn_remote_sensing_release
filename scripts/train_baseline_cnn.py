#!/usr/bin/env python3
"""Train the classical CNN baseline for DIOR multiclass ROI classification."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch import nn

    from src.datasets.roi_dataset import validate_processed_dataset_nonempty
    from src.models.cnn_backbone import CNNClassifier
    from src.utils import (
        build_runtime_payload,
        collect_environment_info,
        namespace_to_dict,
        plot_training_curves,
        save_confusion_matrix_figure,
        save_json,
        set_seed,
    )
    from src.utils.training_common import (
        build_class_weights,
        build_roi_dataloaders,
        build_scheduler,
        collect_split_outputs,
        compute_metrics_from_outputs,
        current_learning_rate,
        model_selection_key,
        resolve_device,
        select_best_threshold,
    )

    IMPORT_ERROR: Exception | None = None
except Exception as exc:
    IMPORT_ERROR = exc


DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dior_multiclass_stratified"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_NAME = "baseline_cnn"


def build_split_sample_caps(args: argparse.Namespace) -> dict[str, int | None]:
    """Resolve per-split sample caps from CLI arguments."""
    return {
        "train": args.train_max_samples_per_class,
        "val": args.val_max_samples_per_class,
        "test": args.test_max_samples_per_class,
    }


def build_final_eval_split_sample_caps(
    args: argparse.Namespace,
    training_split_sample_caps: dict[str, int | None],
) -> dict[str, int | None]:
    """Resolve val/test caps used for final metrics after training."""
    final_caps = dict(training_split_sample_caps)
    if args.full_eval_at_end:
        final_caps["val"] = None
        final_caps["test"] = None
    if args.final_val_max_samples_per_class is not None:
        final_caps["val"] = args.final_val_max_samples_per_class
    if args.final_test_max_samples_per_class is not None:
        final_caps["test"] = args.final_test_max_samples_per_class
    return final_caps


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train a CNN baseline on DIOR multiclass ROI data."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=f"Path to processed ROI dataset (default: {DATASET_ROOT})",
    )
    parser.add_argument(
        "--output-dir",
        "--artifacts-dir",
        dest="output_dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help=f"Directory for saved model and metrics (default: {ARTIFACTS_DIR})",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--image-size", type=int, default=64, help="Input image size.")
    parser.add_argument("--feature-dim", type=int, default=64, help="CNN feature dimension.")
    parser.add_argument(
        "--backbone-type",
        type=str,
        default="small_cnn",
        choices=("small_cnn", "resnet18"),
        help="Backbone family used by the baseline classifier.",
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
        help="Input normalization preset applied after ROI tensor conversion.",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout ratio in the CNN head.")
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional global per-class cap applied to train/val/test when split-specific caps are omitted.",
    )
    parser.add_argument(
        "--train-max-samples-per-class",
        type=int,
        default=None,
        help="Optional train split per-class cap. Overrides --max-samples-per-class for train.",
    )
    parser.add_argument(
        "--val-max-samples-per-class",
        type=int,
        default=None,
        help="Optional val split per-class cap. Overrides --max-samples-per-class for val.",
    )
    parser.add_argument(
        "--test-max-samples-per-class",
        type=int,
        default=None,
        help="Optional test split per-class cap. Overrides --max-samples-per-class for test.",
    )
    parser.add_argument(
        "--full-eval-at-end",
        action="store_true",
        help="Rebuild val/test loaders without training-time caps for final reported metrics.",
    )
    parser.add_argument(
        "--final-val-max-samples-per-class",
        type=int,
        default=None,
        help="Optional final val split per-class cap used after training. Overrides --full-eval-at-end for val.",
    )
    parser.add_argument(
        "--final-test-max-samples-per-class",
        type=int,
        default=None,
        help="Optional final test split per-class cap used after training. Overrides --full-eval-at-end for test.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=("cosine", "step", "none"),
        help="Learning-rate scheduler.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Warmup epochs for the scheduler.")
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm. Set <=0 to disable.",
    )
    parser.add_argument(
        "--train-augment",
        type=str,
        default="none",
        choices=("none", "train_light"),
        help="Optional train-only augmentation preset.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=4,
        help="Stop when val_f1 does not improve for this many epochs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=MODEL_NAME,
        help="Human-readable experiment label saved into artifacts.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Mark outputs as smoke-only and not for paper metrics.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Execution device. Use auto to prefer CUDA, then MPS, then CPU.",
    )
    return parser.parse_args()


def validate_environment(
    dataset_root: Path,
    max_samples_per_class: int | None,
    split_max_samples_per_class: dict[str, int | None] | None,
) -> None:
    """Validate dependencies and dataset presence."""
    if IMPORT_ERROR is not None:
        raise ImportError(
            "PyTorch and local recognition mainline modules are required for train_baseline_cnn.py."
        ) from IMPORT_ERROR
    validate_processed_dataset_nonempty(
        dataset_root=dataset_root,
        max_samples_per_class=max_samples_per_class,
        split_max_samples_per_class=split_max_samples_per_class,
    )


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        batch_size = int(labels.size(0))
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    return running_loss / max(1, sample_count)


def main() -> int:
    """Script entry point."""
    args = parse_args()

    try:
        split_sample_caps = build_split_sample_caps(args)
        validate_environment(
            dataset_root=args.dataset_root,
            max_samples_per_class=args.max_samples_per_class,
            split_max_samples_per_class=split_sample_caps,
        )
        device = resolve_device(args.device)
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    set_seed(args.seed)
    run_start = time.perf_counter()
    split_loaders, split_sizes, split_class_counts, label_map = build_roi_dataloaders(
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_samples_per_class=args.max_samples_per_class,
        split_max_samples_per_class=split_sample_caps,
        seed=args.seed,
        train_augment=args.train_augment,
        input_normalization=args.input_normalization,
    )
    final_eval_split_sample_caps = build_final_eval_split_sample_caps(args, split_sample_caps)
    class_names = list(label_map.keys())
    num_classes = len(class_names)

    model = CNNClassifier(
        feature_dim=args.feature_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        backbone_type=args.backbone_type,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    train_class_weights = build_class_weights(
        split_class_counts,
        class_names=class_names,
        device=device,
    )
    train_criterion = nn.CrossEntropyLoss(weight=train_class_weights)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_name=args.scheduler,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_key: tuple[float, float, float] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    stop_reason = "max_epochs"
    epoch_runtime_seconds: list[float] = []
    history = {
        "epoch": [],
        "phase": [],
        "learning_rate": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_roc_auc": [],
        "val_threshold": [],
    }

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate = current_learning_rate(optimizer)
        train_loss = train_one_epoch(
            model=model,
            loader=split_loaders["train"],
            criterion=train_criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        val_outputs = collect_split_outputs(model, split_loaders["val"], eval_criterion, device=device)
        val_threshold = select_best_threshold(val_outputs["targets"], val_outputs["positive_scores"])
        val_metrics = compute_metrics_from_outputs(
            val_outputs,
            class_names=class_names,
            threshold=val_threshold,
        )
        epoch_runtime = time.perf_counter() - epoch_start
        epoch_runtime_seconds.append(epoch_runtime)
        val_key = model_selection_key(val_metrics)

        improved = best_val_key is None or val_key > best_val_key
        if improved:
            best_val_key = val_key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history["epoch"].append(epoch)
        history["phase"].append("joint")
        history["learning_rate"].append(float(learning_rate))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_precision"].append(float(val_metrics["precision"]))
        history["val_recall"].append(float(val_metrics["recall"]))
        history["val_f1"].append(float(val_metrics["f1"]))
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        history["val_threshold"].append(None if val_threshold is None else float(val_threshold))

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"lr={learning_rate:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"epoch_time={epoch_runtime:.2f}s"
        )

        if scheduler is not None:
            scheduler.step()

        if epochs_without_improvement >= args.early_stopping_patience:
            stop_reason = f"early_stopping_patience_{args.early_stopping_patience}"
            print(f"Early stopping triggered at epoch {epoch:02d}.")
            break

    if best_state is None:
        print("Error: training did not produce a valid checkpoint.", file=sys.stderr)
        return 1

    model.load_state_dict(best_state)
    train_outputs = collect_split_outputs(model, split_loaders["train"], eval_criterion, device=device)
    final_eval_loaders = split_loaders
    final_eval_split_sizes = split_sizes
    final_eval_split_class_counts = split_class_counts
    if final_eval_split_sample_caps != split_sample_caps:
        final_eval_loaders, final_eval_split_sizes, final_eval_split_class_counts, _ = build_roi_dataloaders(
            dataset_root=args.dataset_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            max_samples_per_class=args.max_samples_per_class,
            split_max_samples_per_class=final_eval_split_sample_caps,
            seed=args.seed,
            train_augment="none",
            input_normalization=args.input_normalization,
        )

    val_outputs = collect_split_outputs(model, final_eval_loaders["val"], eval_criterion, device=device)
    test_outputs = collect_split_outputs(model, final_eval_loaders["test"], eval_criterion, device=device)
    best_threshold = select_best_threshold(val_outputs["targets"], val_outputs["positive_scores"])
    train_metrics = compute_metrics_from_outputs(
        train_outputs,
        class_names=class_names,
        threshold=best_threshold,
    )
    val_metrics = compute_metrics_from_outputs(
        val_outputs,
        class_names=class_names,
        threshold=best_threshold,
    )
    test_metrics = compute_metrics_from_outputs(
        test_outputs,
        class_names=class_names,
        threshold=best_threshold,
    )
    runtime_seconds = time.perf_counter() - run_start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{MODEL_NAME}_best.pt"
    metrics_path = args.output_dir / f"{MODEL_NAME}_metrics.json"
    history_path = args.output_dir / f"{MODEL_NAME}_history.json"
    curve_path = args.output_dir / f"{MODEL_NAME}_training_curves.png"
    confusion_path = args.output_dir / f"{MODEL_NAME}_confusion_matrix.png"
    runtime_path = args.output_dir / f"{MODEL_NAME}_runtime.json"
    config_path = args.output_dir / f"{MODEL_NAME}_config.json"
    test_metrics_path = args.output_dir / f"{MODEL_NAME}_test_metrics.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": MODEL_NAME,
            "experiment_name": args.experiment_name,
            "smoke_only": bool(args.smoke_only),
            "not_for_paper_metrics": bool(args.smoke_only),
            "best_epoch": best_epoch,
            "model_config": {
                "image_size": args.image_size,
                "feature_dim": args.feature_dim,
                "num_classes": num_classes,
                "class_names": class_names,
                "dropout": args.dropout,
            },
        },
        checkpoint_path,
    )
    plot_training_curves(history, curve_path, title="Baseline CNN Training Curves")
    save_confusion_matrix_figure(
        test_metrics["confusion_matrix"],
        confusion_path,
        title="Baseline CNN Confusion Matrix",
        labels=class_names,
    )

    config_payload = {
        "model_name": MODEL_NAME,
        "experiment_name": args.experiment_name,
        "smoke_only": bool(args.smoke_only),
        "not_for_paper_metrics": bool(args.smoke_only),
        "cli_args": namespace_to_dict(args),
        "dataset_root": args.dataset_root,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "device": str(device),
        "label_map": label_map,
        "class_names": class_names,
        "num_classes": num_classes,
        "split_sizes": split_sizes,
        "split_class_counts": split_class_counts,
        "split_sample_caps": split_sample_caps,
        "final_eval_split_sizes": final_eval_split_sizes,
        "final_eval_split_class_counts": final_eval_split_class_counts,
        "final_eval_split_sample_caps": final_eval_split_sample_caps,
        "model_config": {
            "image_size": args.image_size,
            "feature_dim": args.feature_dim,
            "num_classes": num_classes,
            "class_names": class_names,
            "dropout": args.dropout,
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "gradient_clip_norm": args.gradient_clip_norm,
            "early_stopping_patience": args.early_stopping_patience,
            "train_augment": args.train_augment,
            "smoke_only": bool(args.smoke_only),
            "train_class_weights": train_class_weights.detach().cpu().tolist(),
            "decision_threshold": best_threshold,
            "split_sample_caps": split_sample_caps,
            "full_eval_at_end": bool(args.full_eval_at_end),
            "final_eval_split_sample_caps": final_eval_split_sample_caps,
        },
        "environment": collect_environment_info(("numpy", "torch", "sklearn")),
    }
    runtime_payload = build_runtime_payload(
        model_name=MODEL_NAME,
        seed=args.seed,
        runtime_seconds=runtime_seconds,
        epoch_runtime_seconds=epoch_runtime_seconds,
        best_epoch=best_epoch,
        device=str(device),
        extra={
            "experiment_name": args.experiment_name,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history["epoch"]),
            "max_samples_per_class": args.max_samples_per_class,
            "smoke_only": bool(args.smoke_only),
            "not_for_paper_metrics": bool(args.smoke_only),
            "stop_reason": stop_reason,
        },
    )
    metrics_payload = {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "input_resolution": f"{args.image_size}x{args.image_size}",
        "model_name": MODEL_NAME,
        "experiment_name": args.experiment_name,
        "smoke_only": bool(args.smoke_only),
        "not_for_paper_metrics": bool(args.smoke_only),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "epochs_completed": len(history["epoch"]),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_epochs": args.warmup_epochs,
        "gradient_clip_norm": args.gradient_clip_norm,
        "dropout": args.dropout,
        "feature_dim": args.feature_dim,
        "label_map": label_map,
        "class_names": class_names,
        "num_classes": num_classes,
        "train_augment": args.train_augment,
        "max_samples_per_class": args.max_samples_per_class,
        "split_sample_caps": split_sample_caps,
        "final_eval_split_sample_caps": final_eval_split_sample_caps,
        "best_epoch": best_epoch,
        "decision_threshold": best_threshold,
        "runtime_seconds": runtime_seconds,
        "stop_reason": stop_reason,
        "split_sizes": split_sizes,
        "split_class_counts": split_class_counts,
        "final_eval_split_sizes": final_eval_split_sizes,
        "final_eval_split_class_counts": final_eval_split_class_counts,
        "results": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "training_curve_path": str(curve_path),
        "confusion_matrix_path": str(confusion_path),
        "runtime_path": str(runtime_path),
        "config_path": str(config_path),
        "test_metrics_path": str(test_metrics_path),
    }
    test_metrics_payload = {
        "model_name": MODEL_NAME,
        "experiment_name": args.experiment_name,
        "smoke_only": bool(args.smoke_only),
        "not_for_paper_metrics": bool(args.smoke_only),
        "seed": args.seed,
        "split": "test",
        "metrics": test_metrics,
        "best_epoch": best_epoch,
        "decision_threshold": best_threshold,
        "checkpoint_path": str(checkpoint_path),
        "confusion_matrix_path": str(confusion_path),
    }

    history["smoke_only"] = bool(args.smoke_only)
    history["not_for_paper_metrics"] = bool(args.smoke_only)

    save_json(config_path, config_payload)
    save_json(history_path, history)
    save_json(runtime_path, runtime_payload)
    save_json(metrics_path, metrics_payload)
    save_json(test_metrics_path, test_metrics_payload)

    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved training curves: {curve_path}")
    print(f"Saved test metrics: {test_metrics_path}")
    print(f"Saved runtime: {runtime_path}")
    print(f"Saved config: {config_path}")
    print(f"Saved confusion matrix: {confusion_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

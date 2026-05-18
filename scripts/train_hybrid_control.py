#!/usr/bin/env python3
"""Train the fusion-style CNN + classical-control classifier for DIOR ROI data."""

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
    from src.models.hybrid_model import HybridFusionControlClassifier
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
        trainable_parameter_names,
        select_best_threshold,
    )

    IMPORT_ERROR: Exception | None = None
except Exception as exc:
    IMPORT_ERROR = exc


DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "dior_multiclass_stratified"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_NAME = "hybrid_control"


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
        description="Train a fusion-style CNN + classical control classifier on DIOR ROI data."
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
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the backbone learning rate during optimization.",
    )
    parser.add_argument("--image-size", type=int, default=64, help="Input image size.")
    parser.add_argument(
        "--backbone-feature-dim",
        type=int,
        default=64,
        help="CNN backbone feature width before the fusion head.",
    )
    parser.add_argument(
        "--backbone-type",
        type=str,
        default="small_cnn",
        choices=("small_cnn", "resnet18"),
        help="Backbone family used by the control model.",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Initialize the control backbone from torchvision pretrained weights when supported.",
    )
    parser.add_argument(
        "--input-normalization",
        type=str,
        default="none",
        choices=("none", "imagenet"),
        help="Input normalization preset applied after ROI tensor conversion.",
    )
    parser.add_argument(
        "--branch-input-dim",
        type=int,
        default=4,
        help="Control-branch projector output width.",
    )
    parser.add_argument(
        "--branch-output-dim",
        type=int,
        default=4,
        help="Control-branch bounded output width.",
    )
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="concat",
        choices=("concat", "add"),
        help="Hybrid fusion mode.",
    )
    parser.add_argument(
        "--classical-head-weight",
        type=float,
        default=1.0,
        help="Residual weight applied to the classical branch logits.",
    )
    parser.add_argument(
        "--control-head-weight",
        type=float,
        default=1.0,
        help="Weight applied to the control branch contribution.",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=2,
        help="Number of warm-start epochs with the backbone frozen.",
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
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout ratio in the backbone and projector.",
    )
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--pretrained-cnn-checkpoint",
        type=Path,
        default=None,
        help="Baseline CNN checkpoint used to initialize the backbone. Required by default.",
    )
    parser.add_argument(
        "--allow-random-backbone-init",
        action="store_true",
        help="Disable the default requirement to initialize the backbone from a baseline CNN checkpoint.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Mark outputs as smoke-only and not for paper metrics.",
    )
    return parser.parse_args()


def validate_environment(
    dataset_root: Path,
    max_samples_per_class: int | None,
    split_max_samples_per_class: dict[str, int | None] | None,
    pretrained_cnn_checkpoint: Path | None,
    allow_random_backbone_init: bool,
) -> None:
    """Validate dependencies and dataset presence."""
    if IMPORT_ERROR is not None:
        raise ImportError(
            "PyTorch and local recognition modules are required for train_hybrid_control.py."
        ) from IMPORT_ERROR
    if pretrained_cnn_checkpoint is None and not allow_random_backbone_init:
        raise ValueError(
            "Control training now requires an explicit --pretrained-cnn-checkpoint for fair comparison. "
            "Pass --allow-random-backbone-init only for dedicated ablations."
        )
    validate_processed_dataset_nonempty(
        dataset_root=dataset_root,
        max_samples_per_class=max_samples_per_class,
        split_max_samples_per_class=split_max_samples_per_class,
    )


def resolve_pretrained_checkpoint(
    checkpoint_path: Path | None,
    allow_random_backbone_init: bool,
) -> Path | None:
    """Validate the explicitly supplied baseline checkpoint."""
    if checkpoint_path is None:
        if allow_random_backbone_init:
            return None
        raise ValueError("Missing required --pretrained-cnn-checkpoint.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained CNN checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def initialize_backbone_from_baseline(
    model: HybridFusionControlClassifier,
    checkpoint_path: Path | None,
) -> tuple[Path | None, list[str]]:
    """Load matching backbone weights from a baseline CNN checkpoint when available."""
    if checkpoint_path is None:
        return None, []

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    loaded_keys: list[str] = []

    for key, value in state_dict.items():
        if not key.startswith("backbone."):
            continue
        if key not in model_state or model_state[key].shape != value.shape:
            continue
        model_state[key] = value
        loaded_keys.append(key)

    model.load_state_dict(model_state, strict=False)
    return checkpoint_path, loaded_keys


def set_backbone_trainable(model: HybridFusionControlClassifier, trainable: bool) -> None:
    """Freeze or unfreeze the CNN backbone."""
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


def build_optimizer(
    model: HybridFusionControlClassifier,
    *,
    learning_rate: float,
    weight_decay: float,
    backbone_lr_scale: float,
) -> torch.optim.Optimizer:
    """Create Adam optimizer with a dedicated backbone learning rate group."""
    backbone_lr_scale = max(0.0, float(backbone_lr_scale))
    backbone_params = list(model.backbone.parameters())
    backbone_param_ids = {id(parameter) for parameter in backbone_params}
    non_backbone_params = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_param_ids
    ]
    param_groups = []
    if non_backbone_params:
        param_groups.append(
            {
                "params": non_backbone_params,
                "lr": float(learning_rate),
            }
        )
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(learning_rate) * backbone_lr_scale,
            }
        )
    return torch.optim.Adam(
        param_groups,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def get_optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> tuple[float, float]:
    """Return head and backbone learning rates from optimizer param groups."""
    head_lr = float(optimizer.param_groups[0]["lr"])
    backbone_lr = float(optimizer.param_groups[-1]["lr"])
    return head_lr, backbone_lr


def train_one_epoch(
    model: HybridFusionControlClassifier,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float,
    backbone_trainable: bool,
) -> float:
    """Run one training epoch on CPU."""
    model.train()
    if not backbone_trainable:
        model.backbone.eval()

    running_loss = 0.0
    sample_count = 0

    for images, labels in loader:
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
            pretrained_cnn_checkpoint=args.pretrained_cnn_checkpoint,
            allow_random_backbone_init=args.allow_random_backbone_init,
        )
        pretrained_checkpoint = resolve_pretrained_checkpoint(
            args.pretrained_cnn_checkpoint,
            args.allow_random_backbone_init,
        )
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

    model = HybridFusionControlClassifier(
        backbone_feature_dim=args.backbone_feature_dim,
        backbone_type=args.backbone_type,
        pretrained_backbone=args.pretrained_backbone,
        branch_input_dim=args.branch_input_dim,
        branch_output_dim=args.branch_output_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        fusion_mode=args.fusion_mode,
        classical_head_weight=args.classical_head_weight,
        control_head_weight=args.control_head_weight,
    )
    loaded_checkpoint_path, loaded_backbone_keys = initialize_backbone_from_baseline(
        model=model,
        checkpoint_path=pretrained_checkpoint,
    )
    if loaded_checkpoint_path is not None:
        print(
            f"Initialized backbone from {loaded_checkpoint_path} "
            f"({len(loaded_backbone_keys)} tensors loaded)"
        )

    train_class_weights = build_class_weights(
        split_class_counts,
        class_names=class_names,
    )
    train_criterion = nn.CrossEntropyLoss(weight=train_class_weights)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        backbone_lr_scale=args.backbone_lr_scale,
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
        "backbone_trainable": [],
        "learning_rate": [],
        "backbone_learning_rate": [],
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
        backbone_trainable = epoch > args.freeze_backbone_epochs
        set_backbone_trainable(model, backbone_trainable)

        epoch_start = time.perf_counter()
        learning_rate = current_learning_rate(optimizer)
        _, backbone_learning_rate = get_optimizer_learning_rates(optimizer)
        train_loss = train_one_epoch(
            model=model,
            loader=split_loaders["train"],
            criterion=train_criterion,
            optimizer=optimizer,
            gradient_clip_norm=args.gradient_clip_norm,
            backbone_trainable=backbone_trainable,
        )
        val_outputs = collect_split_outputs(model, split_loaders["val"], eval_criterion)
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

        phase = "joint" if backbone_trainable else "frozen_backbone"
        history["epoch"].append(epoch)
        history["phase"].append(phase)
        history["backbone_trainable"].append(bool(backbone_trainable))
        history["learning_rate"].append(float(learning_rate))
        history["backbone_learning_rate"].append(float(backbone_learning_rate))
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
            f"phase={phase} | "
            f"lr={learning_rate:.6f} | "
            f"backbone_lr={backbone_learning_rate:.6f} | "
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
    set_backbone_trainable(model, True)
    train_outputs = collect_split_outputs(model, split_loaders["train"], eval_criterion)
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

    val_outputs = collect_split_outputs(model, final_eval_loaders["val"], eval_criterion)
    test_outputs = collect_split_outputs(model, final_eval_loaders["test"], eval_criterion)
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
                "backbone_feature_dim": args.backbone_feature_dim,
                "num_classes": num_classes,
                "class_names": class_names,
                "branch_input_dim": args.branch_input_dim,
                "branch_output_dim": args.branch_output_dim,
                "dropout": args.dropout,
                "fusion_mode": args.fusion_mode,
                "classical_head_weight": args.classical_head_weight,
                "control_head_weight": args.control_head_weight,
            },
            "pretrained_cnn_checkpoint": None if loaded_checkpoint_path is None else str(loaded_checkpoint_path),
            "loaded_backbone_keys": loaded_backbone_keys,
        },
        checkpoint_path,
    )
    plot_training_curves(history, curve_path, title="Hybrid Control Training Curves")
    save_confusion_matrix_figure(
        test_metrics["confusion_matrix"],
        confusion_path,
        title="Hybrid Control Confusion Matrix",
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
        "device": "cpu",
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
            "backbone_feature_dim": args.backbone_feature_dim,
            "num_classes": num_classes,
            "class_names": class_names,
            "branch_input_dim": args.branch_input_dim,
            "branch_output_dim": args.branch_output_dim,
            "dropout": args.dropout,
            "fusion_mode": args.fusion_mode,
            "classical_head_weight": args.classical_head_weight,
            "control_head_weight": args.control_head_weight,
        },
        "optimization": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "warmup_epochs": args.warmup_epochs,
            "gradient_clip_norm": args.gradient_clip_norm,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "train_augment": args.train_augment,
            "smoke_only": bool(args.smoke_only),
            "train_class_weights": train_class_weights.detach().cpu().tolist(),
            "decision_threshold": best_threshold,
            "split_sample_caps": split_sample_caps,
            "full_eval_at_end": bool(args.full_eval_at_end),
            "final_eval_split_sample_caps": final_eval_split_sample_caps,
        },
        "pretrained_backbone": {
            "checkpoint_path": None if loaded_checkpoint_path is None else str(loaded_checkpoint_path),
            "loaded_key_count": len(loaded_backbone_keys),
            "loaded_keys": loaded_backbone_keys,
        },
        "trainable_parameter_names": trainable_parameter_names(model),
        "environment": collect_environment_info(("numpy", "torch", "sklearn")),
    }
    runtime_payload = build_runtime_payload(
        model_name=MODEL_NAME,
        seed=args.seed,
        runtime_seconds=runtime_seconds,
        epoch_runtime_seconds=epoch_runtime_seconds,
        best_epoch=best_epoch,
        device="cpu",
        extra={
            "experiment_name": args.experiment_name,
            "epochs_requested": args.epochs,
            "epochs_completed": len(history["epoch"]),
            "max_samples_per_class": args.max_samples_per_class,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "loaded_backbone_key_count": len(loaded_backbone_keys),
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
        "device": "cpu",
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
        "backbone_feature_dim": args.backbone_feature_dim,
        "label_map": label_map,
        "class_names": class_names,
        "num_classes": num_classes,
        "branch_input_dim": args.branch_input_dim,
        "branch_output_dim": args.branch_output_dim,
        "fusion_mode": args.fusion_mode,
        "classical_head_weight": args.classical_head_weight,
        "control_head_weight": args.control_head_weight,
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
        "pretrained_backbone_checkpoint": None if loaded_checkpoint_path is None else str(loaded_checkpoint_path),
        "loaded_backbone_key_count": len(loaded_backbone_keys),
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

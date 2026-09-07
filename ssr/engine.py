"""General image-dataset SSR/LSL training and checkpoint orchestration."""

from __future__ import annotations

import random
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from math import cos, pi
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from label_wave import LabelWaveRun
from log.checkpoint import SELECTION_METRICS, CheckpointManager
from log.common import JsonlWriter, write_config
from setting.config import ExperimentConfig
from setting.data import ExperimentData, build_experiment_data
from setting.model import SSRNetworks, build_ssr_networks, calibrate_batch_norm
from setting.precision import PrecisionPolicy, full_precision
from ssr.evaluation import evaluate_epoch, predict_training_labels, selection_metrics
from ssr.checkpoint_evaluation import evaluate_checkpoints
from ssr.metrics import evaluate_classification, metrics_from_confusion_matrix
from ssr.sampler import ClassBalancedSampler
from ssr.selection import SelectionResult
from ssr.trainer import train_epoch


@dataclass(frozen=True, slots=True)
class EpochLoaders:
    evaluation: DataLoader
    all_samples: DataLoader
    validation: DataLoader | None
    test: DataLoader | None


def seed_everything(seed: int) -> None:
    """전역 config seed를 Python, NumPy, PyTorch, CUDA에 동일 적용."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_execution(
    config: ExperimentConfig, networks: SSRNetworks, device: torch.device,
) -> None:
    """Enable CUDA training optimizations without reducing selection precision."""
    policy = PrecisionPolicy.from_config(config.training, device)
    if device.type != "cuda":
        return
    torch.backends.cudnn.deterministic = config.runtime.deterministic
    torch.backends.cudnn.benchmark = not config.runtime.deterministic
    # BF16 is scoped to training. k-NN/threshold decisions stay FP32, not TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if policy.channels_last:
        networks.encoder.to(memory_format=torch.channels_last)


def _timestamp(device: torch.device) -> float:
    # Only synchronize at phase boundaries, never for each timed training step.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter()


def _loader_options(
    config: ExperimentConfig,
    device: torch.device,
    *,
    evaluation: bool = False,
) -> dict[str, Any]:
    workers = (
        config.training.eval_num_workers if evaluation else config.training.num_workers
    )
    options: dict[str, Any] = {
        "batch_size": (
            config.training.eval_batch_size if evaluation else config.training.batch_size
        ),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        options.update(
            prefetch_factor=config.training.prefetch_factor,
            # Do not keep evaluation worker pools resident during training.
            persistent_workers=config.training.persistent_workers and not evaluation,
        )
    return options


def _build_epoch_loaders(
    data: ExperimentData,
    config: ExperimentConfig,
    device: torch.device,
) -> EpochLoaders:
    options = _loader_options(config, device)
    eval_options = _loader_options(config, device, evaluation=True)
    return EpochLoaders(
        evaluation=DataLoader(data.evaluation_train, shuffle=False, **eval_options),
        all_samples=DataLoader(
            data.all_train,
            shuffle=True,
            drop_last=True,
            **options,
        ),
        validation=(
            DataLoader(data.validation, shuffle=False, **eval_options)
            if data.validation is not None else None
        ),
        test=(
            DataLoader(data.test, shuffle=False, **eval_options)
            if data.test is not None else None
        ),
    )


def _build_calibration_loader(
    data: ExperimentData,
    config: ExperimentConfig,
    device: torch.device,
) -> DataLoader:
    """Build a one-pass, non-augmented loader with isolated worker RNG state."""
    options = _loader_options(config, device)
    options["persistent_workers"] = False
    return DataLoader(
        data.calibration_train,
        shuffle=False,
        generator=torch.Generator().manual_seed(config.seed),
        **options,
    )


def _build_selected_loader(
    data: ExperimentData,
    selection: SelectionResult,
    config: ExperimentConfig,
    device: torch.device,
) -> DataLoader:
    subset = Subset(data.selected_train, selection.selected_indices.detach().cpu())
    sampler = ClassBalancedSampler(
        selection.modified_labels[selection.selected_indices].detach().cpu(),
        num_classes=data.num_classes,
    )
    options = _loader_options(config, device)
    # This loader is rebuilt every epoch, so persistent workers only delay
    # process cleanup and provide no reuse benefit.
    options["persistent_workers"] = False
    loader = DataLoader(
        subset,
        sampler=sampler,
        drop_last=True,
        **options,
    )
    if len(loader) == 0:
        raise RuntimeError(
            "The class-balanced selected loader has no full batch; "
            "SSR cannot perform its supervised update."
        )
    return loader


def _build_optimizer(
    networks: SSRNetworks,
    config: ExperimentConfig,
) -> Optimizer:
    encoder_group: dict[str, Any] = {"params": networks.encoder.parameters()}
    if config.training.encoder_learning_rate is not None:
        encoder_group["lr"] = config.training.encoder_learning_rate
    parameter_groups = [
        encoder_group,
        {"params": networks.classifier.parameters()},
        {"params": networks.projector.parameters()},
        {"params": networks.predictor.parameters()},
    ]
    if config.training.optimizer == "adamw":
        use_fused = config.training.fused_optimizer and all(
            parameter.is_cuda
            for module in networks.all_modules()
            for parameter in module.parameters()
        )
        return AdamW(
            parameter_groups,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            fused=True if use_fused else None,
        )
    return SGD(
        parameter_groups,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        momentum=config.training.momentum,
    )


def _build_scheduler(
    optimizer: Optimizer,
    config: ExperimentConfig,
) -> LambdaLR:
    """Apply one cosine multiplier to every parameter group's own base LR."""

    minimum_ratio = config.training.scheduler_eta_min_ratio
    total_epochs = config.training.epochs

    def cosine_multiplier(completed_epochs: int) -> float:
        progress = min(completed_epochs, total_epochs) / total_epochs
        return minimum_ratio + (1.0 - minimum_ratio) * (
            1.0 + cos(pi * progress)
        ) / 2.0

    return LambdaLR(optimizer, lr_lambda=cosine_multiplier)


def _prepare_config(config: ExperimentConfig) -> ExperimentConfig:
    """Reserve a fresh output directory before writing any run artifact."""

    config.validate()
    automatic_id = config.runtime.run_id is None
    # Create the shared config directory once so invalid parent paths fail
    # immediately instead of being mistaken for a timestamp collision.
    (config.runtime.output_root / config.run_name).mkdir(parents=True, exist_ok=True)
    while True:
        if automatic_id:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            resolved = replace(config, runtime=replace(config.runtime, run_id=run_id))
        else:
            resolved = config
        try:
            resolved.run_dir.mkdir(exist_ok=False)
        except FileExistsError as error:
            if automatic_id:
                continue
            raise RuntimeError(
                f"Run directory already exists: {resolved.run_dir}. "
                "Choose another runtime.run_id or use None for a fresh run."
            ) from error
        return resolved


def run(config: ExperimentConfig) -> float | None:
    """warm-up 없이 매 epoch relabel -> select -> optional LSL -> train."""
    config = _prepare_config(config)
    print(f"run_dir={config.run_dir}")
    write_config(config, config.run_dir)
    device = config.resolve_device()
    seed_everything(config.seed)
    precision = PrecisionPolicy.from_config(config.training, device)

    data = build_experiment_data(
        config.data,
        structural_labels_enabled=config.structural_labels.enabled,
        augmentation=config.augmentation,
        report_path=config.run_dir / "data_audit.jsonl",
    )
    training_samples = int(data.noisy_labels.numel())
    if training_samples < config.training.batch_size:
        raise ValueError(
            f"training.batch_size={config.training.batch_size} exceeds "
            f"training samples={training_samples}; the drop-last loader would be empty."
        )
    if config.ssr.neighbors > training_samples:
        raise ValueError(
            f"ssr.neighbors={config.ssr.neighbors} exceeds "
            f"training samples={training_samples}."
        )
    if (
        config.structural_labels.enabled
        and config.structural_labels.neighbors > training_samples
    ):
        raise ValueError(
            f"structural_labels.neighbors={config.structural_labels.neighbors} "
            f"exceeds training samples={training_samples}."
        )
    loaders = _build_epoch_loaders(data, config, device)
    noisy_labels = data.noisy_labels.to(device)
    networks = build_ssr_networks(config.model, device, data.num_classes)
    configure_execution(config, networks, device)
    if (
        config.model.calibrate_initial_batch_norm
        and not config.model.pretrained
    ):
        with full_precision(device):
            calibration_batches = calibrate_batch_norm(
                _build_calibration_loader(data, config, device),
                networks.encoder,
                device,
            )
        print(f"initial_batch_norm_calibration_batches={calibration_batches}")
    optimizer = _build_optimizer(networks, config)
    checkpoints = CheckpointManager(config.run_dir, networks, optimizer, config)
    scheduler = _build_scheduler(optimizer, config)

    print(f"device={device} run={config.run_name}")
    print(
        f"train_precision={'bf16' if precision.amp else 'fp32'} "
        f"selection_precision=fp32 channels_last={precision.channels_last} "
        f"fused_adamw={bool(optimizer.defaults.get('fused', False))} "
        f"batch={config.training.batch_size} eval_batch={config.training.eval_batch_size} "
        f"workers={config.training.num_workers} eval_workers={config.training.eval_num_workers}"
    )
    if loaders.validation is None:
        print("validation=disabled; best checkpoints omitted; last.pt saved every epoch.")
    best_accuracy: float | None = None
    best_scores: dict[str, float | None] = {metric: None for metric in SELECTION_METRICS}
    best_epochs: dict[str, int | None] = {metric: None for metric in SELECTION_METRICS}
    last_validation: dict[str, Any] | None = None
    last_completed_epoch = -1
    stopped_by_label_wave = False
    label_wave = (
        LabelWaveRun(config.label_wave, config.run_dir, checkpoints)
        if config.label_wave.enabled
        else None
    )

    with ExitStack() as stack:
        metrics_writer = stack.enter_context(
            JsonlWriter(config.run_dir / "metrics.jsonl")
        )
        if label_wave is not None:
            stack.enter_context(label_wave)
        for epoch in range(config.training.epochs):
            epoch_started = _timestamp(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            selection_started = _timestamp(device)
            supervision = evaluate_epoch(
                loaders.evaluation,
                networks,
                noisy_labels,
                config,
                device,
            )
            selection_seconds = _timestamp(device) - selection_started
            if label_wave is not None:
                observation = label_wave.observe(
                    supervision.predictions,
                    completed_epochs=epoch,
                    validation_accuracy=last_validation["accuracy"] if last_validation else None,
                    validation_metrics=last_validation,
                )
                if observation.should_stop and config.label_wave.stop_training:
                    stopped_by_label_wave = True
                    del supervision
                    break

            training_started = _timestamp(device)
            selected_loader = _build_selected_loader(
                data,
                supervision.selection,
                config,
                device,
            )
            encoder_learning_rate = float(optimizer.param_groups[0]["lr"])
            learning_rate = float(optimizer.param_groups[1]["lr"])
            losses = train_epoch(
                selected_loader,
                loaders.all_samples,
                supervision.selection.modified_labels,
                supervision.structural_targets,
                networks,
                optimizer,
                config,
                device,
                epoch,
            )
            training_seconds = _timestamp(device) - training_started
            train_steps = len(loaders.all_samples)
            samples_per_second = (
                train_steps * config.training.batch_size / max(training_seconds, 1e-9)
            )
            scheduler.step()
            last_completed_epoch = epoch
            validation = None
            validation_started = _timestamp(device)
            try:
                if loaders.validation is not None:
                    validation = evaluate_classification(
                        loaders.validation, networks, device, description="Validation",
                        channels_last=config.training.channels_last,
                        class_names=data.class_names,
                    )
            finally:
                validation_seconds = _timestamp(device) - validation_started
                # A validation error must not discard the just-trained model.
                # Save once per completed epoch, even when evaluation fails.
                checkpoints.save(
                    "last.pt",
                    epoch,
                    metrics={
                        "validation_accuracy": validation["accuracy"] if validation else None,
                        "validation": validation,
                    },
                    selection={"method": "last", "criterion": "last_epoch"},
                )
            last_validation = validation
            current_accuracy = validation["accuracy"] if validation else None
            if current_accuracy is not None:
                best_accuracy = max(best_accuracy if best_accuracy is not None else -1.0, current_accuracy)
                for metric in SELECTION_METRICS:
                    score = validation[metric]
                    if best_scores[metric] is None or score > best_scores[metric]:
                        best_scores[metric], best_epochs[metric] = score, epoch + 1
                        checkpoints.save(
                            f"best_{metric}.pt", epoch,
                            metrics={"validation_accuracy": current_accuracy, "validation": validation},
                            selection={"method": "best", "split": "validation", "criterion": metric, "score": score},
                        )
            # These are comparisons to provided noisy labels, NOT clean accuracy.
            prediction_counts = torch.bincount(
                supervision.predictions, minlength=data.num_classes,
            )
            observed_confusion = torch.bincount(
                noisy_labels * data.num_classes + supervision.predictions,
                minlength=data.num_classes**2,
            ).reshape(data.num_classes, data.num_classes)
            observed_metrics = metrics_from_confusion_matrix(observed_confusion, class_names=data.class_names)
            values = {
                "epoch": epoch,
                "completed_epochs": epoch + 1,
                "metric_units": "rates in [0, 1]",
                "learning_rate": learning_rate,
                "encoder_learning_rate": encoder_learning_rate,
                **asdict(losses),
                "validation_accuracy": current_accuracy,
                "validation_balanced_accuracy": validation["balanced_accuracy"] if validation else None,
                "validation_macro_f1": validation["macro_f1"] if validation else None,
                "validation": validation,
                "best_validation_accuracy": best_accuracy,
                "best_accuracy": best_accuracy,
                "best_validation_balanced_accuracy": best_scores["balanced_accuracy"],
                "best_validation_macro_f1": best_scores["macro_f1"],
                "best_epochs": dict(best_epochs),
                "train_observed_label_metrics_before_update": observed_metrics,
                "prediction_class_counts": prediction_counts.tolist(),
                "prediction_dominant_class_fraction": float(prediction_counts.max().item() / training_samples),
                "total_loss": (
                    losses.supervised_loss
                    + config.ssr.feature_consistency_weight * losses.feature_consistency_loss
                    + config.structural_labels.loss_weight * (losses.structural_loss or 0.0)
                ),
                "selection_seconds": selection_seconds,
                "training_seconds": training_seconds,
                "validation_seconds": validation_seconds,
                "epoch_seconds": _timestamp(device) - epoch_started,
                "train_steps": train_steps,
                "train_samples_per_second": samples_per_second,
                "cuda_peak_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda" else None
                ),
                "cuda_peak_reserved_gib": (
                    torch.cuda.max_memory_reserved(device) / 1024**3
                    if device.type == "cuda" else None
                ),
                **selection_metrics(
                    supervision.selection,
                    noisy_labels,
                    num_classes=config.data.num_classes,
                ),
            }
            metrics_writer.write(values)
            accuracy_summary = (
                f"val_acc={current_accuracy:.2%} "
                f"val_bal_acc={validation['balanced_accuracy']:.2%} "
                f"val_macro_f1={validation['macro_f1']:.2%} "
                f"best_bal_acc={best_scores['balanced_accuracy']:.2%} "
                f"best_macro_f1={best_scores['macro_f1']:.2%}"
                if current_accuracy is not None and best_accuracy is not None
                else "validation_accuracy=unavailable"
            )
            print(
                f"epoch={epoch + 1}/{config.training.epochs} "
                f"selected={values['selected']} "
                f"relabel_candidates={values['relabel_candidates']} "
                f"label_changes={values['label_changes']} "
                f"{accuracy_summary} "
                f"selection_s={selection_seconds:.1f} train_s={training_seconds:.1f} "
                f"train_samples/s={samples_per_second:.1f} "
                f"data_wait_s={losses.all_data_wait_seconds + losses.selected_data_wait_seconds:.1f}",
                flush=True,
            )
            del supervision, selected_loader

        if label_wave is not None and not stopped_by_label_wave:
            final_predictions = predict_training_labels(
                loaders.evaluation,
                networks,
                device,
                channels_last=config.training.channels_last,
            )
            label_wave.observe(
                final_predictions,
                completed_epochs=last_completed_epoch + 1,
                validation_accuracy=last_validation["accuracy"] if last_validation else None,
                validation_metrics=last_validation,
            )

    evaluate_checkpoints(
        checkpoints, networks, loaders.test, device,
        channels_last=config.training.channels_last, class_names=data.class_names,
        evaluator=evaluate_classification,
    )
    return best_accuracy

"""CIFAR-10 SSR/LSL 실행 orchestration."""

from __future__ import annotations

import random
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from math import cos, pi
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from cifar.label_wave import LabelWaveRun
from cifar.log.checkpoint import CheckpointManager
from cifar.log.common import JsonlWriter, write_config
from cifar.setting.config import ExperimentConfig
from cifar.setting.data import ExperimentData, build_experiment_data
from cifar.setting.model import SSRNetworks, build_ssr_networks, calibrate_batch_norm
from cifar.ssr.evaluation import evaluate_epoch, predict_training_labels, selection_metrics
from cifar.ssr.sampler import ClassBalancedSampler
from cifar.ssr.selection import SelectionResult
from cifar.ssr.trainer import test_accuracy, train_epoch


@dataclass(frozen=True, slots=True)
class EpochLoaders:
    evaluation: DataLoader
    all_samples: DataLoader
    reference: DataLoader
    test: DataLoader
    reference_metric: str


def seed_everything(seed: int) -> None:
    """전역 config seed를 Python, NumPy, PyTorch, CUDA에 동일 적용."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _loader_options(
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if config.training.num_workers > 0:
        options.update(
            prefetch_factor=config.training.prefetch_factor,
            persistent_workers=config.training.persistent_workers,
        )
    return options


def _build_epoch_loaders(
    data: ExperimentData,
    config: ExperimentConfig,
    device: torch.device,
) -> EpochLoaders:
    options = _loader_options(config, device)
    reference_dataset = data.validation if data.validation is not None else data.test
    reference_metric = (
        "validation_accuracy" if data.validation is not None else "test_accuracy"
    )
    return EpochLoaders(
        evaluation=DataLoader(data.evaluation_train, shuffle=False, **options),
        all_samples=DataLoader(
            data.all_train,
            shuffle=True,
            drop_last=True,
            **options,
        ),
        reference=DataLoader(reference_dataset, shuffle=False, **options),
        test=DataLoader(data.test, shuffle=False, **options),
        reference_metric=reference_metric,
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
        return AdamW(
            parameter_groups,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
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


def run(config: ExperimentConfig) -> float:
    """warm-up 없이 매 epoch relabel -> select -> optional LSL -> train."""
    config.validate()
    device = config.resolve_device()
    seed_everything(config.seed)

    data = build_experiment_data(
        config.data,
        device,
        seed=config.seed,
        structural_labels_enabled=config.structural_labels.enabled,
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
    clean_labels = (
        data.clean_labels.to(device) if data.clean_labels is not None else None
    )
    networks = build_ssr_networks(config.model, device, data.num_classes)
    if (
        config.model.calibrate_initial_batch_norm
        and config.model.name in {"cifar_resnet18", "cifar_resnet34"}
    ):
        calibration_batches = calibrate_batch_norm(
            _build_calibration_loader(data, config, device),
            networks.encoder,
            device,
        )
        print(f"initial_batch_norm_calibration_batches={calibration_batches}")
    optimizer = _build_optimizer(networks, config)
    checkpoints = CheckpointManager(config.run_dir, networks, optimizer, config)
    scheduler = _build_scheduler(optimizer, config)
    # Persist only after data, model, optimizer, and scheduler construction succeeds.
    write_config(config, config.run_dir)

    noise_summary = (
        f"actual_noise_rate={float(data.noise_mask.float().mean().item()):.4f}"
        if data.noise_mask is not None
        else "actual_noise_rate=unknown"
    )
    print(f"device={device} run={config.run_name} {noise_summary}")
    # Ensure the first completed epoch always produces a best checkpoint.
    best_accuracy = float("-inf")
    last_accuracy: float | None = None
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
            supervision = evaluate_epoch(
                loaders.evaluation,
                networks,
                noisy_labels,
                config,
                device,
            )
            if label_wave is not None:
                observation = label_wave.observe(
                    supervision.predictions,
                    completed_epochs=epoch,
                    reference_accuracy=last_accuracy,
                    reference_metric=loaders.reference_metric,
                )
                if observation.should_stop and config.label_wave.stop_training:
                    stopped_by_label_wave = True
                    break

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
            current_accuracy = test_accuracy(
                loaders.reference,
                networks,
                device,
                description=(
                    "Validation"
                    if loaders.reference_metric == "validation_accuracy"
                    else "Test"
                ),
            )
            scheduler.step()
            last_accuracy = current_accuracy
            last_completed_epoch = epoch

            is_best = current_accuracy > best_accuracy
            if is_best:
                best_accuracy = current_accuracy
            values = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "encoder_learning_rate": encoder_learning_rate,
                **asdict(losses),
                loaders.reference_metric: current_accuracy,
                f"best_{loaders.reference_metric}": best_accuracy,
                "best_accuracy": best_accuracy,
                **selection_metrics(
                    supervision.selection,
                    noisy_labels,
                    clean_labels,
                    num_classes=config.data.num_classes,
                ),
            }
            metrics_writer.write(values)
            print(
                f"epoch={epoch + 1}/{config.training.epochs} "
                f"selected={values['selected']} "
                f"relabel_candidates={values['relabel_candidates']} "
                f"label_changes={values['label_changes']} "
                f"{loaders.reference_metric}={current_accuracy:.4f} "
                f"best={best_accuracy:.4f}"
            )

            if is_best:
                checkpoints.save(
                    "best.pt",
                    epoch,
                    test_accuracy=(
                        current_accuracy
                        if loaders.reference_metric == "test_accuracy"
                        else None
                    ),
                    metrics={loaders.reference_metric: current_accuracy},
                )

        if label_wave is not None and not stopped_by_label_wave:
            final_predictions = predict_training_labels(
                loaders.evaluation,
                networks,
                device,
            )
            label_wave.observe(
                final_predictions,
                completed_epochs=last_completed_epoch + 1,
                reference_accuracy=last_accuracy,
                reference_metric=loaders.reference_metric,
            )

    final_test_accuracy = (
        last_accuracy
        if loaders.reference_metric == "test_accuracy"
        else test_accuracy(loaders.test, networks, device, description="Test")
    )
    if loaders.reference_metric != "test_accuracy":
        print(f"final_test_accuracy={final_test_accuracy:.4f} (last checkpoint)")
    checkpoints.save(
        "last.pt",
        last_completed_epoch,
        test_accuracy=final_test_accuracy,
        metrics={
            loaders.reference_metric: last_accuracy,
            "test_accuracy": final_test_accuracy,
        },
    )
    return best_accuracy

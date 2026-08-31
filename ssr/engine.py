"""CIFAR-10 SSR/LSL 실행 orchestration."""

from __future__ import annotations

import random
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from label_wave import LabelWaveRun
from log.checkpoint import CheckpointManager
from log.common import JsonlWriter, write_config
from setting.config import ExperimentConfig
from setting.data import CIFAR10Data, build_cifar10_data
from setting.model import SSRNetworks, build_ssr_networks
from ssr.evaluation import evaluate_epoch, predict_training_labels, selection_metrics
from ssr.sampler import ClassBalancedSampler
from ssr.selection import SelectionResult
from ssr.trainer import test_accuracy, train_epoch


@dataclass(frozen=True, slots=True)
class EpochLoaders:
    evaluation: DataLoader
    all_samples: DataLoader
    test: DataLoader


def seed_everything(seed: int) -> None:
    """전역 config seed를 Python, NumPy, PyTorch, CUDA에 동일 적용."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


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
    data: CIFAR10Data,
    config: ExperimentConfig,
    device: torch.device,
) -> EpochLoaders:
    options = _loader_options(config, device)
    return EpochLoaders(
        evaluation=DataLoader(data.evaluation_train, shuffle=False, **options),
        all_samples=DataLoader(
            data.all_train,
            shuffle=True,
            drop_last=True,
            **options,
        ),
        test=DataLoader(data.test, shuffle=False, **options),
    )


def _build_selected_loader(
    data: CIFAR10Data,
    selection: SelectionResult,
    config: ExperimentConfig,
    device: torch.device,
) -> DataLoader:
    subset = Subset(data.selected_train, selection.selected_indices.detach().cpu())
    sampler = ClassBalancedSampler(
        selection.modified_labels[selection.selected_indices].detach().cpu(),
        num_classes=10,
    )
    loader = DataLoader(
        subset,
        sampler=sampler,
        drop_last=True,
        **_loader_options(config, device),
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
) -> SGD:
    return SGD(
        [
            {"params": networks.encoder.parameters()},
            {"params": networks.classifier.parameters()},
            {"params": networks.projector.parameters()},
            {"params": networks.predictor.parameters()},
        ],
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        momentum=config.training.momentum,
    )


def run(config: ExperimentConfig) -> float:
    """warm-up 없이 매 epoch relabel -> select -> optional LSL -> train."""
    config.validate()
    device = config.resolve_device()
    seed_everything(config.seed)
    write_config(config, config.run_dir)

    data = build_cifar10_data(
        config.data,
        device,
        seed=config.seed,
        structural_labels_enabled=config.structural_labels.enabled,
    )
    loaders = _build_epoch_loaders(data, config, device)
    noisy_labels = data.noisy_labels.to(device)
    clean_labels = data.clean_labels.to(device)
    networks = build_ssr_networks(config.model, device)
    optimizer = _build_optimizer(networks, config)
    checkpoints = CheckpointManager(config.run_dir, networks, optimizer, config)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs,
        eta_min=(
            config.training.learning_rate
            * config.training.scheduler_eta_min_ratio
        ),
    )

    actual_noise_rate = float(data.noise_mask.float().mean().item())
    print(
        f"device={device} run={config.run_name} "
        f"actual_noise_rate={actual_noise_rate:.4f}"
    )
    best_accuracy = 0.0
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
                    test_accuracy=last_accuracy,
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
            learning_rate = float(optimizer.param_groups[0]["lr"])
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
            current_accuracy = test_accuracy(loaders.test, networks, device)
            scheduler.step()
            last_accuracy = current_accuracy
            last_completed_epoch = epoch

            is_best = current_accuracy > best_accuracy
            if is_best:
                best_accuracy = current_accuracy
            values = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                **asdict(losses),
                "test_accuracy": current_accuracy,
                "best_accuracy": best_accuracy,
                **selection_metrics(
                    supervision.selection,
                    noisy_labels,
                    clean_labels,
                ),
            }
            metrics_writer.write(values)
            print(
                f"epoch={epoch + 1}/{config.training.epochs} "
                f"selected={values['selected']} relabelled={values['relabelled']} "
                f"acc={current_accuracy:.4f} best={best_accuracy:.4f}"
            )

            if is_best:
                checkpoints.save("best.pt", epoch)

        if label_wave is not None and not stopped_by_label_wave:
            final_predictions = predict_training_labels(
                loaders.evaluation,
                networks,
                device,
            )
            label_wave.observe(
                final_predictions,
                completed_epochs=last_completed_epoch + 1,
                test_accuracy=last_accuracy,
            )

    checkpoints.save(
        "last.pt",
        last_completed_epoch,
        test_accuracy=last_accuracy,
    )
    return best_accuracy

"""CIFAR-10 SSR training loop with the official epoch-level mechanism intact."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from log.common import JsonlWriter, atomic_torch_save, write_config
from setting.config import ExperimentConfig
from setting.data import CIFAR10Data, build_cifar10_data
from setting.model import SSRNetworks, build_ssr_networks
from ssr.losses import mixup_two_views, negative_cosine_similarity, soft_cross_entropy
from ssr.sampler import ClassBalancedSampler
from ssr.selection import SelectionResult, select_samples


@dataclass(slots=True)
class Average:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += value * count
        self.count += count

    @property
    def value(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass(frozen=True, slots=True)
class EpochLoaders:
    evaluation: DataLoader
    all_samples: DataLoader
    test: DataLoader


def seed_everything(seed: int) -> None:
    """Preserve the official SSR random and cuDNN setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def _loader_options(config: ExperimentConfig, device: torch.device) -> dict[str, Any]:
    return {
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "pin_memory": device.type == "cuda",
    }


def build_epoch_loaders(
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


def build_selected_loader(
    data: CIFAR10Data,
    selection: SelectionResult,
    config: ExperimentConfig,
    device: torch.device,
) -> DataLoader:
    subset = Subset(data.selected_train, selection.selected_indices.detach().cpu())
    sampler = ClassBalancedSampler(
        selection.modified_labels[selection.selected_indices], num_classes=10
    )
    loader = DataLoader(
        subset,
        sampler=sampler,
        drop_last=True,
        **_loader_options(config, device),
    )
    if len(loader) == 0:
        raise RuntimeError(
            "The class-balanced selected loader has no full batch. "
            "SSR cannot perform its supervised update for this epoch."
        )
    return loader


@torch.no_grad()
def extract_features_and_predictions(
    loader: DataLoader,
    networks: SSRNetworks,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    networks.eval()
    feature_parts: list[Tensor] = []
    logit_parts: list[Tensor] = []
    for images, _, _, _ in tqdm(loader, desc="Feature extraction", leave=False):
        images = images.to(device, non_blocking=True)
        features = networks.encoder(images)
        feature_parts.append(features)
        logit_parts.append(networks.classifier(features))
    normalized_features = F.normalize(torch.cat(feature_parts, dim=0), dim=1)
    probabilities = torch.softmax(torch.cat(logit_parts, dim=0), dim=1)
    return normalized_features, probabilities


def evaluate_and_select(
    loader: DataLoader,
    networks: SSRNetworks,
    noisy_labels: Tensor,
    config: ExperimentConfig,
    device: torch.device,
) -> SelectionResult:
    normalized_features, probabilities = extract_features_and_predictions(
        loader, networks, device
    )
    return select_samples(
        normalized_features,
        noisy_labels,
        probabilities,
        relabel_threshold=config.ssr.relabel_threshold,
        selection_threshold=config.ssr.selection_threshold,
        neighbors=config.ssr.neighbors,
        chunks=config.ssr.knn_chunks,
    )


def train_epoch(
    selected_loader: DataLoader,
    all_samples_loader: DataLoader,
    modified_labels: Tensor,
    networks: SSRNetworks,
    optimizer: SGD,
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    networks.train()
    supervised_average = Average()
    consistency_average = Average()
    selected_iterator = iter(selected_loader)
    progress = tqdm(all_samples_loader, desc=f"Train {epoch + 1}", leave=False)

    for (weak_strong_views, _, _, _) in progress:
        try:
            strong_views, _, _, indices = next(selected_iterator)
        except StopIteration:
            selected_iterator = iter(selected_loader)
            strong_views, _, _, indices = next(selected_iterator)

        first_strong = strong_views[0].to(device, non_blocking=True)
        second_strong = strong_views[1].to(device, non_blocking=True)
        labels = modified_labels[indices.to(device)]
        mixed_inputs, mixed_targets, _ = mixup_two_views(
            first_strong,
            second_strong,
            labels,
            num_classes=10,
            alpha=config.ssr.mixup_alpha,
        )
        logits = networks.classifier(networks.encoder(mixed_inputs))
        supervised_loss = soft_cross_entropy(logits, mixed_targets)

        weak = weak_strong_views[0].to(device, non_blocking=True)
        strong = weak_strong_views[1].to(device, non_blocking=True)
        weak_features = networks.encoder(weak)
        strong_features = networks.encoder(strong)
        weak_projection = networks.projector(weak_features)
        strong_projection = networks.projector(strong_features)

        # The weak prediction is unused by the loss, but its forward pass is
        # part of upstream SSR and updates predictor BatchNorm statistics.
        _weak_prediction = networks.predictor(weak_projection)
        strong_prediction = networks.predictor(strong_projection)
        consistency_loss = negative_cosine_similarity(
            strong_prediction, weak_projection
        )
        loss = (
            supervised_loss
            + config.ssr.feature_consistency_weight * consistency_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        supervised_average.update(supervised_loss.item())
        consistency_average.update(consistency_loss.item())
        progress.set_postfix(
            lr=f"{optimizer.param_groups[0]['lr']:.6f}",
            ce=f"{supervised_average.value:.4f}",
            fc=f"{consistency_average.value:.4f}",
        )
    return supervised_average.value, consistency_average.value


@torch.no_grad()
def test_accuracy(
    loader: DataLoader,
    networks: SSRNetworks,
    device: torch.device,
) -> float:
    networks.eval()
    accuracy = Average()
    for images, labels, _ in tqdm(loader, desc="Test", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = networks.classifier(networks.encoder(images)).argmax(dim=1)
        batch_accuracy = predictions.eq(labels).sum().item() / images.size(0)
        accuracy.update(batch_accuracy, images.size(0))
    return accuracy.value


def selection_metrics(
    selection: SelectionResult,
    noisy_labels: Tensor,
    clean_labels: Tensor,
) -> dict[str, int | float]:
    selected = selection.selected_indices
    rejected = selection.rejected_indices
    modified = selection.modified_labels
    relabelled = selection.relabelled_indices
    return {
        "selected": int(selected.numel()),
        "rejected": int(rejected.numel()),
        "tp": int(modified[selected].eq(clean_labels[selected]).sum().item()),
        "fp": int(modified[selected].ne(clean_labels[selected]).sum().item()),
        "tn": int(modified[rejected].ne(clean_labels[rejected]).sum().item()),
        "fn": int(modified[rejected].eq(clean_labels[rejected]).sum().item()),
        "relabelled": int(relabelled.numel()),
        "relabel_correct": int(modified[relabelled].eq(clean_labels[relabelled]).sum().item()),
        "relabel_original_correct": int(
            noisy_labels[relabelled].eq(clean_labels[relabelled]).sum().item()
        ),
        "confidence_mean": float(selection.confidences.mean().item()),
        "confidence_min": float(selection.confidences.min().item()),
        "confidence_max": float(selection.confidences.max().item()),
    }


def checkpoint_state(
    epoch: int,
    networks: SSRNetworks,
    optimizer: SGD,
) -> dict[str, Any]:
    return {
        "cur_epoch": epoch,
        "classifier": networks.classifier.state_dict(),
        "encoder": networks.encoder.state_dict(),
        "proj_head": networks.projector.state_dict(),
        "pred_head": networks.predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def run(config: ExperimentConfig) -> float:
    """Run SSR: relabel -> structural select -> train, from epoch zero."""
    config.validate()
    device = config.resolve_device()
    seed_everything(config.training.seed)
    write_config(config, config.run_dir)

    data = build_cifar10_data(config.data, device)
    loaders = build_epoch_loaders(data, config, device)
    noisy_labels = data.noisy_labels.to(device)
    clean_labels = data.clean_labels.to(device)
    networks = build_ssr_networks(device)
    optimizer = SGD(
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
    with JsonlWriter(config.run_dir / "metrics.jsonl") as metrics_writer:
        for epoch in range(config.training.epochs):
            selection = evaluate_and_select(
                loaders.evaluation,
                networks,
                noisy_labels,
                config,
                device,
            )
            selected_loader = build_selected_loader(data, selection, config, device)
            supervised_loss, consistency_loss = train_epoch(
                selected_loader,
                loaders.all_samples,
                selection.modified_labels,
                networks,
                optimizer,
                config,
                device,
                epoch,
            )
            current_accuracy = test_accuracy(loaders.test, networks, device)
            scheduler.step()
            is_best = current_accuracy > best_accuracy
            if is_best:
                best_accuracy = current_accuracy
            values = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "supervised_loss": supervised_loss,
                "feature_consistency_loss": consistency_loss,
                "test_accuracy": current_accuracy,
                "best_accuracy": best_accuracy,
                **selection_metrics(selection, noisy_labels, clean_labels),
            }
            metrics_writer.write(values)
            print(
                f"epoch={epoch + 1}/{config.training.epochs} "
                f"selected={values['selected']} relabelled={values['relabelled']} "
                f"acc={current_accuracy:.4f} best={best_accuracy:.4f}"
            )

            if is_best:
                atomic_torch_save(
                    checkpoint_state(epoch, networks, optimizer),
                    config.run_dir / "best.pt",
                )

    atomic_torch_save(
        checkpoint_state(config.training.epochs, networks, optimizer),
        config.run_dir / "last.pt",
    )
    return best_accuracy

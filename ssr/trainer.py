"""SSR baseline loss와 optional structural loss를 적용하는 epoch trainer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from lsl import structural_mixup_loss
from setting.config import ExperimentConfig
from setting.model import SSRNetworks
from ssr.losses import (
    mixup_hard_label_views,
    negative_cosine_similarity,
    soft_cross_entropy,
)


@dataclass(slots=True)
class _RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def value(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass(frozen=True, slots=True)
class TrainingLosses:
    supervised_loss: float
    feature_consistency_loss: float
    structural_loss: float | None


def train_epoch(
    selected_loader: DataLoader,
    all_samples_loader: DataLoader,
    modified_labels: Tensor,
    structural_targets: Tensor | None,
    networks: SSRNetworks,
    optimizer: SGD,
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
) -> TrainingLosses:
    networks.train()
    supervised_average = _RunningAverage()
    consistency_average = _RunningAverage()
    structural_average = _RunningAverage() if structural_targets is not None else None
    selected_iterator = iter(selected_loader)
    progress = tqdm(all_samples_loader, desc=f"Train {epoch + 1}", leave=False)

    for all_views, all_indices in progress:
        try:
            selected_views, selected_indices = next(selected_iterator)
        except StopIteration:
            selected_iterator = iter(selected_loader)
            selected_views, selected_indices = next(selected_iterator)

        first_selected = selected_views[0].to(device, non_blocking=True)
        second_selected = selected_views[1].to(device, non_blocking=True)
        selected_labels = modified_labels[selected_indices.to(device)]
        mixed_inputs, mixed_targets, _ = mixup_hard_label_views(
            first_selected,
            second_selected,
            selected_labels,
            num_classes=10,
            alpha=config.ssr.mixup_alpha,
        )
        supervised_logits = networks.classifier(networks.encoder(mixed_inputs))
        supervised_loss = soft_cross_entropy(supervised_logits, mixed_targets)

        weak_view = all_views[0].to(device, non_blocking=True)
        first_strong_view = all_views[1].to(device, non_blocking=True)
        weak_projection = networks.projector(networks.encoder(weak_view))
        strong_projection = networks.projector(networks.encoder(first_strong_view))

        # p_weak는 loss에 직접 쓰이지 않지만
        # 공식 SSR의 predictor BN 갱신에 필요하다.
        _weak_prediction = networks.predictor(weak_projection)
        strong_prediction = networks.predictor(strong_projection)
        consistency_loss = negative_cosine_similarity(
            strong_prediction,
            weak_projection,
        )
        total_loss = (
            supervised_loss
            + config.ssr.feature_consistency_weight * consistency_loss
        )

        current_structural_loss = None
        if structural_targets is not None:
            if len(all_views) != 3:
                raise RuntimeError("LSL requires [weak, strong, strong] all-sample views.")
            second_strong_view = all_views[2].to(device, non_blocking=True)
            batch_structural_targets = structural_targets[all_indices.to(device)]
            current_structural_loss = structural_mixup_loss(
                networks.encoder,
                networks.classifier,
                first_strong_view,
                second_strong_view,
                batch_structural_targets,
                mixup_alpha=config.ssr.mixup_alpha,
            )
            total_loss = (
                total_loss
                + config.structural_labels.loss_weight * current_structural_loss
            )
        elif len(all_views) != 2:
            raise RuntimeError("SSR requires [weak, strong] all-sample views.")

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        supervised_average.update(supervised_loss.item())
        consistency_average.update(consistency_loss.item())
        postfix = {
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            "ce": f"{supervised_average.value:.4f}",
            "fc": f"{consistency_average.value:.4f}",
        }
        if structural_average is not None and current_structural_loss is not None:
            structural_average.update(current_structural_loss.item())
            postfix["st"] = f"{structural_average.value:.4f}"
        progress.set_postfix(**postfix)

    return TrainingLosses(
        supervised_loss=supervised_average.value,
        feature_consistency_loss=consistency_average.value,
        structural_loss=(
            structural_average.value if structural_average is not None else None
        ),
    )


@torch.no_grad()
def test_accuracy(
    loader: DataLoader,
    networks: SSRNetworks,
    device: torch.device,
) -> float:
    networks.eval()
    correct = torch.zeros((), dtype=torch.long, device=device)
    samples = 0
    for images, labels in tqdm(loader, desc="Test", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = networks.classifier(networks.encoder(images)).argmax(dim=1)
        correct += predictions.eq(labels).sum()
        samples += images.size(0)
    return float((correct / samples).item())

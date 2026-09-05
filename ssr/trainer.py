"""SSR baseline loss와 optional structural loss를 적용하는 epoch trainer."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from lsl import structural_mixup_loss
from setting.config import ExperimentConfig
from setting.model import SSRNetworks
from setting.precision import PrecisionPolicy, full_precision
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


def _loss_value(loss: Tensor, name: str) -> float:
    value = loss.item()
    if not isfinite(value):
        raise FloatingPointError(f"Non-finite {name} loss; optimizer step was not applied.")
    return value


def train_epoch(
    selected_loader: DataLoader,
    all_samples_loader: DataLoader,
    modified_labels: Tensor,
    structural_targets: Tensor | None,
    networks: SSRNetworks,
    optimizer: Optimizer,
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
) -> TrainingLosses:
    networks.train()
    training = getattr(config, "training", None)
    precision = PrecisionPolicy.from_config(training, device)
    log_interval = getattr(training, "log_interval", 20)
    supervised_average = _RunningAverage()
    consistency_average = _RunningAverage()
    structural_average = _RunningAverage() if structural_targets is not None else None
    selected_iterator = iter(selected_loader)
    progress = tqdm(all_samples_loader, desc=f"Train {epoch + 1}", leave=False)

    for step, (all_views, all_indices) in enumerate(progress, start=1):
        try:
            selected_views, selected_indices = next(selected_iterator)
        except StopIteration:
            selected_iterator = iter(selected_loader)
            selected_views, selected_indices = next(selected_iterator)

        # Release the previous step's gradients before allocating activations.
        optimizer.zero_grad(set_to_none=True)
        first_selected = precision.to_device(selected_views[0])
        second_selected = precision.to_device(selected_views[1])
        selected_labels = modified_labels[selected_indices.to(device)]
        mixed_inputs, mixed_targets, _ = mixup_hard_label_views(
            first_selected,
            second_selected,
            selected_labels,
            num_classes=config.data.num_classes,
            alpha=config.ssr.mixup_alpha,
        )
        with precision.autocast():
            supervised_logits = networks.classifier(networks.encoder(mixed_inputs))
        supervised_loss = soft_cross_entropy(supervised_logits, mixed_targets)
        supervised_value = _loss_value(supervised_loss, "supervised")
        # Additive loss gradients accumulate at the same parameter values.
        # Backpropagate each branch before allocating the next encoder graph.
        supervised_loss.backward()
        supervised_average.update(supervised_value)
        del supervised_logits, supervised_loss, mixed_inputs, mixed_targets
        del first_selected, second_selected, selected_labels

        weak_view = precision.to_device(all_views[0])
        first_strong_view = precision.to_device(all_views[1])
        # SSR stops gradients through its weak target. Train-mode BN updates
        # and stochastic forward calls are still performed in the same order.
        # Separate AMP scopes prevent cached no-grad weights leaking to strong.
        with torch.no_grad(), precision.autocast():
            weak_projection = networks.projector(networks.encoder(weak_view))
        with precision.autocast():
            strong_projection = networks.projector(networks.encoder(first_strong_view))

        # p_weak는 loss에 직접 쓰이지 않지만
        # 공식 SSR의 predictor BN 갱신에 필요하다.
        with torch.no_grad(), precision.autocast():
            networks.predictor(weak_projection)
        with precision.autocast():
            strong_prediction = networks.predictor(strong_projection)
        consistency_loss = negative_cosine_similarity(
            strong_prediction,
            weak_projection,
        )
        consistency_value = _loss_value(consistency_loss, "feature-consistency")
        (config.ssr.feature_consistency_weight * consistency_loss).backward()
        consistency_average.update(consistency_value)
        del consistency_loss, strong_prediction, strong_projection, weak_projection
        del weak_view

        if structural_targets is not None:
            if len(all_views) != 3:
                raise RuntimeError("LSL requires [weak, strong, strong] all-sample views.")
            second_strong_view = precision.to_device(all_views[2])
            batch_structural_targets = structural_targets[all_indices.to(device)]
            with precision.autocast():
                current_structural_loss = structural_mixup_loss(
                    networks.encoder,
                    networks.classifier,
                    first_strong_view,
                    second_strong_view,
                    batch_structural_targets,
                    mixup_alpha=config.ssr.mixup_alpha,
                )
            structural_value = _loss_value(current_structural_loss, "structural")
            (config.structural_labels.loss_weight * current_structural_loss).backward()
            structural_average.update(structural_value)
            del current_structural_loss, batch_structural_targets, second_strong_view
        elif len(all_views) != 2:
            raise RuntimeError("SSR requires [weak, strong] all-sample views.")

        optimizer.step()
        del first_strong_view

        if step % log_interval != 0 and step != len(all_samples_loader):
            continue
        encoder_learning_rate = optimizer.param_groups[0]["lr"]
        head_learning_rate = (
            optimizer.param_groups[1]["lr"]
            if len(optimizer.param_groups) > 1
            else encoder_learning_rate
        )
        postfix = {
            "lr": f"{head_learning_rate:.6f}",
            "ce": f"{supervised_average.value:.4f}",
            "fc": f"{consistency_average.value:.4f}",
        }
        if encoder_learning_rate != head_learning_rate:
            postfix["enc_lr"] = f"{encoder_learning_rate:.6f}"
        if structural_average is not None:
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
    *,
    description: str = "Evaluation",
    channels_last: bool = False,
) -> float:
    networks.eval()
    device = torch.device(device)
    precision = PrecisionPolicy(
        device=device, channels_last=channels_last and device.type == "cuda"
    )
    correct = torch.zeros((), dtype=torch.long, device=device)
    samples = 0
    for images, labels in tqdm(loader, desc=description, leave=False):
        images = precision.to_device(images)
        if images.dtype in {torch.float16, torch.bfloat16}:
            images = images.float()
        labels = labels.to(device, non_blocking=True)
        with full_precision(device):
            logits = networks.classifier(networks.encoder(images))
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError(f"{description} model produced non-finite logits.")
        predictions = logits.argmax(dim=1)
        correct += predictions.eq(labels).sum()
        samples += images.size(0)
    if samples == 0:
        raise RuntimeError(f"{description} loader is empty.")
    return float((correct / samples).item())

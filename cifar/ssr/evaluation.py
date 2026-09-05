"""전체 train feature 추출, SSR relabel/selection, LSL target 생성."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from cifar.setting.config import ExperimentConfig
from cifar.setting.model import SSRNetworks
from cifar.ssr.selection import SelectionResult, select_samples
from cifar.lsl import extract_structural_labels


@dataclass(frozen=True, slots=True)
class EpochSupervision:
    """한 epoch에서 고정해 사용하는 SSR 선택 결과와 optional LSL target."""

    selection: SelectionResult
    structural_targets: Tensor | None
    predictions: Tensor


@torch.no_grad()
def _extract_features_and_predictions(
    loader: DataLoader,
    networks: SSRNetworks,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Extract directly into dataset-order buffers to avoid a second full copy."""

    networks.eval()
    sample_count = len(loader.dataset)
    features_all: Tensor | None = None
    probabilities_all: Tensor | None = None
    seen = torch.zeros(sample_count, dtype=torch.bool)

    for images, indices in tqdm(loader, desc="Feature extraction", leave=False):
        images = images.to(device, non_blocking=True)
        features = networks.encoder(images)
        logits = networks.classifier(features)
        probabilities = torch.softmax(logits, dim=1)
        cpu_indices = torch.as_tensor(indices, dtype=torch.long)
        if (
            cpu_indices.numel() != images.size(0)
            or int(cpu_indices.min()) < 0
            or int(cpu_indices.max()) >= sample_count
            or seen[cpu_indices].any()
        ):
            raise RuntimeError("Evaluation loader returned invalid or duplicate indices.")
        seen[cpu_indices] = True
        device_indices = cpu_indices.to(device, non_blocking=True)
        if features_all is None:
            features_all = features.new_empty((sample_count, features.size(1)))
            probabilities_all = probabilities.new_empty(
                (sample_count, probabilities.size(1))
            )
        features_all.index_copy_(0, device_indices, features)
        probabilities_all.index_copy_(0, device_indices, probabilities)

    if features_all is None or probabilities_all is None:
        raise RuntimeError("Evaluation loader must contain at least one batch.")
    if not seen.all():
        raise RuntimeError("Evaluation loader did not cover every dataset sample.")
    return features_all, probabilities_all


@torch.no_grad()
def evaluate_epoch(
    loader: DataLoader,
    networks: SSRNetworks,
    noisy_labels: Tensor,
    config: ExperimentConfig,
    device: torch.device,
) -> EpochSupervision:
    """논문 Algorithm 2의 relabel -> select -> structural-label 순서."""
    raw_features, probabilities = _extract_features_and_predictions(
        loader,
        networks,
        device,
    )
    selection = select_samples(
        raw_features,
        noisy_labels,
        probabilities,
        relabel_threshold=config.ssr.relabel_threshold,
        selection_threshold=config.ssr.selection_threshold,
        neighbors=config.ssr.neighbors,
        chunks=config.ssr.knn_chunks,
        num_classes=config.data.num_classes,
    )
    structural_targets = None
    if config.structural_labels.enabled:
        structural_targets = extract_structural_labels(
            raw_features,
            selection.modified_labels,
            neighbors=config.structural_labels.neighbors,
            chunks=config.structural_labels.knn_chunks,
            num_classes=config.data.num_classes,
        )
    predictions = probabilities.argmax(dim=1)
    return EpochSupervision(selection, structural_targets, predictions)


@torch.no_grad()
def predict_training_labels(
    loader: DataLoader,
    networks: SSRNetworks,
    device: torch.device,
) -> Tensor:
    """Return raw model predictions in the evaluation dataset's fixed order."""
    networks.eval()
    sample_count = len(loader.dataset)
    predictions = torch.empty(sample_count, dtype=torch.long)
    seen = torch.zeros(sample_count, dtype=torch.bool)
    for images, indices in tqdm(loader, desc="Prediction extraction", leave=False):
        images = images.to(device, non_blocking=True)
        features = networks.encoder(images)
        batch_predictions = networks.classifier(features).argmax(dim=1).cpu()
        cpu_indices = torch.as_tensor(indices, dtype=torch.long)
        if (
            cpu_indices.numel() != images.size(0)
            or int(cpu_indices.min()) < 0
            or int(cpu_indices.max()) >= sample_count
            or seen[cpu_indices].any()
        ):
            raise RuntimeError("Prediction loader returned invalid or duplicate indices.")
        seen[cpu_indices] = True
        predictions.index_copy_(0, cpu_indices, batch_predictions)
    if not seen.any():
        raise RuntimeError("Prediction loader must contain at least one batch.")
    if not seen.all():
        raise RuntimeError("Prediction loader did not cover every dataset sample.")
    return predictions


def selection_metrics(
    selection: SelectionResult,
    noisy_labels: Tensor,
    clean_labels: Tensor | None,
    *,
    num_classes: int = 10,
) -> dict[str, Any]:
    """Return label-free counts and optional clean-label diagnostics."""
    selected = selection.selected_indices
    rejected = selection.rejected_indices
    modified = selection.modified_labels
    relabelled = selection.relabelled_indices
    changed = selection.changed_indices
    transitions: list[dict[str, int]] = []
    if changed.numel():
        source = noisy_labels[changed]
        target = modified[changed]
        encoded = source * num_classes + target
        values, counts = torch.unique(encoded, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            transitions.append(
                {
                    "from": int(value // num_classes),
                    "to": int(value % num_classes),
                    "count": int(count),
                }
            )

    metrics: dict[str, Any] = {
        "selected": int(selected.numel()),
        "rejected": int(rejected.numel()),
        "relabel_candidates": int(relabelled.numel()),
        # Kept for compatibility with existing result sheets/log parsers.
        "relabelled": int(relabelled.numel()),
        "label_changes": int(changed.numel()),
        "label_change_rate": float(changed.numel() / noisy_labels.numel()),
        "label_change_transitions": transitions,
        "confidence_mean": float(selection.confidences.mean().item()),
        "confidence_min": float(selection.confidences.min().item()),
        "confidence_max": float(selection.confidences.max().item()),
    }
    if clean_labels is not None:
        metrics.update(
            tp=int(modified[selected].eq(clean_labels[selected]).sum().item()),
            fp=int(modified[selected].ne(clean_labels[selected]).sum().item()),
            tn=int(modified[rejected].ne(clean_labels[rejected]).sum().item()),
            fn=int(modified[rejected].eq(clean_labels[rejected]).sum().item()),
            relabel_correct=int(
                modified[relabelled].eq(clean_labels[relabelled]).sum().item()
            ),
            relabel_original_correct=int(
                noisy_labels[relabelled].eq(clean_labels[relabelled]).sum().item()
            ),
            label_change_correct=int(
                modified[changed].eq(clean_labels[changed]).sum().item()
            ),
            label_change_original_correct=int(
                noisy_labels[changed].eq(clean_labels[changed]).sum().item()
            ),
        )
    return metrics

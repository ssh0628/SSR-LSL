"""전체 train feature 추출, SSR relabel/selection, LSL target 생성."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from setting.config import ExperimentConfig
from setting.model import SSRNetworks
from ssr.selection import SelectionResult, select_samples
from lsl import extract_structural_labels


@dataclass(frozen=True, slots=True)
class EpochSupervision:
    """한 epoch에서 고정해 사용하는 SSR 선택 결과와 optional LSL target."""

    selection: SelectionResult
    structural_targets: Tensor | None


@torch.no_grad()
def _extract_features_and_predictions(
    loader: DataLoader,
    networks: SSRNetworks,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    networks.eval()
    feature_parts: list[Tensor] = []
    logit_parts: list[Tensor] = []
    for images, _ in tqdm(loader, desc="Feature extraction", leave=False):
        images = images.to(device, non_blocking=True)
        features = networks.encoder(images)
        feature_parts.append(features)
        logit_parts.append(networks.classifier(features))

    features = torch.cat(feature_parts, dim=0)
    probabilities = torch.softmax(torch.cat(logit_parts, dim=0), dim=1)
    return features, probabilities


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
    normalized_features = F.normalize(raw_features, dim=1)
    selection = select_samples(
        normalized_features,
        noisy_labels,
        probabilities,
        relabel_threshold=config.ssr.relabel_threshold,
        selection_threshold=config.ssr.selection_threshold,
        neighbors=config.ssr.neighbors,
        chunks=config.ssr.knn_chunks,
    )
    structural_targets = None
    if config.structural_labels.enabled:
        structural_targets = extract_structural_labels(
            raw_features,
            selection.modified_labels,
            neighbors=config.structural_labels.neighbors,
            chunks=config.structural_labels.knn_chunks,
            num_classes=10,
        )
    return EpochSupervision(selection, structural_targets)


def selection_metrics(
    selection: SelectionResult,
    noisy_labels: Tensor,
    clean_labels: Tensor,
) -> dict[str, int | float]:
    """clean label을 학습이 아닌 관찰에만 사용하는 SSR diagnostics."""
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
        "relabel_correct": int(
            modified[relabelled].eq(clean_labels[relabelled]).sum().item()
        ),
        "relabel_original_correct": int(
            noisy_labels[relabelled].eq(clean_labels[relabelled]).sum().item()
        ),
        "confidence_mean": float(selection.confidences.mean().item()),
        "confidence_min": float(selection.confidences.min().item()),
        "confidence_max": float(selection.confidences.max().item()),
    }

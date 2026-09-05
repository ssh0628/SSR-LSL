"""전체 train feature 추출, SSR relabel/selection, LSL target 생성."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from lsl import extract_structural_labels
from setting.config import ExperimentConfig
from setting.model import SSRNetworks
from setting.precision import full_precision
from ssr.selection import SelectionResult, select_samples


@dataclass(frozen=True, slots=True)
class EpochSupervision:
    """한 epoch에서 고정해 사용하는 SSR 선택 결과와 optional LSL target."""

    selection: SelectionResult
    structural_targets: Tensor | None
    predictions: Tensor


def _record_indices(indices: Tensor, batch_size: int, seen: Tensor) -> Tensor:
    """Require each dataset row exactly once, including within this batch."""
    cpu_indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
    if (
        cpu_indices.ndim != 1
        or cpu_indices.numel() == 0
        or cpu_indices.numel() != batch_size
        or int(cpu_indices.min()) < 0
        or int(cpu_indices.max()) >= seen.numel()
        or cpu_indices.unique().numel() != cpu_indices.numel()
        or seen[cpu_indices].any()
    ):
        raise RuntimeError("Evaluation loader returned invalid or duplicate indices.")
    seen[cpu_indices] = True
    return cpu_indices


def _validate_model_outputs(features: Tensor, logits: Tensor) -> None:
    if not (torch.isfinite(features).all() & torch.isfinite(logits).all()).item():
        raise FloatingPointError("Model produced non-finite features or logits during evaluation.")


def _evaluation_images(images: Tensor, device: torch.device, channels_last: bool) -> Tensor:
    # AMP is limited to training; preserve double precision in diagnostic tests.
    dtype = torch.float32 if images.dtype in (torch.float16, torch.bfloat16) else images.dtype
    if channels_last and device.type == "cuda" and images.ndim == 4:
        return images.to(
            device, dtype=dtype, non_blocking=True, memory_format=torch.channels_last
        )
    return images.to(device, dtype=dtype, non_blocking=True)


@torch.no_grad()
def _extract_features_and_predictions(
    loader: DataLoader,
    networks: SSRNetworks,
    device: torch.device,
    *,
    channels_last: bool = False,
) -> tuple[Tensor, Tensor]:
    """Extract directly into dataset-order buffers to avoid a second full copy."""

    networks.eval()
    sample_count = len(loader.dataset)
    features_all: Tensor | None = None
    probabilities_all: Tensor | None = None
    seen = torch.zeros(sample_count, dtype=torch.bool)

    with full_precision(device):
        for images, indices in tqdm(loader, desc="Feature extraction", leave=False):
            cpu_indices = _record_indices(indices, images.size(0), seen)
            images = _evaluation_images(images, device, channels_last)
            features = networks.encoder(images)
            logits = networks.classifier(features)
            _validate_model_outputs(features, logits)
            probabilities = torch.softmax(logits, dim=1)
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
    # k-NN rankings and confidence thresholds must not inherit a caller's AMP.
    with full_precision(device):
        raw_features, probabilities = _extract_features_and_predictions(
            loader,
            networks,
            device,
            channels_last=getattr(getattr(config, "training", None), "channels_last", False),
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
    *,
    channels_last: bool = False,
) -> Tensor:
    """Return raw model predictions in the evaluation dataset's fixed order."""
    networks.eval()
    sample_count = len(loader.dataset)
    predictions = torch.empty(sample_count, dtype=torch.long)
    seen = torch.zeros(sample_count, dtype=torch.bool)
    with full_precision(device):
        for images, indices in tqdm(loader, desc="Prediction extraction", leave=False):
            cpu_indices = _record_indices(indices, images.size(0), seen)
            images = _evaluation_images(images, device, channels_last)
            features = networks.encoder(images)
            logits = networks.classifier(features)
            _validate_model_outputs(features, logits)
            batch_predictions = logits.argmax(dim=1).cpu()
            predictions.index_copy_(0, cpu_indices, batch_predictions)
    if not seen.any():
        raise RuntimeError("Prediction loader must contain at least one batch.")
    if not seen.all():
        raise RuntimeError("Prediction loader did not cover every dataset sample.")
    return predictions


def selection_metrics(
    selection: SelectionResult,
    noisy_labels: Tensor,
    *,
    num_classes: int,
) -> dict[str, Any]:
    """Return observed-label correction counts without assuming clean GT."""
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

    return {
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

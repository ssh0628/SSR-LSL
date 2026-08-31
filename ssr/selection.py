"""Per-epoch SSR relabelling followed by structural sample selection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ssr.knn import balanced_knn_scores


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_indices: Tensor
    rejected_indices: Tensor
    modified_labels: Tensor
    relabelled_indices: Tensor
    confidences: Tensor
    consistency: Tensor


@torch.no_grad()
def relabel_from_predictions(
    noisy_labels: Tensor,
    prediction_probabilities: Tensor,
    threshold: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reset to noisy labels, then replace predictions strictly above theta_r."""
    confidences, predicted_labels = prediction_probabilities.max(dim=1)
    relabelled_indices = torch.where(confidences > threshold)[0]
    modified_labels = noisy_labels.clone().detach()
    modified_labels[relabelled_indices] = predicted_labels[relabelled_indices]
    return modified_labels, relabelled_indices, confidences


@torch.no_grad()
def select_samples(
    normalized_features: Tensor,
    noisy_labels: Tensor,
    prediction_probabilities: Tensor,
    *,
    relabel_threshold: float,
    selection_threshold: float,
    neighbors: int,
    chunks: int,
) -> SelectionResult:
    modified_labels, relabelled_indices, confidences = relabel_from_predictions(
        noisy_labels, prediction_probabilities, relabel_threshold
    )
    knn_scores = balanced_knn_scores(
        normalized_features,
        normalized_features,
        modified_labels,
        num_classes=10,
        neighbors=neighbors,
        chunks=chunks,
    )
    label_scores = torch.gather(knn_scores, 1, modified_labels.view(-1, 1)).squeeze(1)
    maximum_scores = knn_scores.max(dim=1).values
    consistency = label_scores / maximum_scores
    selected_indices = torch.where(consistency >= selection_threshold)[0]
    rejected_indices = torch.where(consistency < selection_threshold)[0]
    return SelectionResult(
        selected_indices=selected_indices,
        rejected_indices=rejected_indices,
        modified_labels=modified_labels,
        relabelled_indices=relabelled_indices,
        confidences=confidences,
        consistency=consistency,
    )

"""The class-balanced hard-voting k-NN used by SSR sample selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def hard_knn_scores(
    queries: Tensor,
    feature_bank_transposed: Tensor,
    feature_labels: Tensor,
    num_classes: int,
    neighbors: int,
) -> Tensor:
    """Uniformly vote over the cosine-nearest labels, including self matches."""
    similarity = torch.mm(queries, feature_bank_transposed)
    neighbor_indices = similarity.topk(k=neighbors, dim=-1).indices
    neighbor_labels = feature_labels[neighbor_indices]
    scores = torch.zeros(
        queries.size(0),
        num_classes,
        device=neighbor_labels.device,
        dtype=queries.dtype,
    )
    scores.scatter_add_(
        dim=1,
        index=neighbor_labels,
        src=torch.ones_like(neighbor_labels, dtype=scores.dtype),
    )
    return scores / neighbors


@torch.no_grad()
def balanced_knn_scores(
    current_features: Tensor,
    feature_bank: Tensor,
    labels: Tensor,
    *,
    num_classes: int = 10,
    neighbors: int = 200,
    chunks: int = 10,
) -> Tensor:
    """Reproduce official SSR global-prior-corrected k-NN scores."""
    if current_features.ndim != 2 or feature_bank.ndim != 2:
        raise ValueError("current_features and feature_bank must be rank-2 tensors.")
    if current_features.size(1) != feature_bank.size(1):
        raise ValueError("query and feature-bank dimensions must match.")
    if labels.numel() != feature_bank.size(0):
        raise ValueError("labels must contain one entry per feature-bank row.")
    if labels.ndim != 1:
        raise ValueError("labels must be a rank-1 tensor.")
    if not 1 <= neighbors <= feature_bank.size(0):
        raise ValueError("neighbors must be in [1, feature-bank size].")
    if chunks < 1:
        raise ValueError("chunks must be positive.")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError("labels contain an invalid class index.")

    counts = torch.bincount(labels, minlength=num_classes).to(feature_bank.dtype) + 1e-10
    class_prior = counts / counts.sum()
    normalized_bank = F.normalize(feature_bank, dim=1)
    normalized_queries = (
        normalized_bank
        if current_features is feature_bank
        else F.normalize(current_features, dim=1)
    )
    feature_bank_transposed = normalized_bank.T
    chunk_size = max(1, (len(current_features) + chunks - 1) // chunks)
    score_parts: list[Tensor] = []

    for start in range(0, len(normalized_queries), chunk_size):
        end = min(start + chunk_size, len(normalized_queries))
        score_parts.append(
            hard_knn_scores(
                normalized_queries[start:end],
                feature_bank_transposed,
                labels,
                num_classes,
                neighbors,
            )
        )

    if not score_parts:
        return torch.empty((0, num_classes), device=feature_bank.device)
    scores = torch.cat(score_parts, dim=0)
    scores = scores / class_prior
    return scores / scores.sum(dim=1, keepdim=True)

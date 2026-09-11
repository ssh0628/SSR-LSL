"""The class-balanced hard-voting k-NN used by SSR sample selection."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# Exact k-NN still compares every query with the full bank. Only the number
# of query rows processed together is reduced for large datasets.
MAX_SIMILARITY_BYTES = 256 * 1024 * 1024


def similarity_chunk_size(
    query_count: int,
    bank_count: int,
    element_size: int,
    chunks: int,
) -> int:
    """Respect the requested chunk count and cap each similarity allocation."""
    requested_rows = max(1, (query_count + chunks - 1) // chunks)
    memory_rows = max(1, MAX_SIMILARITY_BYTES // (bank_count * element_size))
    return min(requested_rows, memory_rows)


@torch.no_grad()
def hard_knn_scores(
    queries: Tensor,
    feature_bank_transposed: Tensor,
    feature_labels: Tensor,
    num_classes: int,
    neighbors: int,
    *,
    chunks: int = 1,
) -> Tensor:
    """Uniformly vote over the cosine-nearest labels, including self matches."""
    if chunks < 1:
        raise ValueError("chunks must be positive.")
    scores = torch.zeros(
        queries.size(0),
        num_classes,
        device=feature_labels.device,
        dtype=queries.dtype,
    )
    chunk_size = similarity_chunk_size(
        len(queries), feature_bank_transposed.size(1), queries.element_size(), chunks
    )
    for start in range(0, len(queries), chunk_size):
        end = min(start + chunk_size, len(queries))
        similarity = torch.mm(queries[start:end], feature_bank_transposed)
        neighbor_indices = similarity.topk(k=neighbors, dim=-1).indices
        del similarity
        neighbor_labels = feature_labels[neighbor_indices]
        scores[start:end].scatter_add_(
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
    if len(normalized_queries) == 0:
        return torch.empty((0, num_classes), device=feature_bank.device)
    scores = hard_knn_scores(
        normalized_queries,
        normalized_bank.T,
        labels,
        num_classes,
        neighbors,
        chunks=chunks,
    )
    scores = scores / class_prior
    return scores / scores.sum(dim=1, keepdim=True)

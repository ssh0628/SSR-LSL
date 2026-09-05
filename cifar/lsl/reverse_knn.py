"""LSL Algorithm 1: reverse k-NN structural-label extraction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _chunk_bounds(size: int, chunks: int) -> list[tuple[int, int]]:
    """빈 chunk 없이 전체 [0, size)를 순서대로 분할."""
    chunk_size = (size + chunks - 1) // chunks
    return [
        (start, min(start + chunk_size, size))
        for start in range(0, size, chunk_size)
    ]


@torch.no_grad()
def extract_structural_labels(
    features: Tensor,
    relabelled_labels: Tensor,
    *,
    neighbors: int = 20,
    chunks: int = 10,
    num_classes: int = 10,
) -> Tensor:
    """각 source label을 그 source의 k개 이웃에게 전파해 soft target 생성.

    일반 k-NN은 각 query가 자기 이웃들의 label을 가져온다. 이 함수는 반대로
    source i의 relabelled label을 ``topk(sim(f_i, F))``가 가리키는 sample들에
    누적한다. 마지막 행 정규화가 논문 Algorithm 1의 ``T / T.sum(1)``이다.
    """
    if features.ndim != 2:
        raise ValueError("features must have shape (N, D).")
    if relabelled_labels.ndim != 1 or relabelled_labels.numel() != features.size(0):
        raise ValueError("relabelled_labels must have shape (N,).")
    if not 1 <= neighbors <= features.size(0):
        raise ValueError("neighbors must be in [1, N].")
    if chunks < 1:
        raise ValueError("chunks must be positive.")
    if relabelled_labels.numel() == 0:
        raise ValueError("Structural labels require at least one sample.")
    if int(relabelled_labels.min()) < 0 or int(relabelled_labels.max()) >= num_classes:
        raise ValueError("relabelled_labels contain an invalid class index.")

    normalized_features = F.normalize(features, dim=1)
    feature_bank = normalized_features.T
    edge_counts = torch.zeros(
        features.size(0),
        num_classes,
        dtype=normalized_features.dtype,
        device=features.device,
    )
    flat_edge_counts = edge_counts.view(-1)

    for start, end in _chunk_bounds(features.size(0), chunks):
        similarity = torch.mm(normalized_features[start:end], feature_bank)
        local_rows = torch.arange(end - start, device=features.device)
        # Self is a valid nearest neighbor in Algorithm 1. Pinning it to the
        # maximum also guarantees every sample receives at least one label.
        similarity[local_rows, local_rows + start] = torch.inf
        neighbor_indices = similarity.topk(k=neighbors, dim=1).indices
        source_labels = relabelled_labels[start:end, None].expand(-1, neighbors)
        flat_targets = (
            neighbor_indices.reshape(-1) * num_classes + source_labels.reshape(-1)
        )
        flat_edge_counts.scatter_add_(
            0,
            flat_targets,
            torch.ones_like(flat_targets, dtype=edge_counts.dtype),
        )

    return edge_counts / edge_counts.sum(dim=1, keepdim=True)

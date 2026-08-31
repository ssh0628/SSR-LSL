"""SSR's supervised mixup and one-direction feature-consistency losses."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def negative_cosine_similarity(prediction: Tensor, projection: Tensor) -> Tensor:
    return -F.cosine_similarity(prediction, projection.detach(), dim=-1).mean()


def soft_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * targets, dim=1))


def mixup_two_views(
    first_view: Tensor,
    second_view: Tensor,
    labels: Tensor,
    *,
    num_classes: int = 10,
    alpha: float = 4.0,
) -> tuple[Tensor, Tensor, float]:
    """Apply the exact Beta(alpha, alpha) two-view mixup from official SSR."""
    one_hot = torch.zeros(
        labels.size(0), num_classes, device=first_view.device
    ).scatter_(1, labels.view(-1, 1), 1)
    coefficient = float(np.random.beta(alpha, alpha))
    coefficient = max(coefficient, 1.0 - coefficient)
    all_inputs = torch.cat([first_view, second_view], dim=0)
    all_targets = torch.cat([one_hot, one_hot], dim=0)

    # The upstream implementation creates this permutation on CPU. Keeping it
    # there preserves its RNG stream; PyTorch accepts CPU indices for CUDA data.
    permutation = torch.randperm(all_inputs.size(0))
    mixed_inputs = coefficient * all_inputs + (1.0 - coefficient) * all_inputs[permutation]
    mixed_targets = coefficient * all_targets + (1.0 - coefficient) * all_targets[permutation]
    return mixed_inputs, mixed_targets, coefficient

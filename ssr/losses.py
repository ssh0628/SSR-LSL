"""SSR의 mixup CE와 one-direction feature-consistency loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def negative_cosine_similarity(prediction: Tensor, projection: Tensor) -> Tensor:
    """projection branch의 gradient를 차단한 공식 SSR consistency loss."""
    return -F.cosine_similarity(prediction, projection.detach(), dim=-1).mean()


def soft_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """hard one-hot과 soft structural target을 모두 받는 cross-entropy."""
    return -(F.log_softmax(logits, dim=1) * targets).sum(dim=1).mean()


def mixup_soft_target_views(
    first_view: Tensor,
    second_view: Tensor,
    targets: Tensor,
    *,
    alpha: float,
) -> tuple[Tensor, Tensor, float]:
    """두 view와 soft target에 input-level Beta mixup 적용."""
    if first_view.shape != second_view.shape:
        raise ValueError("The two mixup views must have the same shape.")
    if targets.ndim != 2 or targets.size(0) != first_view.size(0):
        raise ValueError("targets must have shape (batch, classes).")

    coefficient = float(np.random.beta(alpha, alpha))
    coefficient = max(coefficient, 1.0 - coefficient)
    all_inputs = torch.cat([first_view, second_view], dim=0)
    all_targets = torch.cat([targets, targets], dim=0)

    # 공식 SSR처럼 CPU permutation을 생성해 기존 RNG stream을 유지한다.
    permutation = torch.randperm(all_inputs.size(0)).to(all_inputs.device)
    mixed_inputs = (
        coefficient * all_inputs
        + (1.0 - coefficient) * all_inputs[permutation]
    )
    mixed_targets = (
        coefficient * all_targets
        + (1.0 - coefficient) * all_targets[permutation]
    )
    return mixed_inputs, mixed_targets, coefficient


def mixup_hard_label_views(
    first_view: Tensor,
    second_view: Tensor,
    labels: Tensor,
    *,
    num_classes: int = 10,
    alpha: float = 4.0,
) -> tuple[Tensor, Tensor, float]:
    """공식 SSR selected-sample branch의 two-view mixup."""
    one_hot_targets = torch.zeros(
        labels.size(0),
        num_classes,
        device=first_view.device,
    ).scatter_(1, labels.view(-1, 1), 1)
    return mixup_soft_target_views(
        first_view,
        second_view,
        one_hot_targets,
        alpha=alpha,
    )

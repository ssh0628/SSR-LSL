"""LSL structural-label strong-view mixup cross-entropy."""

from __future__ import annotations

from torch import Tensor, nn

from ssr.losses import mixup_soft_target_views, soft_cross_entropy


def structural_mixup_loss(
    encoder: nn.Module,
    classifier: nn.Module,
    first_strong_view: Tensor,
    second_strong_view: Tensor,
    structural_targets: Tensor,
    *,
    mixup_alpha: float,
) -> Tensor:
    """논문 Algorithm 2의 L_st를 두 independent strong view로 계산."""
    mixed_inputs, mixed_targets, _ = mixup_soft_target_views(
        first_strong_view,
        second_strong_view,
        structural_targets,
        alpha=mixup_alpha,
    )
    logits = classifier(encoder(mixed_inputs))
    return soft_cross_entropy(logits, mixed_targets)

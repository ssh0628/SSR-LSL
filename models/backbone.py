"""Feature encoders for general image datasets."""

from __future__ import annotations

from torch import nn


def build_encoder(
    name: str,
    *,
    pretrained: bool,
    drop_path_rate: float,
) -> tuple[nn.Module, int]:
    from timm import create_model

    if name == "convnextv2_tiny":
        name = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    encoder = create_model(
        name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
        drop_path_rate=drop_path_rate,
    )
    feature_dim = int(encoder.num_features)
    if feature_dim < 1:
        raise ValueError(f"Backbone {name} has no pooled feature dimension.")
    return encoder, feature_dim

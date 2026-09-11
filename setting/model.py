"""일반 이미지 backbone과 SSR projection heads 구성."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.backbone import build_encoder
from setting.config import ModelConfig


BatchNorm: TypeAlias = (
    nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d | nn.SyncBatchNorm
)
_BATCH_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


@dataclass(frozen=True, slots=True)
class SSRNetworks:
    """SSR와 LSL이 공동으로 최적화하는 네 개의 network."""

    encoder: nn.Module
    classifier: nn.Module
    projector: nn.Module
    predictor: nn.Module

    def train(self) -> None:
        for module in self.all_modules():
            module.train()

    def eval(self) -> None:
        for module in self.all_modules():
            module.eval()

    def all_modules(self) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        return self.encoder, self.classifier, self.projector, self.predictor


def build_ssr_networks(
    config: ModelConfig,
    device: torch.device,
    num_classes: int,
) -> SSRNetworks:
    """backbone만 교체 가능하게 두고 SSR heads는 공식 구성 그대로 생성."""
    encoder, feature_dim = build_encoder(
        config.name,
        pretrained=config.pretrained,
        drop_path_rate=config.drop_path_rate,
    )
    classifier = nn.Linear(feature_dim, num_classes)
    projector = nn.Sequential(
        nn.Linear(feature_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    predictor = nn.Sequential(
        nn.Linear(128, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    networks = SSRNetworks(encoder, classifier, projector, predictor)
    for module in networks.all_modules():
        module.to(device)
    return networks


@torch.no_grad()
def calibrate_batch_norm(
    loader: DataLoader,
    encoder: nn.Module,
    device: torch.device,
) -> int:
    """Estimate fresh encoder BN statistics without updating model parameters."""
    batch_norms: list[BatchNorm] = [
        module
        for module in encoder.modules()
        if isinstance(module, _BATCH_NORM_TYPES) and module.track_running_stats
    ]
    if not batch_norms:
        return 0

    training_states = [(module, module.training) for module in encoder.modules()]
    momenta = [(module, module.momentum) for module in batch_norms]
    batches = 0
    try:
        encoder.train()
        for module in batch_norms:
            module.reset_running_stats()
            # Cumulative average uses every calibration batch equally.
            module.momentum = None
        for images, _ in tqdm(loader, desc="BN calibration", leave=False, disable=None):
            encoder(images.to(device, non_blocking=True))
            batches += 1
    finally:
        for module, momentum in momenta:
            module.momentum = momentum
        for module, was_training in training_states:
            module.training = was_training
    return batches

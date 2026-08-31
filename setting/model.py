"""선택한 CIFAR backbone과 SSR projection heads 구성."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.cifar_resnet import build_cifar_resnet
from models.preresnet import PreResNet18
from setting.config import ModelConfig


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


def _build_encoder(config: ModelConfig) -> tuple[nn.Module, int]:
    if config.name == "preact_resnet18":
        encoder = PreResNet18(num_classes=10)
        feature_dim = encoder.fc.in_features
    else:
        encoder = build_cifar_resnet(config.name, num_classes=10)
        feature_dim = encoder.feature_dim
    encoder.fc = nn.Identity()
    return encoder, feature_dim


def build_ssr_networks(config: ModelConfig, device: torch.device) -> SSRNetworks:
    """backbone만 교체 가능하게 두고 SSR heads는 공식 구성 그대로 생성."""
    encoder, feature_dim = _build_encoder(config)
    classifier = nn.Linear(feature_dim, 10)
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

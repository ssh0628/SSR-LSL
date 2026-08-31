"""Construct the exact encoder and heads used by the official CIFAR SSR code."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.preresnet import PreResNet18


@dataclass(frozen=True, slots=True)
class SSRNetworks:
    encoder: nn.Module
    classifier: nn.Module
    projector: nn.Module
    predictor: nn.Module

    def train(self) -> None:
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        for module in self.modules():
            module.eval()

    def modules(self) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        return self.encoder, self.classifier, self.projector, self.predictor


def build_ssr_networks(device: torch.device) -> SSRNetworks:
    encoder = PreResNet18(num_classes=10)
    feature_dim = encoder.fc.in_features
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
    encoder.fc = nn.Identity()
    networks = SSRNetworks(encoder, classifier, projector, predictor)
    for module in networks.modules():
        module.to(device)
    return networks

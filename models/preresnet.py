"""공식 SSR CIFAR 코드의 PreActResNet-18 backbone.

원본 파일에 함께 있던 미사용 post-activation block, bottleneck,
34/50/101/152 factory는 제거했다. 아래 PreActBlock 순서와 module 생성 순서는
원본과 동일하다.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = _conv3x3(in_channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = _conv3x3(channels, channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = F.relu(self.bn1(inputs))
        shortcut = self.shortcut(outputs)
        outputs = self.conv1(outputs)
        outputs = self.conv2(F.relu(self.bn2(outputs)))
        return outputs + shortcut


class PreActResNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = _conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers: list[nn.Module] = []
        for block_stride in strides:
            layers.append(PreActBlock(self.in_channels, channels, block_stride))
            self.in_channels = channels * PreActBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = F.relu(self.bn1(self.conv1(inputs)))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = F.avg_pool2d(outputs, 4)
        return self.fc(outputs.view(outputs.size(0), -1))


def PreResNet18(num_classes: int = 10) -> PreActResNet:
    """공식 SSR과 동일한 CIFAR PreActResNet-18."""
    return PreActResNet(num_classes=num_classes)


__all__ = ["PreActResNet", "PreResNet18"]

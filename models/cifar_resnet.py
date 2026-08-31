"""RLNLC의 CIFAR-sized torchvision ResNet-18/34 backbone."""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, ResNet


class CifarResNet(ResNet):
    """ImageNet ResNet의 stem만 32x32 CIFAR 입력에 맞춘 모델."""

    def __init__(self, layers: tuple[int, int, int, int], num_classes: int) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")
        super().__init__(block=BasicBlock, layers=layers, num_classes=num_classes)
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.maxpool = nn.Identity()
        self.feature_dim = self.fc.in_features
        self._reset_cifar_parameters()

    def _reset_cifar_parameters(self) -> None:
        """RLNLC 구현과 같은 CIFAR backbone 초기화."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward_features(self, images: Tensor) -> Tensor:
        features = self.conv1(images)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        return self.layer4(features)

    def forward_head(self, features: Tensor, *, pre_logits: bool = False) -> Tensor:
        embeddings = self.avgpool(features).flatten(1)
        return embeddings if pre_logits else self.fc(embeddings)

    def _forward_impl(self, images: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(images))


def build_cifar_resnet(model_name: str, num_classes: int) -> CifarResNet:
    """RLNLC와 동일한 layer layout으로 ResNet-18 또는 ResNet-34 생성."""
    layer_layouts = {
        "cifar_resnet18": (2, 2, 2, 2),
        "cifar_resnet34": (3, 4, 6, 3),
    }
    try:
        return CifarResNet(layer_layouts[model_name], num_classes)
    except KeyError as error:
        raise ValueError(f"Unsupported CIFAR ResNet: {model_name!r}.") from error

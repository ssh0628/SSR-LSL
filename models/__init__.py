"""CIFAR-10 backbone definitions."""

from models.cifar_resnet import CifarResNet, build_cifar_resnet
from models.preresnet import PreResNet18

__all__ = ["CifarResNet", "PreResNet18", "build_cifar_resnet"]

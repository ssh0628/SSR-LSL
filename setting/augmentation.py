"""CIFAR-10 transforms used by the official SSR training recipe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image
from torch import Tensor
from torchvision import transforms

from setting.autoaugment import CIFAR10Policy


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
ImageTransform = Callable[[Image.Image], Tensor]


@dataclass(frozen=True, slots=True)
class CIFAR10Transforms:
    none: ImageTransform
    weak: ImageTransform
    strong: ImageTransform


class TwoStrongViews:
    """The two strong views used by SSR's supervised mixup branch."""

    def __init__(self, strong_transform: ImageTransform) -> None:
        self.strong_transform = strong_transform

    def __call__(self, image: Image.Image) -> list[Tensor]:
        return [self.strong_transform(image), self.strong_transform(image)]


class AllSampleViews:
    """전체 데이터 branch의 weak/strong view 묶음.

    LSL이 꺼지면 공식 SSR과 같은 ``[weak, strong]``만 생성한다. 켜지면
    structural mixup 전용 independent strong view를 하나 더 생성한다.
    """

    def __init__(
        self,
        weak_transform: ImageTransform,
        strong_transform: ImageTransform,
        *,
        include_structural_view: bool,
    ) -> None:
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.include_structural_view = include_structural_view

    def __call__(self, image: Image.Image) -> list[Tensor]:
        views = [self.weak_transform(image), self.strong_transform(image)]
        if self.include_structural_view:
            views.append(self.strong_transform(image))
        return views


def build_cifar10_transforms() -> CIFAR10Transforms:
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    weak = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    strong = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    none = transforms.Compose([transforms.ToTensor(), normalize])
    return CIFAR10Transforms(none=none, weak=weak, strong=strong)

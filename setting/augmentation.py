"""Image transforms and independent SSR/LSL training views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PIL import Image
from torch import Tensor
from torchvision import transforms

from setting.config import AugmentationConfig, DataConfig


ImageTransform = Callable[[Image.Image], Tensor]


@dataclass(frozen=True, slots=True)
class ImageTransforms:
    evaluation: ImageTransform
    weak: ImageTransform
    strong: ImageTransform


class TwoStrongViews:
    """Two independent strong views for SSR's selected-sample mixup CE."""

    def __init__(self, strong_transform: ImageTransform) -> None:
        self.strong_transform = strong_transform

    def __call__(self, image: Image.Image) -> list[Tensor]:
        return [self.strong_transform(image), self.strong_transform(image)]


class AllSampleViews:
    """Weak/strong feature-consistency views, plus LSL's independent strong view."""

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


def build_image_transforms(data: DataConfig, config: AugmentationConfig) -> ImageTransforms:
    """Augment images already cropped and resized once by the dataset."""
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(data.mean, data.std)

    def augmented(rotation: float, *, color: bool) -> ImageTransform:
        operations = [
            transforms.RandomHorizontalFlip(config.horizontal_flip),
            transforms.RandomVerticalFlip(config.vertical_flip),
            transforms.RandomRotation(rotation),
        ]
        if color:
            operations.append(transforms.ColorJitter(*config.color_jitter))
        return transforms.Compose([*operations, to_tensor, normalize])

    return ImageTransforms(
        evaluation=transforms.Compose([to_tensor, normalize]),
        weak=augmented(config.weak_rotation, color=False),
        strong=augmented(config.strong_rotation, color=True),
    )

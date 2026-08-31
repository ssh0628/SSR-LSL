"""공식 SSR에서 사용하는 CIFAR-10 AutoAugment policy만 보존한 모듈."""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


class _SubPolicy:
    def __init__(
        self,
        probability1: float,
        operation1: str,
        magnitude_index1: int,
        probability2: float,
        operation2: str,
        magnitude_index2: int,
        fill_color: tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        ranges = {
            "sheary": np.linspace(0, 0.3, 10),
            "translatex": np.linspace(0, 150 / 331, 10),
            "translatey": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }

        def rotate_with_fill(image: Image.Image, magnitude: float) -> Image.Image:
            rotated = image.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rotated,
                Image.new("RGBA", rotated.size, (128,) * 4),
                rotated,
            ).convert(image.mode)

        operations: dict[str, Callable[[Image.Image, float], Image.Image]] = {
            "sheary": lambda image, magnitude: image.transform(
                image.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fill_color,
            ),
            "translatex": lambda image, magnitude: image.transform(
                image.size,
                Image.AFFINE,
                (1, 0, magnitude * image.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fill_color,
            ),
            "translatey": lambda image, magnitude: image.transform(
                image.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * image.size[1] * random.choice([-1, 1])),
                fillcolor=fill_color,
            ),
            "rotate": rotate_with_fill,
            "color": lambda image, magnitude: ImageEnhance.Color(image).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "posterize": lambda image, magnitude: ImageOps.posterize(image, magnitude),
            "solarize": lambda image, magnitude: ImageOps.solarize(image, magnitude),
            "contrast": lambda image, magnitude: ImageEnhance.Contrast(image).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "sharpness": lambda image, magnitude: ImageEnhance.Sharpness(image).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "brightness": lambda image, magnitude: ImageEnhance.Brightness(image).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "autocontrast": lambda image, _: ImageOps.autocontrast(image),
            "equalize": lambda image, _: ImageOps.equalize(image),
            "invert": lambda image, _: ImageOps.invert(image),
        }

        name1 = operation1.lower()
        name2 = operation2.lower()
        self.probability1 = probability1
        self.operation1 = operations[name1]
        self.magnitude1 = ranges[name1][magnitude_index1]
        self.probability2 = probability2
        self.operation2 = operations[name2]
        self.magnitude2 = ranges[name2][magnitude_index2]

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.probability1:
            image = self.operation1(image, self.magnitude1)
        if random.random() < self.probability2:
            image = self.operation2(image, self.magnitude2)
        return image


class CIFAR10Policy:
    """AutoAugment 논문에서 찾은 CIFAR-10 25개 sub-policy 중 하나를 무작위 적용."""

    def __init__(self, fill_color: tuple[int, int, int] = (128, 128, 128)) -> None:
        self.policies = [
            _SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fill_color),
            _SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fill_color),
            _SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fill_color),
            _SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fill_color),
            _SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fill_color),
            _SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fill_color),
            _SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fill_color),
            _SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fill_color),
            _SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fill_color),
            _SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fill_color),
            _SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fill_color),
            _SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fill_color),
            _SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fill_color),
            _SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fill_color),
            _SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fill_color),
            _SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fill_color),
            _SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fill_color),
            _SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fill_color),
            _SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fill_color),
            _SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fill_color),
            _SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fill_color),
            _SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fill_color),
            _SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fill_color),
            _SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fill_color),
            _SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fill_color),
        ]

    def __call__(self, image: Image.Image) -> Image.Image:
        policy_index = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_index](image)

    def __repr__(self) -> str:
        return "AutoAugment CIFAR10 Policy"


__all__ = ["CIFAR10Policy"]

"""CIFAR-10-only data setup and reproducible synthetic label noise."""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from scipy import stats
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from setting.augmentation import TwoStrongViews, WeakStrongViews, build_cifar10_transforms
from setting.config import DataConfig


NUM_CLASSES = 10
ASYMMETRIC_TRANSITION = {0: 0, 1: 1, 2: 0, 3: 5, 4: 7, 5: 3, 6: 6, 7: 7, 8: 8, 9: 1}


@torch.no_grad()
def inject_instance_dependent_noise(
    images: Tensor,
    clean_labels: Tensor,
    *,
    noise_rate: float,
    flip_rate_std: float = 0.1,
    num_classes: int | None = None,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate Xia et al. IDN, preserving the RLNLC implementation."""
    if images.ndim != 4:
        raise ValueError(f"images must have shape (N, C, H, W), got {tuple(images.shape)}.")
    if clean_labels.ndim != 1:
        raise ValueError(f"clean_labels must have shape (N,), got {tuple(clean_labels.shape)}.")
    if images.shape[0] != clean_labels.numel():
        raise ValueError("images and clean_labels must contain the same number of samples.")
    if clean_labels.numel() == 0:
        raise ValueError("clean_labels must not be empty.")
    if not 0.0 <= noise_rate < 1.0:
        raise ValueError("noise_rate must be in [0, 1).")
    if flip_rate_std <= 0.0:
        raise ValueError("flip_rate_std must be positive.")

    labels = clean_labels.detach().cpu().to(torch.long).contiguous()
    inferred_classes = int(labels.max()) + 1
    class_count = inferred_classes if num_classes is None else num_classes
    if class_count < 2:
        raise ValueError("num_classes must be at least 2.")
    if int(labels.min()) < 0 or int(labels.max()) >= class_count:
        raise ValueError("clean_labels contain a class outside [0, num_classes).")
    if noise_rate == 0.0:
        return labels.clone(), torch.zeros_like(labels, dtype=torch.bool)

    if device is None:
        compute_device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    else:
        compute_device = torch.device(device)
    features = images.detach().reshape(images.size(0), -1).to(
        device=compute_device, dtype=torch.float32
    )

    # RandomState and the class-wise operations intentionally preserve RLNLC's
    # random-number sequence and floating-point procedure.
    rng = np.random.RandomState(seed)
    distribution = stats.truncnorm(
        (0.0 - noise_rate) / flip_rate_std,
        (1.0 - noise_rate) / flip_rate_std,
        loc=noise_rate,
        scale=flip_rate_std,
    )
    flip_rates = distribution.rvs(labels.numel(), random_state=rng).astype(
        np.float32, copy=False
    )
    weights = torch.from_numpy(
        rng.randn(class_count, features.size(1), class_count).astype(np.float32, copy=False)
    ).to(compute_device)
    probabilities = torch.empty((labels.numel(), class_count), dtype=torch.float32)

    for class_id in range(class_count):
        indices = labels.eq(class_id).nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        device_indices = indices.to(compute_device)
        logits = features[device_indices].matmul(weights[class_id])
        logits[:, class_id] = -torch.inf
        class_probabilities = torch.softmax(logits, dim=1)
        class_flip_rates = torch.from_numpy(flip_rates[indices.numpy()]).to(compute_device)
        class_probabilities.mul_(class_flip_rates[:, None])
        class_probabilities[:, class_id] = 1.0 - class_flip_rates
        probabilities[indices] = class_probabilities.cpu()

    probability_array = probabilities.numpy().astype(np.float64, copy=False)
    probability_array /= probability_array.sum(axis=1, keepdims=True)
    noisy_array = np.fromiter(
        (rng.choice(class_count, p=row) for row in probability_array),
        dtype=np.int64,
        count=labels.numel(),
    )
    noisy_labels = torch.from_numpy(noisy_array).contiguous()
    return noisy_labels, noisy_labels.ne(labels).contiguous()


def inject_closed_set_noise(
    clean_labels: Tensor,
    *,
    noise_kind: str,
    noise_rate: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Generate the official SSR symmetric or asymmetric CIFAR-10 setup."""
    if noise_kind not in {"symmetric", "asymmetric"}:
        raise ValueError("noise_kind must be 'symmetric' or 'asymmetric'.")
    labels = clean_labels.detach().cpu().to(torch.long).contiguous()
    noisy = labels.clone()
    rng = random.Random(seed)
    indices = list(range(labels.numel()))
    rng.shuffle(indices)
    for index in indices[: int(noise_rate * labels.numel())]:
        if noise_kind == "symmetric":
            noisy[index] = rng.randint(0, NUM_CLASSES - 1)
        else:
            noisy[index] = ASYMMETRIC_TRANSITION[int(labels[index])]
    return noisy, noisy.ne(labels).contiguous()


class CIFAR10TrainDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        noisy_labels: Tensor,
        clean_labels: Tensor,
        transform: Callable[[Image.Image], object],
    ) -> None:
        self.images = images
        self.noisy_labels = noisy_labels
        self.clean_labels = clean_labels
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[object, Tensor, Tensor, int]:
        image = Image.fromarray(self.images[index])
        return (
            self.transform(image),
            self.noisy_labels[index],
            self.clean_labels[index],
            index,
        )

    def __len__(self) -> int:
        return len(self.images)


class CIFAR10TestDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        clean_labels: Tensor,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        self.images = images
        self.clean_labels = clean_labels
        self.transform = transform

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        return self.transform(Image.fromarray(self.images[index])), self.clean_labels[index], index

    def __len__(self) -> int:
        return len(self.images)


@dataclass(frozen=True, slots=True)
class CIFAR10Data:
    selected_train: CIFAR10TrainDataset
    evaluation_train: CIFAR10TrainDataset
    all_train: CIFAR10TrainDataset
    test: CIFAR10TestDataset
    noisy_labels: Tensor
    clean_labels: Tensor
    noise_mask: Tensor


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_or_create_noisy_labels(
    images: np.ndarray,
    clean_labels: Tensor,
    config: DataConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if config.noise_file.exists():
        artifact = torch.load(config.noise_file, map_location="cpu", weights_only=True)
        expected_metadata = {
            "dataset": "CIFAR10",
            "noise_kind": config.noise_kind,
            "noise_rate": config.noise_rate,
            "noise_seed": config.noise_seed,
        }
        if config.noise_kind == "idn":
            expected_metadata["idn_flip_rate_std"] = config.idn_flip_rate_std
        for key, expected in expected_metadata.items():
            if artifact.get(key) != expected:
                raise RuntimeError(
                    f"Noise artifact metadata mismatch for {key}: {config.noise_file}"
                )
        noisy_labels = artifact["noisy_labels"].to(torch.long).contiguous()
        if not torch.equal(artifact["clean_labels"].to(torch.long), clean_labels):
            raise RuntimeError(f"Noise artifact does not match CIFAR-10 labels: {config.noise_file}")
        return noisy_labels, noisy_labels.ne(clean_labels).contiguous()

    if config.noise_kind == "idn":
        raw = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        noisy_labels, noise_mask = inject_instance_dependent_noise(
            raw,
            clean_labels,
            noise_rate=config.noise_rate,
            flip_rate_std=config.idn_flip_rate_std,
            num_classes=NUM_CLASSES,
            seed=config.noise_seed,
            device=device,
        )
    else:
        noisy_labels, noise_mask = inject_closed_set_noise(
            clean_labels,
            noise_kind=config.noise_kind,
            noise_rate=config.noise_rate,
            seed=config.noise_seed,
        )

    _atomic_torch_save(
        {
            "dataset": "CIFAR10",
            "noise_kind": config.noise_kind,
            "noise_rate": config.noise_rate,
            "noise_seed": config.noise_seed,
            "idn_flip_rate_std": config.idn_flip_rate_std,
            "noisy_labels": noisy_labels,
            "clean_labels": clean_labels,
        },
        config.noise_file,
    )
    return noisy_labels, noise_mask


def build_cifar10_data(config: DataConfig, device: torch.device) -> CIFAR10Data:
    """Build the four dataset views needed by SSR from one CIFAR-10 copy."""
    train_source = CIFAR10(root=config.root, train=True, download=config.download)
    test_source = CIFAR10(root=config.root, train=False, download=config.download)
    train_images = np.asarray(train_source.data)
    test_images = np.asarray(test_source.data)
    clean_labels = torch.as_tensor(train_source.targets, dtype=torch.long)
    test_labels = torch.as_tensor(test_source.targets, dtype=torch.long)
    noisy_labels, noise_mask = _load_or_create_noisy_labels(
        train_images, clean_labels, config, device
    )
    augmentations = build_cifar10_transforms()

    return CIFAR10Data(
        selected_train=CIFAR10TrainDataset(
            train_images,
            noisy_labels,
            clean_labels,
            TwoStrongViews(augmentations.strong),
        ),
        evaluation_train=CIFAR10TrainDataset(
            train_images, noisy_labels, clean_labels, augmentations.weak
        ),
        all_train=CIFAR10TrainDataset(
            train_images,
            noisy_labels,
            clean_labels,
            WeakStrongViews(augmentations.weak, augmentations.strong),
        ),
        test=CIFAR10TestDataset(test_images, test_labels, augmentations.none),
        noisy_labels=noisy_labels,
        clean_labels=clean_labels,
        noise_mask=noise_mask,
    )

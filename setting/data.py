"""Path/label NPY datasets with stable per-image training indices."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from setting.augmentation import AllSampleViews, TwoStrongViews, build_image_transforms
from setting.config import AugmentationConfig, DataConfig, SplitConfig
from setting.image_io import load_rgb_image, verify_image_files


@dataclass(frozen=True, slots=True)
class SplitSource:
    name: str
    paths: tuple[str, ...]
    labels: Tensor


def _resolve_array(root: Path, filename: str) -> Path:
    path = Path(filename)
    path = path if path.is_absolute() else root / path
    if not path.is_file():
        raise FileNotFoundError(f"Dataset array not found: {path}")
    return path


def _load_split(config: DataConfig, split: SplitConfig, name: str) -> SplitSource:
    paths_file = _resolve_array(config.root, split.paths)
    labels_file = _resolve_array(config.root, split.labels)
    paths_array = np.load(paths_file, allow_pickle=False)
    labels_array = np.load(labels_file, allow_pickle=False)
    if (
        paths_array.ndim != 1 or labels_array.ndim != 1
        or len(paths_array) != len(labels_array)
    ):
        raise ValueError(f"{name}: paths and labels must be aligned rank-1 arrays.")
    if not len(paths_array):
        raise ValueError(f"{name}: dataset is empty.")
    if paths_array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name}: paths must be a string array saved without pickle.")
    if labels_array.dtype.kind not in {"i", "u", "f"}:
        raise ValueError(f"{name}: labels must be numeric class indices.")
    if labels_array.dtype.kind == "f" and (
        not np.isfinite(labels_array).all()
        or np.any(labels_array != np.floor(labels_array))
    ):
        raise ValueError(f"{name}: labels must be finite integer class indices.")
    minimum, maximum = config.label_offset, config.label_offset + config.num_classes
    if np.any(labels_array < minimum) or np.any(labels_array >= maximum):
        raise ValueError(
            f"{name}: labels must be in [{minimum}, {maximum}); "
            "check class_names and label_offset."
        )
    labels = labels_array.astype(np.int64, copy=False)
    if config.label_offset:
        labels = labels - config.label_offset
    image_root = config.image_root or config.root
    paths: list[str] = []
    for raw_path in paths_array:
        value = raw_path.decode("utf-8") if isinstance(raw_path, bytes) else str(raw_path)
        if not value.strip():
            raise ValueError(f"{name}: empty image path at index {len(paths)}.")
        path = Path(value).expanduser()
        paths.append(str((path if path.is_absolute() else image_root / path).resolve()))
    if len(set(paths)) != len(paths):
        raise ValueError(f"{name}: duplicate image paths are not allowed.")
    return SplitSource(name, tuple(paths), torch.from_numpy(labels))


def inspect_dataset(
    config: DataConfig, *, report_path: Path | None = None,
) -> dict[str, SplitSource]:
    """Validate configured arrays, split separation, and optional full image decode."""
    config.validate()
    split_configs = (
        ("train", config.train), ("val", config.validation), ("test", config.test)
    )
    sources = {
        name: _load_split(config, split_config, name)
        for name, split_config in split_configs
        if split_config is not None
    }
    seen: set[str] = set()
    for name, source in sources.items():
        overlap = seen.intersection(source.paths)
        if overlap:
            raise ValueError(
                f"train/val/test paths must be disjoint; "
                f"{name} overlaps at {min(overlap)}."
            )
        seen.update(source.paths)
        counts = torch.bincount(source.labels, minlength=config.num_classes).tolist()
        print(
            f"[{name}] {len(source.paths)} images, "
            f"class counts={dict(zip(config.class_names, counts))}"
        )
    if config.verify_images:
        _verify_sources(config, sources, report_path)
    return sources


def _verify_sources(
    config: DataConfig, sources: dict[str, SplitSource], report_path: Path | None,
) -> None:
    tasks = (
        (name, index, path, config.allow_truncated_images)
        for name, source in sources.items()
        for index, path in enumerate(source.paths)
    )
    verify_image_files(
        tasks, count=sum(len(source.paths) for source in sources.values()),
        workers=config.image_check_workers, report_path=report_path,
    )


class IndexedImageDataset(Dataset):
    """Decode and resize a full image in memory; never modify source files."""

    def __init__(
        self,
        source: SplitSource,
        transform: Callable[[Image.Image], object],
        *,
        image_size: int,
        allow_truncated: bool,
        labeled: bool = False,
    ) -> None:
        self.source = source
        self.transform = transform
        self.image_size = image_size
        self.allow_truncated = allow_truncated
        self.labeled = labeled

    def __len__(self) -> int:
        return len(self.source.paths)

    def __getitem__(self, index: int) -> tuple[object, int | Tensor]:
        path = self.source.paths[index]
        try:
            image, _ = load_rgb_image(path, allow_truncated=self.allow_truncated)
            with closing(image):
                resized = image.resize(
                    (self.image_size, self.image_size), Image.Resampling.BICUBIC,
                )
            # Release the full-resolution image before creating multiple views.
            with closing(resized):
                transformed = self.transform(resized)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise RuntimeError(f"Failed to load {self.source.name}[{index}]: {path}") from error
        return transformed, self.source.labels[index] if self.labeled else index


@dataclass(frozen=True, slots=True)
class ExperimentData:
    calibration_train: Dataset
    selected_train: Dataset
    evaluation_train: Dataset
    all_train: Dataset
    validation: Dataset | None
    test: Dataset | None
    noisy_labels: Tensor
    num_classes: int
    class_names: tuple[str, ...]


def build_experiment_data(
    config: DataConfig,
    *,
    structural_labels_enabled: bool,
    augmentation: AugmentationConfig,
    report_path: Path | None = None,
) -> ExperimentData:
    """Use provided labels unchanged; transforms run in seeded DataLoader workers."""
    sources = inspect_dataset(config, report_path=report_path)
    transforms = build_image_transforms(config, augmentation)

    def dataset(
        source: SplitSource,
        transform: Callable[[Image.Image], object],
        *,
        labeled: bool = False,
    ) -> IndexedImageDataset:
        return IndexedImageDataset(
            source, transform, image_size=config.image_size,
            allow_truncated=config.allow_truncated_images, labeled=labeled,
        )

    train = sources["train"]
    fixed_train = dataset(train, transforms.evaluation)
    return ExperimentData(
        calibration_train=fixed_train,
        selected_train=dataset(train, TwoStrongViews(transforms.strong)),
        evaluation_train=fixed_train,
        all_train=dataset(
            train,
            AllSampleViews(
                transforms.weak, transforms.strong,
                include_structural_view=structural_labels_enabled,
            ),
        ),
        validation=(
            dataset(sources["val"], transforms.evaluation, labeled=True)
            if "val" in sources else None
        ),
        test=(
            dataset(sources["test"], transforms.evaluation, labeled=True)
            if "test" in sources else None
        ),
        noisy_labels=train.labels,
        num_classes=config.num_classes,
        class_names=tuple(config.class_names),
    )

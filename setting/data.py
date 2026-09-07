"""Provided path/label NPY datasets with stable per-image SSR indices."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from setting.augmentation import AllSampleViews, TwoStrongViews, build_image_transforms
from setting.bbox import BBox, clamp_bbox, prepare_bboxes
from setting.config import AugmentationConfig, DataConfig, SplitConfig
from setting.image_io import load_rgb_image, verify_image_files


@dataclass(frozen=True, slots=True)
class SplitSource:
    name: str
    paths: tuple[str, ...]
    labels: Tensor
    raw_paths: tuple[str, ...] = ()


def _resolve_array(root: Path, filename: str) -> Path:
    path = Path(filename)
    path = path if path.is_absolute() else root / path
    if not path.is_file() and path.name in {
        f"{split}_{suffix}.npy"
        for split in ("train", "val", "test")
        for suffix in ("path", "paths")
    }:
        old, new = (
            ("_paths.npy", "_path.npy")
            if path.name.endswith("_paths.npy") else ("_path.npy", "_paths.npy")
        )
        alias = path.with_name(path.name.replace(old, new))
        if alias.is_file():
            return alias
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
    raw_paths = tuple(str(value.item()) if isinstance(value, np.generic) else str(value) for value in paths_array)
    return SplitSource(name, tuple(paths), torch.from_numpy(labels), raw_paths)


def dataset_bboxes(
    config: DataConfig, sources: dict[str, SplitSource], *,
    report_path: Path | None = None,
) -> dict[str, tuple[BBox | None, ...]]:
    """Resolve coordinates once, before image preprocessing or model creation."""
    if not config.crop_bbox:
        return {}
    split_configs = {"train": config.train, "val": config.validation, "test": config.test}
    return {
        name: prepare_bboxes(
            config, split_configs[name], name, source.paths,
            raw_paths=source.raw_paths or None, report_path=report_path,
        )
        for name, source in sources.items()
    }


def filter_bbox_sources(
    config: DataConfig,
    sources: dict[str, SplitSource],
    boxes: dict[str, tuple[BBox | None, ...]],
    *, report_path: Path | None = None,
) -> tuple[dict[str, SplitSource], dict[str, tuple[BBox | None, ...]]]:
    """Apply one stable mask to paths, labels and boxes before training."""
    if not config.crop_bbox or config.missing_bbox != "drop":
        return sources, boxes
    filtered_sources: dict[str, SplitSource] = {}
    filtered_boxes: dict[str, tuple[BBox | None, ...]] = {}
    for name, source in sources.items():
        values = boxes[name]
        if len(values) != len(source.paths):
            raise ValueError(f"{name}: bbox count does not match image count.")
        kept = [index for index, box in enumerate(values) if box is not None]
        excluded_count = len(values) - len(kept)
        labels = (
            source.labels[torch.tensor(kept, dtype=torch.int64)]
            if excluded_count else source.labels
        )
        counts = torch.bincount(labels, minlength=config.num_classes).tolist()
        class_counts = dict(zip(config.class_names, counts))
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("a", encoding="utf-8") as report:
                for index, box in enumerate(values):
                    if box is None:
                        label = int(source.labels[index])
                        report.write(json.dumps({
                            "status": "bbox_excluded", "split": name, "index": index,
                            "path": source.paths[index], "label": label,
                            "class_name": config.class_names[label], "reason": "missing_bbox",
                        }, ensure_ascii=False) + "\n")
                report.write(json.dumps({
                    "status": "bbox_filter_summary", "split": name,
                    "input_count": len(values), "excluded_count": excluded_count,
                    "retained_count": len(kept), "class_counts": class_counts,
                }, ensure_ascii=False) + "\n")
        print(
            f"[{name}] Bbox filter: {len(values)} -> {len(kept)} images; "
            f"excluded={excluded_count}, class counts={class_counts}"
        )
        if not kept:
            raise ValueError(f"{name}: no samples remain after excluding missing bboxes.")
        if excluded_count:
            source = replace(
                source, paths=tuple(source.paths[index] for index in kept), labels=labels,
                raw_paths=tuple(source.raw_paths[index] for index in kept) if source.raw_paths else (),
            )
            values = tuple(values[index] for index in kept)
        filtered_sources[name] = source
        filtered_boxes[name] = values
    return filtered_sources, filtered_boxes


def _validated_crop_boxes(
    source: SplitSource, boxes: tuple[BBox | None, ...],
    *, missing: str, report_path: Path | None,
) -> tuple[BBox | None, ...]:
    """Clamp against native dimensions once; reject unusable ROIs before training."""
    if len(boxes) != len(source.paths):
        raise ValueError(f"{source.name}: bbox count does not match image count.")
    valid = []
    failures = []
    exceptional = []
    for index, (path, box) in enumerate(zip(source.paths, boxes)):
        if box is None:
            valid.append(None)
            continue
        with Image.open(path) as image:
            width, height = image.size
        clipped = clamp_bbox(box, width, height)
        if clipped is None or clipped != box:
            exceptional.append({
                "split": source.name, "index": index, "path": path,
                "status": "bbox_clipped" if clipped else "bbox_outside_image",
                "bbox": box, "clipped_bbox": clipped,
            })
        if clipped is None and missing != "full":
            failures.append(f"{source.name}[{index}]: {path} bbox={box}")
        valid.append(clipped)
    if exceptional and report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("a", encoding="utf-8") as report:
            for row in exceptional:
                report.write(json.dumps(row, ensure_ascii=False) + "\n")
    if failures:
        raise ValueError("Bboxes outside source images:\n" + "\n".join(failures[:10]))
    missing_count = sum(box is None for box in valid)
    print(f"[{source.name}] ROI crop: {len(valid) - missing_count}, full-image fallback: {missing_count}")
    return tuple(valid)


def inspect_dataset(
    config: DataConfig, *, report_path: Path | None = None
) -> dict[str, SplitSource]:
    """Validate configured arrays, split separation, and optional full image decode."""
    config.validate()
    split_configs = (
        ("train", config.train), ("val", config.validation), ("test", config.test)
    )
    sources = {
        name: _load_split(config, split, name)
        for name, split in split_configs
        if split is not None
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
    def __init__(
        self,
        source: SplitSource,
        transform: Callable[[Image.Image], object],
        *,
        allow_truncated: bool,
        image_size: int,
        labeled: bool = False,
        bboxes: tuple[BBox | None, ...] | None = None,
    ) -> None:
        self.source = source
        self.transform = transform
        self.allow_truncated = allow_truncated
        self.labeled = labeled
        self.image_size = image_size
        self.bboxes = bboxes

    def __len__(self) -> int:
        return len(self.source.paths)

    def __getitem__(self, index: int) -> tuple[object, int | Tensor]:
        path = self.source.paths[index]
        box = self.bboxes[index] if self.bboxes is not None else None
        image = None
        try:
            image, _ = load_rgb_image(path, allow_truncated=self.allow_truncated)
            if box is not None:
                cropped = image.crop(box)
                image.close()
                image = cropped
            size = (self.image_size, self.image_size)
            if image.size != size:
                resized = image.resize(size, Image.Resampling.BICUBIC)
                image.close()
                image = resized
            transformed = self.transform(image)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise RuntimeError(f"Failed to load {self.source.name}[{index}]: {path}") from error
        finally:
            if image is not None:
                image.close()
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
    sources = inspect_dataset(
        replace(config, verify_images=False), report_path=report_path,
    )
    boxes = dataset_bboxes(config, sources, report_path=report_path)
    sources, boxes = filter_bbox_sources(config, sources, boxes, report_path=report_path)
    if config.verify_images:
        _verify_sources(config, sources, report_path)
    boxes = {
        name: _validated_crop_boxes(
            sources[name], values,
            missing=config.missing_bbox, report_path=report_path,
        )
        for name, values in boxes.items()
    }
    transforms = build_image_transforms(config, augmentation)

    def dataset(
        name: str, transform: Callable, *, labeled: bool = False
    ) -> IndexedImageDataset | None:
        source = sources.get(name)
        if source is None:
            return None
        return IndexedImageDataset(
            source, transform,
            allow_truncated=config.allow_truncated_images, labeled=labeled,
            image_size=config.image_size, bboxes=boxes.get(name),
        )

    fixed_train = dataset("train", transforms.evaluation)
    return ExperimentData(
        calibration_train=fixed_train,
        selected_train=dataset("train", TwoStrongViews(transforms.strong)),
        evaluation_train=fixed_train,
        all_train=dataset(
            "train",
            AllSampleViews(
                transforms.weak, transforms.strong,
                include_structural_view=structural_labels_enabled,
            ),
        ),
        validation=dataset("val", transforms.evaluation, labeled=True),
        test=dataset("test", transforms.evaluation, labeled=True),
        noisy_labels=sources["train"].labels,
        num_classes=config.num_classes,
        class_names=tuple(config.class_names),
    )

"""Ordered bbox metadata compatible with multi_roi's version-1 caches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Iterable
from zipfile import BadZipFile

import numpy as np
from tqdm import tqdm

from setting.config import validate_missing_bbox

if TYPE_CHECKING:
    from setting.config import DataConfig, SplitConfig

BBox = tuple[int, int, int, int]
_MISSING = (-1, -1, -1, -1)
_VERSION = 1


def _fingerprint(paths: Iterable[str]) -> str:
    digest = hashlib.sha256(b"bbox-cache-v1\n")
    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cache_path(config: DataConfig, split: SplitConfig, name: str) -> Path:
    if split.bboxes is not None:
        path = Path(split.bboxes)
        path = path if path.is_absolute() else config.root / path
        if path.is_file():
            return path
        aliases = (f"{name}_bbox_cache_v1.npz", f"{name}_bboxes_cache_v1.npz")
        if path.name in aliases:
            alternate = path.with_name(aliases[1 - aliases.index(path.name)])
            if alternate.is_file():
                return alternate
        return path
    candidates = [
        config.root / f"{name}_{suffix}_cache_v1.npz"
        for suffix in ("bbox", "bboxes")
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError(
            f"{name}: both bbox cache aliases exist; set the split's bboxes filename explicitly."
        )
    return existing[0] if existing else candidates[1]


def _read_cache(path: Path, paths: tuple[str, ...]) -> tuple[BBox | None, ...]:
    try:
        with np.load(path, allow_pickle=False) as cached:
            if cached["version"].item() != _VERSION:
                raise ValueError("unsupported bbox cache version")
            if cached["paths_sha256"].item() != _fingerprint(paths):
                raise ValueError("ordered image paths do not match the bbox cache")
            array = cached["bboxes"]
            if array.shape != (len(paths), 4) or array.dtype.kind not in "iuf":
                raise ValueError("bboxes must be a numeric (number of images, 4) array")
            if not np.isfinite(array).all() or np.any(array != np.floor(array)):
                raise ValueError("bbox coordinates must be finite integers")
            limits = np.iinfo(np.int32)
            if np.any(array < limits.min) or np.any(array > limits.max):
                raise ValueError("bbox coordinates exceed int32 range")
            boxes: list[BBox | None] = []
            for index, row in enumerate(array):
                box = tuple(int(value) for value in row)
                if box == _MISSING:
                    boxes.append(None)
                elif box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f"empty or inverted bbox at index {index}")
                else:
                    boxes.append(box)
            return tuple(boxes)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, BadZipFile) as error:
        raise RuntimeError(f"Invalid bbox cache {path}: {error}") from error


def _annotation_path(config: DataConfig, image_path: str) -> Path | None:
    image = Path(image_path)
    if config.annotation_root is not None:
        try:
            relative = image.relative_to(config.image_root or config.root)
        except ValueError as error:
            raise ValueError(
                f"Cannot map {image} into annotation_root. "
                "Set data.image_root to the common image directory; "
                "annotation_root must mirror its subdirectories."
            ) from error
        image = config.annotation_root / relative
    candidates = (
        image.with_suffix(".json"), image.with_suffix(".JSON"),
        image.with_name(image.name + ".json"),
    )
    return next((path for path in candidates if path.is_file()), None)


def _coordinate(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("bbox coordinates must be numeric integers")
    if not np.isfinite(value) or value != int(value):
        raise ValueError("bbox coordinates must be finite integers")
    return int(value)


def _read_annotation(task: tuple[DataConfig, str]) -> tuple[BBox | None, str, str | None]:
    config, image_path = task
    annotation = None
    try:
        annotation = _annotation_path(config, image_path)
        if annotation is None:
            return None, "missing", "annotation JSON not found"
        document = json.loads(annotation.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("labelingInfo"), list):
            raise ValueError("annotation must contain a labelingInfo list")
        for item in document["labelingInfo"]:
            if not isinstance(item, dict):
                raise ValueError("labelingInfo items must be objects")
            box = item.get("box")
            if not box:
                continue
            if not isinstance(box, dict):
                raise ValueError("box must be an object")
            locations = box.get("location") or []
            if not isinstance(locations, list):
                raise ValueError("box.location must be a list")
            if not locations:
                continue
            location = locations[0]
            if not isinstance(location, dict):
                raise ValueError("box.location items must be objects")
            x, y, width, height = (
                _coordinate(location.get(key)) for key in ("x", "y", "width", "height")
            )
            if width <= 0 or height <= 0:
                continue
            coordinates = (x, y, x + width, y + height)
            limits = np.iinfo(np.int32)
            if any(value < limits.min or value > limits.max for value in coordinates):
                raise ValueError("bbox coordinates exceed int32 range")
            return coordinates, "ok", None
        return None, "missing", f"no positive-size bbox in {annotation}"
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return None, "unreadable_json", f"{annotation or image_path}: {error}"
    except (OSError, ValueError, TypeError, OverflowError) as error:
        return None, "invalid", f"{annotation or image_path}: {error}"


def _audit_missing(
    boxes: tuple[BBox | None, ...], paths: tuple[str, ...], name: str,
    policy: str, report_path: Path | None,
) -> None:
    missing = sum(box is None for box in boxes)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("a", encoding="utf-8") as stream:
            for index, box in enumerate(boxes):
                if box is None:
                    stream.write(json.dumps({
                        "status": "bbox_missing", "split": name,
                        "index": index, "path": paths[index],
                    }, ensure_ascii=False) + "\n")
            stream.write(json.dumps({
                "status": "bbox_summary", "split": name, "checked": len(boxes),
                "with_bbox": len(boxes) - missing, "missing": missing, "policy": policy,
            }) + "\n")
    print(f"[{name}] Bboxes: {len(boxes) - missing}/{len(boxes)}; missing={missing}.")
    if missing and policy == "error":
        examples = list(islice((paths[i] for i, box in enumerate(boxes) if box is None), 5))
        raise RuntimeError(
            f"{name}: {missing} missing bbox(es); crop requires valid annotations.\n"
            + "\n".join(examples)
            + (f"\nFull report: {report_path}" if report_path else "")
        )
    if missing and policy == "drop":
        print(f"[{name}] Excluding {missing} missing bbox(es) from the dataset (missing_bbox='drop').")
    elif missing:
        print(f"WARNING [{name}]: using full images for {missing} missing bbox(es) (missing_bbox='full').")


def _validate_inputs(
    config: DataConfig, paths: tuple[str, ...], raw_paths: tuple[str, ...] | None,
) -> tuple[str, ...]:
    validate_missing_bbox(config.missing_bbox)
    originals = raw_paths if raw_paths is not None else paths
    if len(originals) != len(paths):
        raise ValueError("Raw and resolved bbox paths must have the same length.")
    return originals


def load_bboxes(
    config: DataConfig, split: SplitConfig, name: str, paths: tuple[str, ...], *,
    raw_paths: tuple[str, ...] | None = None, report_path: Path | None = None,
) -> tuple[BBox | None, ...]:
    """Read an aligned cache once at startup; never read JSON in training workers."""
    if not config.crop_bbox:
        return (None,) * len(paths)
    originals = _validate_inputs(config, paths, raw_paths)
    path = _cache_path(config, split, name)
    if not path.is_file():
        raise RuntimeError(f"Missing bbox cache {path}; run.py creates it automatically from JSON.")
    try:
        boxes = _read_cache(path, originals)
    except RuntimeError as error:
        raise RuntimeError(f"{error}; run.py rebuilds stale caches from JSON.") from error
    _audit_missing(boxes, paths, name, config.missing_bbox, report_path)
    return boxes


def prepare_bboxes(
    config: DataConfig, split: SplitConfig, name: str, paths: tuple[str, ...], *,
    raw_paths: tuple[str, ...] | None = None, report_path: Path | None = None,
) -> tuple[BBox | None, ...]:
    """Build metadata or repair only missing entries; preserve original row alignment."""
    if not config.crop_bbox:
        return (None,) * len(paths)
    originals = _validate_inputs(config, paths, raw_paths)
    path = _cache_path(config, split, name)
    cached_boxes = None
    if path.is_file():
        try:
            cached_boxes = _read_cache(path, originals)
        except RuntimeError as error:
            print(f"[{name}] Rebuilding bbox cache: {error}")
    boxes_list = list(cached_boxes) if cached_boxes is not None else [None] * len(paths)
    indices = [index for index, box in enumerate(boxes_list) if box is None]
    if not indices:
        boxes = tuple(boxes_list)
        _audit_missing(boxes, paths, name, config.missing_bbox, report_path)
        return boxes
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
    output = report_path.open("a", encoding="utf-8") if report_path else nullcontext(None)
    workers = max(1, min(config.bbox_workers, len(indices)))
    recovered = []
    invalid, examples = 0, []
    with output as report, ThreadPoolExecutor(max_workers=workers) as executor:
        pending = iter(indices)
        action = "Repair" if cached_boxes is not None else "Cache"
        with tqdm(total=len(indices), desc=f"{action} {name} bboxes", unit="img") as progress:
            while batch := list(islice(pending, workers * 32)):
                tasks = ((config, paths[index]) for index in batch)
                for index, (box, status, error) in zip(batch, executor.map(_read_annotation, tasks)):
                    boxes_list[index] = box
                    if cached_boxes is not None and box is not None:
                        recovered.append(index)
                    if status in {"invalid", "unreadable_json"}:
                        exclude = status == "unreadable_json" and config.missing_bbox == "drop"
                        if not exclude:
                            invalid += 1
                            if len(examples) < 5:
                                examples.append(f"{paths[index]}: {error}")
                        if report is not None:
                            report.write(json.dumps({
                                "status": "bbox_invalid", "split": name, "index": index,
                                "path": paths[index], "error": error,
                                "action": "exclude" if exclude else "error",
                                "reason": "invalid_json" if status == "unreadable_json" else "invalid_bbox",
                            }, ensure_ascii=False) + "\n")
                    progress.update()
    boxes = tuple(boxes_list)
    if invalid:
        raise RuntimeError(
            f"{name}: {invalid} invalid bbox annotation(s); cache was not changed.\n"
            + "\n".join(examples)
            + (f"\nFull report: {report_path}" if report_path else "")
        )
    if cached_boxes is None:
        _audit_missing(boxes, paths, name, config.missing_bbox, report_path)
        _save_cache(path, originals, boxes)
        print(f"[{name}] Saved bbox cache: {path}")
    else:
        if recovered:
            _save_cache(path, originals, boxes)
        if report_path is not None:
            with report_path.open("a", encoding="utf-8") as stream:
                for index in recovered:
                    stream.write(json.dumps({
                        "status": "bbox_recovered", "split": name,
                        "index": index, "path": paths[index], "bbox": boxes[index],
                    }, ensure_ascii=False) + "\n")
                stream.write(json.dumps({
                    "status": "bbox_repair_summary", "split": name,
                    "checked": len(indices), "recovered": len(recovered),
                    "missing": len(indices) - len(recovered),
                }) + "\n")
        print(f"[{name}] Recovered {len(recovered)}/{len(indices)} missing bbox(es) from JSON.")
        _audit_missing(boxes, paths, name, config.missing_bbox, report_path)
    return boxes


def _save_cache(path: Path, originals: tuple[str, ...], boxes: tuple[BBox | None, ...]) -> None:
    """Publish a complete ordered cache only after all annotation reads succeed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(
                stream, bboxes=np.asarray([box or _MISSING for box in boxes], dtype=np.int32).reshape(-1, 4),
                paths_sha256=np.asarray(_fingerprint(originals)), version=np.asarray(_VERSION, dtype=np.int32),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def clamp_bbox(bbox: BBox | None, image_width: int, image_height: int) -> BBox | None:
    """Clip against the original pixels; None means no usable intersection."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    clipped = max(0, x1), max(0, y1), min(image_width, x2), min(image_height, y2)
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None

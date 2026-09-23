"""Image lifetime and model-free dataset auditing."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import audit
from setting.config import CONFIG
from setting.data import IndexedImageDataset, SplitSource
from tests import test_dataset


class DataCleanupTest(unittest.TestCase):
    def test_audit_checks_images_without_training_or_mutating_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = test_dataset.DatasetTest()._fixture(root)
            originals = {p: p.read_bytes() for p in root.iterdir() if p.is_file()}
            config = replace(CONFIG, data=data, runtime=replace(
                CONFIG.runtime, output_root=root / "outputs",
            ))
            with patch("audit.CONFIG", config):
                audit.main()
            records = [json.loads(line) for line in
                       (root / "outputs" / "data_audit.jsonl").read_text().splitlines()]
            self.assertEqual(records[-1]["checked"], 7)
            self.assertFalse(list(root.rglob("*.pt")))
            self.assertFalse((root / "image_cache").exists())
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)

    def test_dataset_releases_full_image_before_augmentation(self):
        source = SplitSource("train", ("example.png",), torch.tensor([0]))
        image = Image.new("RGB", (48, 32))
        views = []

        def transform(resized):
            with self.assertRaises(ValueError):
                image.getpixel((0, 0))
            self.assertEqual(resized.size, (16, 16))
            views.append(resized)
            return torch.zeros(3, 16, 16)

        dataset = IndexedImageDataset(source, transform, allow_truncated=False, image_size=16)
        with patch("setting.data.load_rgb_image", return_value=(image, False)):
            self.assertEqual(dataset[0][1], 0)
        with self.assertRaises(ValueError):
            views[0].getpixel((0, 0))

    def test_dataset_releases_images_on_resize_or_transform_failure(self):
        source = SplitSource("train", ("example.png",), torch.tensor([0]))
        dataset = IndexedImageDataset(source, lambda image: image, allow_truncated=False, image_size=16)
        for operation in ("resize", "transform"):
            with self.subTest(operation=operation):
                image = Image.new("RGB", (32, 32))
                target = image if operation == "resize" else dataset
                with patch("setting.data.load_rgb_image", return_value=(image, False)):
                    with patch.object(target, operation, side_effect=ValueError("bad image")):
                        with self.assertRaisesRegex(RuntimeError, "example.png"):
                            dataset[0]
                with self.assertRaises(ValueError):
                    image.getpixel((0, 0))


if __name__ == "__main__":
    unittest.main()

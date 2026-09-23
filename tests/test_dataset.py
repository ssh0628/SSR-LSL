from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader

from setting.config import AugmentationConfig, DataConfig, SplitConfig
from setting.data import build_experiment_data, inspect_dataset
from setting.image_io import load_rgb_image


class DatasetTest(unittest.TestCase):
    def _fixture(self, root: Path, *, train_size: int = 3) -> DataConfig:
        for name, size in (("train", train_size), ("val", 2), ("test", 2)):
            paths = []
            for index in range(size):
                filename = f"{name}_{index}.jpg"
                image = Image.fromarray(
                    np.random.default_rng(index).integers(0, 256, (48, 64, 3), dtype=np.uint8)
                )
                image.save(root / filename)
                paths.append(filename)
            np.save(root / f"{name}_paths.npy", np.asarray(paths))
            np.save(root / f"{name}_labels.npy", np.arange(size) % 2)
        return replace(
            DataConfig(), root=root, class_names=("A1", "A2"),
            image_size=16, image_check_workers=1, verify_images=True,
            train=SplitConfig("train_paths.npy", "train_labels.npy"),
            validation=SplitConfig("val_paths.npy", "val_labels.npy"),
            test=SplitConfig("test_paths.npy", "test_labels.npy"),
        )

    def _data(self, config: DataConfig, **kwargs):
        return build_experiment_data(
            config,
            structural_labels_enabled=True, augmentation=AugmentationConfig(), **kwargs,
        )

    def test_existing_split_without_metadata_and_deterministic_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            data = self._data(config)
            self.assertEqual(len(data.evaluation_train), 3)
            self.assertTrue(torch.equal(data.noisy_labels, torch.tensor([0, 1, 0])))
            first, index = data.evaluation_train[1]
            second, _ = data.evaluation_train[1]
            self.assertEqual(index, 1)
            self.assertTrue(torch.equal(first, second))
            self.assertEqual(len(data.selected_train[0][0]), 2)
            self.assertEqual(len(data.all_train[0][0]), 3)
            self.assertEqual(data.test[1][1].item(), 1)
            self.assertFalse((root / "split_config.json").exists())

    def test_fractional_nonfinite_and_out_of_range_labels_are_not_cast(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            for labels in ([0.2, 1.0, 0.0], [0, float("nan"), 0], [0, 2, 0]):
                with self.subTest(labels=labels):
                    np.save(root / "train_labels.npy", np.asarray(labels))
                    with self.assertRaisesRegex(ValueError, "labels must"):
                        inspect_dataset(config)

    def test_explicit_filenames_image_root_and_one_based_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            arrays = root / "arrays"
            arrays.mkdir()
            np.save(arrays / "images.npy", np.asarray(["train_0.jpg", "train_1.jpg"]))
            np.save(arrays / "targets.npy", np.asarray([1, 2]))
            config = replace(
                config, root=arrays, image_root=root,
                train=SplitConfig("images.npy", "targets.npy"),
                validation=None, test=None, label_offset=1,
            )
            data = self._data(config)
            self.assertTrue(torch.equal(data.noisy_labels, torch.tensor([0, 1])))
            self.assertIsNone(data.validation)
            self.assertIsNone(data.test)

    def test_normalized_duplicate_paths_and_split_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._fixture(root)
            np.save(root / "train_paths.npy", np.asarray(["train_0.jpg", "./train_0.jpg", "train_2.jpg"]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                inspect_dataset(config)
            self._fixture(root)
            np.save(root / "test_paths.npy", np.asarray(["train_0.jpg", "test_1.jpg"]))
            with self.assertRaisesRegex(ValueError, "disjoint"):
                inspect_dataset(config)

    def test_truncated_jpeg_in_preflight_and_spawned_loader_keeps_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(self._fixture(root), image_check_workers=2)
            path = root / "train_1.jpg"
            path.write_bytes(path.read_bytes()[:-2])
            previous = ImageFile.LOAD_TRUNCATED_IMAGES
            with self.assertRaisesRegex(OSError, "truncated"):
                load_rgb_image(str(path), allow_truncated=False)
            self.assertEqual(ImageFile.LOAD_TRUNCATED_IMAGES, previous)
            report = root / "image_check.jsonl"
            data = self._data(config, report_path=report)
            records = [json.loads(line) for line in report.read_text().splitlines()]
            recovered = [record for record in records if record["status"] == "recovered"]
            self.assertEqual([(r["split"], r["index"]) for r in recovered], [("train", 1)])
            loader = DataLoader(
                data.evaluation_train, batch_size=2, num_workers=2,
                multiprocessing_context="spawn",
            )
            indices = []
            for images, batch_indices in loader:
                self.assertEqual(tuple(images.shape[1:]), (3, 16, 16))
                indices.extend(batch_indices.tolist())
            self.assertEqual(indices, [0, 1, 2])
            self.assertTrue(torch.equal(data.noisy_labels, torch.tensor([0, 1, 0])))

    def test_bad_files_are_aggregated_without_placeholder_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # More than one bounded submission window; failures on both ends.
            config = replace(self._fixture(root, train_size=70), image_check_workers=2)
            (root / "train_0.jpg").write_bytes(b"not an image")
            (root / "test_1.jpg").unlink()
            report = root / "image_check.jsonl"
            with self.assertRaisesRegex(RuntimeError, "2 unreadable image"):
                self._data(config, report_path=report)
            records = [json.loads(line) for line in report.read_text().splitlines()]
            failed = [record for record in records if record["status"] == "failed"]
            self.assertEqual([(r["split"], r["index"]) for r in failed], [("train", 0), ("test", 1)])
            self.assertEqual(records[-1]["checked"], 74)
            self.assertEqual(len(np.load(root / "train_labels.npy")), 70)


if __name__ == "__main__":
    unittest.main()

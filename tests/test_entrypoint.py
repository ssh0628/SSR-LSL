"""Generic image loading and four checkpoint types through run.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.optim import SGD

import run as entrypoint
from log.checkpoint import CHECKPOINT_FILENAMES
from setting.config import CONFIG
from ssr.engine import _build_scheduler
from tests import test_generic_run


class EntrypointTest(unittest.TestCase):
    def test_direct_pipeline_saves_four_checkpoints_without_pixel_cache(self):
        # Tiny backbone only; data, SSR/LSL losses, LW, metrics and I/O run for real.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = test_generic_run.GenericRunTest()
            config = fixture._config(root)
            originals = {path: path.read_bytes() for path in root.iterdir() if path.is_file()}
            with patch("run.CONFIG", config), patch("ssr.engine.build_ssr_networks", return_value=fixture._networks()):
                entrypoint.main()
            self.assertEqual({path.name for path in config.run_dir.glob("*.pt")}, set(CHECKPOINT_FILENAMES))
            for filename in CHECKPOINT_FILENAMES:
                checkpoint = torch.load(config.run_dir / filename, weights_only=True)
                inputs = checkpoint["input_config"]
                self.assertEqual(inputs["image_size"], 32)
                self.assertEqual(inputs["mean"], list(config.data.mean))
                self.assertEqual(inputs["std"], list(config.data.std))
                self.assertEqual(inputs["label_offset"], 0)
                self.assertNotIn("multi_roi", inputs)
            rows = [json.loads(line) for line in (config.run_dir / "checkpoint_results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["test"]["sample_count"] == 4 for row in rows))
            self.assertEqual(sum(not row["test_reused_same_epoch"] for row in rows), 1)
            self.assertFalse(list(root.glob("*.npz")))
            self.assertFalse((root / "image_cache").exists())
            self.assertEqual(set(root.rglob("*.npy")), {path for path in originals if path.suffix == ".npy"})
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content, str(path))

    def test_entrypoint_uses_config_once(self):
        with patch("run.run") as train:
            entrypoint.main()
        train.assert_called_once_with(CONFIG)

    def test_invalid_config_stops_before_data_or_model_creation(self):
        config = replace(CONFIG, training=replace(CONFIG.training, epochs=0))
        with patch("run.CONFIG", config), patch("ssr.engine.build_experiment_data") as data, patch("ssr.engine.build_ssr_networks") as model:
            with self.assertRaises(ValueError):
                entrypoint.main()
        data.assert_not_called()
        model.assert_not_called()

    def test_default_300_epoch_cosine_schedule_reaches_configured_floor(self):
        self.assertEqual(CONFIG.training.epochs, 300)
        optimizer = SGD(nn.Linear(1, 1).parameters(), lr=0.1)
        scheduler = _build_scheduler(optimizer, CONFIG)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)
        for _ in range(150):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1 * (1 + CONFIG.training.scheduler_eta_min_ratio) / 2)
        for _ in range(150):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1 * CONFIG.training.scheduler_eta_min_ratio)


if __name__ == "__main__":
    unittest.main()

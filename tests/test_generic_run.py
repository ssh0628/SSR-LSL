from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from log.checkpoint import CHECKPOINT_FILENAMES
from setting.config import CONFIG, ModelConfig
from setting.model import SSRNetworks, build_ssr_networks
from ssr.engine import _build_optimizer, _prepare_config, run
from ssr.trainer import train_epoch


class _Views(Dataset):
    def __init__(self, images: torch.Tensor, view_count: int):
        self.images = images
        self.view_count = view_count

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return [self.images[index] for _ in range(self.view_count)], index


class GenericRunTest(unittest.TestCase):
    def _config(self, root: Path):
        for split in ("train", "val", "test"):
            paths = []
            for index in range(4):
                path = root / f"{split}_{index}.jpg"
                Image.new("RGB", (32, 32), (index * 50, 30, 60)).save(path)
                paths.append(str(path))
            np.save(root / f"{split}_path.npy", np.asarray(paths))
            np.save(root / f"{split}_labels.npy", np.arange(4) % 2)
        return replace(
            CONFIG,
            data=replace(CONFIG.data, root=root, class_names=("A1", "A2"), image_size=32, image_check_workers=1),
            model=replace(CONFIG.model, pretrained=False, calibrate_initial_batch_norm=False),
            ssr=replace(CONFIG.ssr, neighbors=1),
            structural_labels=replace(CONFIG.structural_labels, neighbors=1),
            label_wave=replace(CONFIG.label_wave, moving_average_window=1),
            training=replace(CONFIG.training, epochs=1, batch_size=2, num_workers=0, eval_batch_size=2, eval_num_workers=0),
            runtime=replace(CONFIG.runtime, device="cpu", output_root=root / "outputs", run_id="smoke"),
        )

    @staticmethod
    def _networks():
        return SSRNetworks(
            encoder=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 8)),
            classifier=nn.Linear(8, 2),
            projector=nn.Sequential(nn.Linear(8, 4), nn.BatchNorm1d(4)),
            predictor=nn.Linear(4, 4),
        )

    def test_generic_loop_optional_splits_and_toggles(self):
        for val, test, lsl, lw in ((True, True, True, True), (False, True, False, True), (False, False, False, False)):
            with self.subTest(val=val, test=test, lsl=lsl, lw=lw), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = self._config(root)
                config = replace(
                    config,
                    data=replace(config.data, validation=config.data.validation if val else None, test=config.data.test if test else None),
                    structural_labels=replace(config.structural_labels, enabled=lsl),
                    label_wave=replace(config.label_wave, enabled=lw),
                )
                networks = self._networks()
                def evaluate_scores(*args, description, **kwargs):
                    score = 0.5 if description == "Validation" else 1.0
                    return {"accuracy": score, "balanced_accuracy": score, "macro_f1": score}

                with patch("ssr.engine.build_ssr_networks", return_value=networks), patch("ssr.engine.evaluate_classification", side_effect=evaluate_scores) as evaluate:
                    result = run(config)
                self.assertEqual(result, 0.5 if val else None)
                self.assertEqual([call.kwargs["description"] for call in evaluate.call_args_list], (["Validation"] if val else []) + (["Test epoch 1"] if test else []))
                for filename in CHECKPOINT_FILENAMES:
                    expected = val if filename.startswith("best_") else lw if filename == "label_wave.pt" else True
                    self.assertEqual((config.run_dir / filename).exists(), expected)
                checkpoint = torch.load(config.run_dir / "last.pt", weights_only=True)
                self.assertEqual(checkpoint["class_names"], ["A1", "A2"])
                self.assertEqual(checkpoint["cur_epoch"], 0)
                self.assertNotIn("test_accuracy", checkpoint)
                results = [json.loads(line) for line in (config.run_dir / "checkpoint_results.jsonl").read_text().splitlines()]
                self.assertEqual(len(results), 4)
                for row in results:
                    if row["status"] != "not_created":
                        self.assertEqual(row["test"] is not None, test)
                if lw:
                    selected = torch.load(config.run_dir / "label_wave.pt", weights_only=True)
                    self.assertEqual(selected["cur_epoch"], 0)
                values = json.loads((config.run_dir / "metrics.jsonl").read_text())
                self.assertEqual(values["structural_loss"] is not None, lsl)
                self.assertIn("label_changes", values)
                self.assertGreater(values["training_seconds"], 0)
                self.assertGreater(values["epoch_seconds"], 0)
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    run(config)

    def test_last_checkpoint_survives_final_test_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            config = replace(config, data=replace(config.data, validation=None))
            with patch("ssr.engine.build_ssr_networks", return_value=self._networks()), patch("ssr.engine.evaluate_classification", side_effect=RuntimeError("test decoding failed")):
                with self.assertRaisesRegex(RuntimeError, "test decoding failed"):
                    run(config)
            state = torch.load(config.run_dir / "last.pt", weights_only=True)
            self.assertEqual(state["cur_epoch"], 0)
            self.assertTrue(state["optimizer"]["state"])
            self.assertNotIn("test_accuracy", state)

    def test_auto_run_ids_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(CONFIG, runtime=replace(CONFIG.runtime, output_root=Path(directory)))
            first, second = _prepare_config(config), _prepare_config(config)
            self.assertNotEqual(first.run_dir, second.run_dir)
            self.assertEqual(first.run_name, second.run_name)

    def test_real_convnext_ssr_lsl_backward(self):
        torch.manual_seed(0)
        config = replace(CONFIG, data=replace(CONFIG.data, class_names=("A1", "A2")))
        networks = build_ssr_networks(ModelConfig(pretrained=False), torch.device("cpu"), 2)
        images = torch.randn(2, 3, 32, 32)
        selected = DataLoader(_Views(images, 2), batch_size=2)
        all_samples = DataLoader(_Views(images, 3), batch_size=2)
        labels = torch.tensor([0, 1])
        before = next(networks.encoder.parameters()).detach().clone()
        losses = train_epoch(
            selected, all_samples, labels, torch.eye(2), networks,
            _build_optimizer(networks, config), config, torch.device("cpu"), epoch=0,
        )
        self.assertTrue(np.isfinite([losses.supervised_loss, losses.feature_consistency_loss, losses.structural_loss]).all())
        self.assertFalse(torch.equal(before, next(networks.encoder.parameters())))
        networks.eval()
        with torch.no_grad():
            self.assertEqual(networks.classifier(networks.encoder(images)).shape, (2, 2))


if __name__ == "__main__":
    unittest.main()

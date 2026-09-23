"""Runtime failures must preserve the trained epoch and selection metadata."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, TensorDataset

from label_wave import LabelWaveRun
from log.checkpoint import CheckpointManager
from setting.config import CONFIG
from setting.data import ExperimentData
from setting.model import SSRNetworks
from ssr.engine import run


class _TrainingViews(Dataset):
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return [self.images[index], self.images[index]], index


class RuntimeRegressionTest(unittest.TestCase):
    def test_completed_epoch_survives_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                CONFIG,
                data=replace(CONFIG.data, class_names=("A1", "A2"), image_size=4),
                model=replace(CONFIG.model, calibrate_initial_batch_norm=False),
                ssr=replace(CONFIG.ssr, neighbors=1, selection_threshold=0.0),
                structural_labels=replace(CONFIG.structural_labels, enabled=False),
                label_wave=replace(CONFIG.label_wave, enabled=False),
                training=replace(CONFIG.training, epochs=1, batch_size=2, num_workers=0, eval_batch_size=2, eval_num_workers=0),
                runtime=replace(CONFIG.runtime, device="cpu", output_root=Path(directory), run_id="validation-error"),
            )
            images = torch.randn(4, 3, 4, 4)
            labels = torch.tensor([0, 1, 0, 1])
            evaluation = TensorDataset(images, torch.arange(4))
            views = _TrainingViews(images)
            data = ExperimentData(
                calibration_train=evaluation,
                selected_train=views,
                evaluation_train=evaluation,
                all_train=views,
                validation=TensorDataset(images, labels),
                test=None,
                noisy_labels=labels,
                num_classes=2,
                class_names=("A1", "A2"),
            )
            networks = SSRNetworks(
                encoder=nn.Sequential(nn.Flatten(), nn.Linear(48, 8)),
                classifier=nn.Linear(8, 2),
                projector=nn.Linear(8, 4),
                predictor=nn.Linear(4, 4),
            )
            initial = networks.classifier.weight.detach().clone()
            with (
                patch("ssr.engine.build_experiment_data", return_value=data),
                patch("ssr.engine.build_ssr_networks", return_value=networks),
                patch("ssr.engine.evaluate_classification", side_effect=RuntimeError("validation decode failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "validation decode failed"):
                    run(config)

            state = torch.load(config.run_dir / "last.pt", weights_only=True)
            self.assertEqual(state["cur_epoch"], 0)
            self.assertIsNone(state["metrics"]["validation_accuracy"])
            self.assertTrue(state["optimizer"]["state"])
            self.assertFalse(torch.equal(initial, state["classifier"]["weight"]))
            torch.testing.assert_close(state["classifier"]["weight"], networks.classifier.weight)
            self.assertFalse((config.run_dir / "best_balanced_accuracy.pt").exists())
            self.assertFalse((config.run_dir / "best_macro_f1.pt").exists())
            self.assertEqual({path.name for path in config.run_dir.glob("*.pt")}, {"last.pt"})
            self.assertEqual(state["selection"]["criterion"], "last_epoch")

    def test_label_wave_keeps_selected_weights_and_completed_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                CONFIG,
                data=replace(CONFIG.data, class_names=("A1", "A2")),
                label_wave=replace(CONFIG.label_wave, moving_average_window=1, patience=1),
                runtime=replace(CONFIG.runtime, device="cpu", output_root=Path(directory)),
            )
            modules = [nn.Linear(2, 2) for _ in range(4)]
            networks = SSRNetworks(*modules)
            optimizer = SGD([parameter for module in modules for parameter in module.parameters()], lr=0.1)
            checkpoints = CheckpointManager(config.run_dir, networks, optimizer, config)
            with LabelWaveRun(config.label_wave, config.run_dir, checkpoints) as wave:
                wave.observe(torch.tensor([0, 0, 0, 0]), completed_epochs=0)
                with torch.no_grad():
                    networks.classifier.weight.fill_(1.0)
                wave.observe(torch.tensor([1, 0, 0, 0]), completed_epochs=1, validation_accuracy=0.5)
                with torch.no_grad():
                    networks.classifier.weight.fill_(2.0)
                decision = wave.observe(torch.tensor([1, 1, 1, 0]), completed_epochs=2, validation_accuracy=1.0)
                self.assertTrue(decision.should_stop)
                wave.observe(torch.tensor([1, 1, 1, 0]), completed_epochs=3, validation_accuracy=1.0)

            state = torch.load(config.run_dir / "label_wave.pt", weights_only=True)
            self.assertEqual(state["cur_epoch"], 0)
            self.assertEqual(state["label_wave"]["selected_epoch"], 1)
            self.assertEqual(state["metrics"]["validation_accuracy"], 0.5)
            self.assertTrue(torch.equal(state["classifier"]["weight"], torch.ones(2, 2)))
            self.assertNotIn("test_accuracy", state)
            self.assertEqual({path.name for path in config.run_dir.glob("*.pt")}, {"label_wave.pt"})
            self.assertEqual(state["selection"]["criterion"], "prediction_change")


if __name__ == "__main__":
    unittest.main()

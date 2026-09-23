"""Four checkpoints preserve independent best, last, and Label Wave selection."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, TensorDataset

from log.checkpoint import CHECKPOINT_FILENAMES, CheckpointManager
from setting.config import CONFIG
from setting.data import ExperimentData
from setting.model import SSRNetworks
from ssr.engine import run
from ssr.evaluation import EpochSupervision
from ssr.selection import SelectionResult
from ssr.trainer import TrainingLosses


class _Views(Dataset):
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        return [self.images[index], self.images[index]], index


def _networks() -> SSRNetworks:
    return SSRNetworks(
        encoder=nn.Sequential(nn.Flatten(), nn.Linear(48, 8)),
        classifier=nn.Linear(8, 2),
        projector=nn.Linear(8, 4),
        predictor=nn.Linear(4, 4),
    )


class CheckpointSelectionTest(unittest.TestCase):
    def test_four_names_select_independent_validation_bests_and_freeze_label_wave(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                CONFIG,
                data=replace(CONFIG.data, class_names=("A1", "A2"), image_size=4),
                model=replace(CONFIG.model, calibrate_initial_batch_norm=False),
                ssr=replace(CONFIG.ssr, neighbors=1),
                structural_labels=replace(CONFIG.structural_labels, enabled=False),
                label_wave=replace(CONFIG.label_wave, enabled=True, moving_average_window=1, patience=1, stop_training=False),
                training=replace(CONFIG.training, epochs=4, batch_size=2, num_workers=0, eval_batch_size=2, eval_num_workers=0),
                runtime=replace(CONFIG.runtime, device="cpu", output_root=Path(directory), run_id="four-checkpoints"),
            )
            images = torch.randn(4, 3, 4, 4)
            labels = torch.tensor([0, 1, 0, 1])
            evaluation = TensorDataset(images, torch.arange(4))
            data = ExperimentData(
                calibration_train=evaluation,
                selected_train=_Views(images),
                evaluation_train=evaluation,
                all_train=_Views(images),
                validation=TensorDataset(images, labels),
                test=TensorDataset(images, labels),
                noisy_labels=labels,
                num_classes=2,
                class_names=("A1", "A2"),
            )
            selection = SelectionResult(
                selected_indices=torch.arange(4),
                rejected_indices=torch.empty(0, dtype=torch.long),
                modified_labels=labels,
                relabelled_indices=torch.arange(4),
                changed_indices=torch.empty(0, dtype=torch.long),
                confidences=torch.full((4,), 0.95),
                consistency=torch.ones(4),
            )
            # PC: 1 -> 2 (patience reached) -> 0 -> 0. The later minima
            # must not replace the first, already finalized LW decision.
            predictions = [
                torch.tensor(values)
                for values in ((0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 1, 0), (1, 1, 1, 0))
            ]
            supervision = [EpochSupervision(selection, None, values) for values in predictions]
            validation = {
                1: {"accuracy": 0.50, "balanced_accuracy": 0.60, "macro_f1": 0.40},
                2: {"accuracy": 0.55, "balanced_accuracy": 0.90, "macro_f1": 0.30},
                3: {"accuracy": 0.60, "balanced_accuracy": 0.70, "macro_f1": 0.95},
                4: {"accuracy": 0.70, "balanced_accuracy": 0.80, "macro_f1": 0.80},
            }
            networks = _networks()
            events = []

            def train(*args):
                optimizer, epoch = args[5], args[8]
                optimizer.step()
                with torch.no_grad():
                    for module in networks.all_modules():
                        for parameter in module.parameters():
                            parameter.fill_(epoch + 1)
                events.append(("train", epoch + 1))
                return TrainingLosses(1.0, 0.2, None)

            def evaluate(*args, description, **kwargs):
                epoch = round(float(networks.classifier.weight[0, 0].detach()))
                events.append((description, epoch))
                if description == "Validation":
                    return dict(validation[epoch])
                # Highest held-out scores at epoch 1 must not affect the
                # two best selections, which were already fixed by validation.
                score = 1.0 / epoch
                return {"accuracy": score, "balanced_accuracy": score, "macro_f1": score}

            with (
                patch("ssr.engine.build_experiment_data", return_value=data),
                patch("ssr.engine.build_ssr_networks", return_value=networks),
                patch("ssr.engine.evaluate_epoch", side_effect=supervision),
                patch("ssr.engine.train_epoch", side_effect=train),
                patch("ssr.engine.predict_training_labels", return_value=predictions[-1]),
                patch("ssr.engine.evaluate_classification", side_effect=evaluate),
            ):
                self.assertEqual(run(config), 0.70)

            self.assertEqual({path.name for path in config.run_dir.glob("*.pt")}, set(CHECKPOINT_FILENAMES))
            expected_epochs = {
                "best_balanced_accuracy.pt": 2,
                "best_macro_f1.pt": 3,
                "last.pt": 4,
                "label_wave.pt": 1,
            }
            for filename, epoch in expected_epochs.items():
                state = torch.load(config.run_dir / filename, weights_only=True)
                self.assertEqual(state["completed_epochs"], epoch)
                self.assertEqual(state["cur_epoch"], epoch - 1)
                self.assertEqual(state["metrics"]["validation"], validation[epoch])
                torch.testing.assert_close(state["classifier"]["weight"], torch.full_like(state["classifier"]["weight"], epoch))
                self.assertNotIn("test_accuracy", state)
                if filename == "label_wave.pt":
                    self.assertEqual(state["selection"]["criterion"], "prediction_change")
                    self.assertEqual(state["label_wave"]["selected_epoch"], 1)
                elif filename.startswith("best_"):
                    self.assertEqual(state["selection"]["split"], "validation")
            self.assertEqual(events[:8], [(kind, epoch) for epoch in range(1, 5) for kind in ("train", "Validation")])
            self.assertEqual(events[8:], [(f"Test epoch {epoch}", epoch) for epoch in (2, 3, 4, 1)])
            rows = [json.loads(line) for line in (config.run_dir / "checkpoint_results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(sum(not row["test_reused_same_epoch"] for row in rows), 4)
            for row in rows:
                self.assertEqual(row["completed_epochs"], expected_epochs[row["filename"]])
                self.assertEqual(row["test"]["balanced_accuracy"], 1.0 / row["completed_epochs"])
            metrics = [json.loads(line) for line in (config.run_dir / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(metrics[-1]["best_epochs"], {"balanced_accuracy": 2, "macro_f1": 3})
            self.assertEqual(metrics[-1]["validation_macro_f1"], 0.80)
            self.assertIn("train_observed_label_metrics_before_update", metrics[-1])

    def test_replacing_checkpoint_does_not_modify_existing_hard_link_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            networks = _networks()
            optimizer = SGD(networks.classifier.parameters(), lr=0.1)
            checkpoints = CheckpointManager(run_dir, networks, optimizer, CONFIG)
            with torch.no_grad():
                networks.classifier.weight.fill_(1)
            checkpoints.save("last.pt", 0)
            snapshot = run_dir / "previous_epoch.pt"
            os.link(run_dir / "last.pt", snapshot)
            with torch.no_grad():
                networks.classifier.weight.fill_(2)
            checkpoints.save("last.pt", 1)
            old_state = torch.load(snapshot, weights_only=True)
            self.assertEqual(old_state["completed_epochs"], 1)
            self.assertTrue(old_state["classifier"]["weight"].eq(1).all())
            state = torch.load(run_dir / "last.pt", weights_only=True)
            self.assertEqual(state["completed_epochs"], 2)
            self.assertTrue(state["classifier"]["weight"].eq(2).all())
            self.assertEqual(list(run_dir.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

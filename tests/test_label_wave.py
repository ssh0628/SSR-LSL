from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn
from torch.optim import SGD

from label_wave import LabelWaveRun, LabelWaveTracker
from log.checkpoint import CheckpointManager
from setting.config import CONFIG, LabelWaveConfig, RuntimeConfig, TrainingConfig
from setting.model import SSRNetworks


class LabelWaveTrackerTest(unittest.TestCase):
    @staticmethod
    def _change(previous: torch.Tensor, count: int) -> torch.Tensor:
        current = previous.clone()
        current[:count] = 1 - current[:count]
        return current

    def test_selects_first_local_minimum_after_patience(self) -> None:
        tracker = LabelWaveTracker(window=3, patience=2)
        predictions = torch.zeros(10, dtype=torch.long)
        baseline = tracker.update(predictions, epoch=0)
        self.assertIsNone(baseline.prediction_changes)

        observations = []
        for epoch, changes in enumerate([9, 6, 3, 8, 9, 10], start=1):
            predictions = self._change(predictions, changes)
            observations.append(tracker.update(predictions, epoch=epoch))

        self.assertTrue(observations[2].is_candidate)
        self.assertEqual(observations[2].moving_average, 6.0)
        self.assertTrue(observations[3].is_candidate)
        self.assertAlmostEqual(observations[3].moving_average, 17.0 / 3.0)
        self.assertFalse(observations[4].should_stop)
        self.assertTrue(observations[5].should_stop)
        self.assertEqual(tracker.selected_epoch, 4)

    def test_monitoring_freezes_selection_after_stop_decision(self) -> None:
        tracker = LabelWaveTracker(window=1, patience=1)
        predictions = torch.zeros(4, dtype=torch.long)
        tracker.update(predictions, epoch=0)
        predictions = self._change(predictions, 2)
        tracker.update(predictions, epoch=1)
        predictions = self._change(predictions, 3)
        decision = tracker.update(predictions, epoch=2)
        self.assertTrue(decision.should_stop)

        predictions = self._change(predictions, 1)
        later = tracker.update(predictions, epoch=3)
        self.assertFalse(later.is_candidate)
        self.assertFalse(later.should_stop)
        self.assertEqual(later.selected_epoch, 1)

    def test_rejects_changed_vector_size_and_duplicate_epoch(self) -> None:
        tracker = LabelWaveTracker(window=1, patience=1)
        tracker.update(torch.zeros(3), epoch=0)
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            tracker.update(torch.zeros(3), epoch=0)
        with self.assertRaisesRegex(ValueError, "vector size"):
            tracker.update(torch.zeros(4), epoch=1)

    def test_candidate_checkpoint_uses_completed_epoch_alignment(self) -> None:
        with TemporaryDirectory() as directory:
            config = replace(
                CONFIG,
                label_wave=LabelWaveConfig(
                    enabled=True,
                    stop_training=False,
                    moving_average_window=1,
                    patience=1,
                ),
                runtime=RuntimeConfig(
                    device="cpu",
                    output_root=Path(directory),
                ),
            )
            modules = [nn.Linear(2, 2) for _ in range(4)]
            networks = SSRNetworks(*modules)
            optimizer = SGD(
                [parameter for module in modules for parameter in module.parameters()],
                lr=0.1,
            )
            log_path = config.run_dir / "label_wave.jsonl"
            checkpoints = CheckpointManager(
                config.run_dir,
                networks,
                optimizer,
                config,
            )
            with LabelWaveRun(config.label_wave, config.run_dir, checkpoints) as run:
                run.observe(
                    torch.tensor([0, 0, 0]),
                    completed_epochs=0,
                    test_accuracy=None,
                )
                run.observe(
                    torch.tensor([1, 0, 1]),
                    completed_epochs=1,
                    test_accuracy=0.8,
                )

            checkpoint = torch.load(
                config.run_dir / "label_wave.pt",
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(checkpoint["cur_epoch"], 0)
            self.assertEqual(checkpoint["label_wave"]["selected_epoch"], 1)
            self.assertEqual(checkpoint["test_accuracy"], 0.8)
            self.assertEqual(len(log_path.read_text(encoding="utf-8").splitlines()), 2)

    def test_label_wave_config_rejects_ambiguous_or_impossible_modes(self) -> None:
        disabled_stop = replace(
            CONFIG,
            label_wave=LabelWaveConfig(enabled=False, stop_training=True),
        )
        with self.assertRaisesRegex(ValueError, "Enable label_wave"):
            disabled_stop.validate()

        oversized_window = replace(
            CONFIG,
            label_wave=LabelWaveConfig(
                enabled=True,
                moving_average_window=4,
            ),
            training=TrainingConfig(epochs=3),
        )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            oversized_window.validate()


if __name__ == "__main__":
    unittest.main()

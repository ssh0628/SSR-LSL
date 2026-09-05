"""Run-level Label Wave logging and checkpoint selection."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from torch import Tensor

from cifar.label_wave.tracker import LabelWaveObservation, LabelWaveTracker
from cifar.log.checkpoint import CheckpointManager
from cifar.log.common import JsonlWriter
from cifar.setting.config import LabelWaveConfig


class LabelWaveRun:
    """Own the tracker, JSONL log, and selected checkpoint for one run."""

    def __init__(
        self,
        config: LabelWaveConfig,
        run_dir: Path,
        checkpoints: CheckpointManager,
    ) -> None:
        if not config.enabled:
            raise ValueError("LabelWaveRun requires label_wave.enabled=True.")
        self.tracker = LabelWaveTracker(
            window=config.moving_average_window,
            patience=config.patience,
        )
        self._log_path = run_dir / "label_wave.jsonl"
        self._checkpoints = checkpoints
        self._writer: JsonlWriter | None = None

    def __enter__(self) -> "LabelWaveRun":
        self._writer = JsonlWriter(self._log_path)
        return self

    def __exit__(self, *_: object) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def observe(
        self,
        predictions: Tensor,
        *,
        completed_epochs: int,
        test_accuracy: float | None = None,
        reference_accuracy: float | None = None,
        reference_metric: str = "test_accuracy",
    ) -> LabelWaveObservation:
        if self._writer is None:
            raise RuntimeError("LabelWaveRun must be used as a context manager.")

        if test_accuracy is not None and reference_accuracy is not None:
            raise ValueError("Pass test_accuracy or reference_accuracy, not both.")
        accuracy = (
            test_accuracy if reference_accuracy is None else reference_accuracy
        )
        observation = self.tracker.update(predictions, epoch=completed_epochs)
        values = asdict(observation)
        values[reference_metric] = accuracy
        self._writer.write(values)

        if observation.is_candidate:
            self._checkpoints.save(
                "label_wave.pt",
                completed_epochs - 1,
                label_wave=asdict(observation),
                test_accuracy=(accuracy if reference_metric == "test_accuracy" else None),
                metrics={reference_metric: accuracy},
            )
            print(
                f"label_wave candidate_epoch={completed_epochs} "
                f"pc_ma={observation.moving_average:.2f}"
            )
        if observation.should_stop:
            print(
                f"label_wave selected_epoch={observation.selected_epoch} "
                f"stop_epoch={completed_epochs}"
            )
        return observation

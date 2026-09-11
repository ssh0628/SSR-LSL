"""Run-level Label Wave logging and checkpoint selection."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from torch import Tensor

from label_wave.tracker import LabelWaveObservation, LabelWaveTracker
from log.checkpoint import CheckpointManager
from log.common import JsonlWriter
from setting.config import LabelWaveConfig


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
        validation_accuracy: float | None = None,
        validation_metrics: Mapping[str, Any] | None = None,
    ) -> LabelWaveObservation:
        if self._writer is None:
            raise RuntimeError("LabelWaveRun must be used as a context manager.")

        observation = self.tracker.update(predictions, epoch=completed_epochs)
        values = asdict(observation)
        values["validation_accuracy"] = validation_accuracy
        values["validation"] = dict(validation_metrics) if validation_metrics is not None else None
        self._writer.write(values)

        if observation.is_candidate:
            self._checkpoints.save(
                "label_wave.pt",
                completed_epochs - 1,
                label_wave=asdict(observation),
                metrics={
                    "validation_accuracy": validation_accuracy,
                    "validation": dict(validation_metrics) if validation_metrics is not None else None,
                },
                selection={"method": "label_wave", "criterion": "prediction_change"},
            )
        if observation.should_stop:
            print(
                f"label_wave selected_epoch={observation.selected_epoch} "
                f"stop_epoch={completed_epochs}"
            )
        return observation

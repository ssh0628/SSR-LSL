"""Training checkpoint construction and persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from torch.optim import Optimizer

from cifar.log.common import atomic_torch_save
from cifar.setting.config import ExperimentConfig
from cifar.setting.model import SSRNetworks


class CheckpointManager:
    """Persist consistently structured checkpoints for one training run."""

    def __init__(
        self,
        run_dir: Path,
        networks: SSRNetworks,
        optimizer: Optimizer,
        config: ExperimentConfig,
    ) -> None:
        self.run_dir = run_dir
        self.networks = networks
        self.optimizer = optimizer
        self.config = config

    def save(
        self,
        filename: str,
        epoch: int,
        *,
        test_accuracy: float | None = None,
        metrics: Mapping[str, float | None] | None = None,
        label_wave: Mapping[str, Any] | None = None,
    ) -> None:
        if Path(filename).name != filename:
            raise ValueError("checkpoint filename must not contain a directory.")
        state: dict[str, Any] = {
            "cur_epoch": epoch,
            "model_name": self.config.model.name,
            "structural_labels_enabled": self.config.structural_labels.enabled,
            "classifier": self.networks.classifier.state_dict(),
            "encoder": self.networks.encoder.state_dict(),
            "proj_head": self.networks.projector.state_dict(),
            "pred_head": self.networks.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if label_wave is not None:
            state["label_wave"] = dict(label_wave)
        if test_accuracy is not None:
            state["test_accuracy"] = test_accuracy
        if metrics is not None:
            state["metrics"] = dict(metrics)
        atomic_torch_save(state, self.run_dir / filename)

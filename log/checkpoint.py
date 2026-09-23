"""Training checkpoint construction and persistence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from torch.optim import Optimizer

from log.common import atomic_torch_save
from setting.config import ExperimentConfig
from setting.model import SSRNetworks


SELECTION_METRICS = ("balanced_accuracy", "macro_f1")
CHECKPOINT_FILENAMES = (
    "best_balanced_accuracy.pt",
    "best_macro_f1.pt",
    "last.pt",
    "label_wave.pt",
)


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
        self.records: dict[str, dict[str, Any]] = {}

    def save(
        self,
        filename: str,
        epoch: int,
        *,
        metrics: Mapping[str, Any] | None = None,
        label_wave: Mapping[str, Any] | None = None,
        selection: Mapping[str, Any] | None = None,
    ) -> None:
        if Path(filename).name != filename:
            raise ValueError("checkpoint filename must not contain a directory.")
        state: dict[str, Any] = {
            "cur_epoch": epoch,
            "completed_epochs": epoch + 1,
            "metric_units": "rates in [0, 1]; loss/entropy in nats; MCC/kappa in [-1, 1]",
            "model_name": self.config.model.name,
            "model_config": asdict(self.config.model),
            "class_names": list(self.config.data.class_names),
            "num_classes": self.config.data.num_classes,
            "input_config": {
                "image_size": self.config.data.image_size,
                "mean": list(self.config.data.mean),
                "std": list(self.config.data.std),
                "label_offset": self.config.data.label_offset,
            },
            "structural_labels_enabled": self.config.structural_labels.enabled,
            "classifier": self.networks.classifier.state_dict(),
            "encoder": self.networks.encoder.state_dict(),
            "proj_head": self.networks.projector.state_dict(),
            "pred_head": self.networks.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if label_wave is not None:
            state["label_wave"] = dict(label_wave)
        if metrics is not None:
            state["metrics"] = dict(metrics)
        if selection is not None:
            state["selection"] = dict(selection)
        atomic_torch_save(state, self.run_dir / filename)
        self.records[filename] = {
            "filename": filename,
            "epoch": epoch,
            "completed_epochs": epoch + 1,
            "validation": dict(metrics or {}).get("validation"),
            "selection": dict(selection or {}),
            "label_wave": dict(label_wave) if label_wave is not None else None,
        }

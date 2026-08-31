"""Single source of truth for the CIFAR-10 SSR experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NoiseKind = Literal["symmetric", "asymmetric", "idn"]


@dataclass(frozen=True, slots=True)
class DataConfig:
    """CIFAR-10 and synthetic-noise settings."""

    root: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    download: bool = True
    noise_kind: NoiseKind = "idn"
    noise_rate: float = 0.5
    noise_seed: int = 0
    idn_flip_rate_std: float = 0.1

    @property
    def noise_file(self) -> Path:
        rate = f"{self.noise_rate:.4f}".rstrip("0").rstrip(".")
        standard_deviation = (
            f"_std{self.idn_flip_rate_std:g}" if self.noise_kind == "idn" else ""
        )
        return (
            self.root
            / "noise"
            / (
                f"cifar10_{self.noise_kind}_{rate}{standard_deviation}"
                f"_seed{self.noise_seed}.pt"
            )
        )


@dataclass(frozen=True, slots=True)
class SSRConfig:
    """Algorithm settings from the official CIFAR SSR implementation."""

    relabel_threshold: float = 0.55
    selection_threshold: float = 1.0
    neighbors: int = 200
    knn_chunks: int = 10
    feature_consistency_weight: float = 1.0
    mixup_alpha: float = 4.0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization settings used for CIFAR in the SSR paper."""

    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 0.02
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler_eta_min_ratio: float = 1.0 / 50.0
    num_workers: int = 4
    seed: int = 0


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Execution and artifact settings; these do not alter SSR itself."""

    device: str = "auto"
    output_root: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    ssr: SSRConfig = field(default_factory=SSRConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def run_name(self) -> str:
        rate = f"{self.data.noise_rate:.4f}".rstrip("0").rstrip(".")
        return (
            f"cifar10_{self.data.noise_kind}{rate}"
            f"_nseed{self.data.noise_seed}"
            f"_tr{self.ssr.relabel_threshold:g}"
            f"_ts{self.ssr.selection_threshold:g}"
            f"_seed{self.training.seed}"
        )

    @property
    def run_dir(self) -> Path:
        return self.runtime.output_root / self.run_name

    def validate(self) -> None:
        if self.data.noise_kind not in {"symmetric", "asymmetric", "idn"}:
            raise ValueError("data.noise_kind must be symmetric, asymmetric, or idn.")
        if not 0.0 <= self.data.noise_rate < 1.0:
            raise ValueError("data.noise_rate must be in [0, 1).")
        if self.data.idn_flip_rate_std <= 0.0:
            raise ValueError("data.idn_flip_rate_std must be positive.")
        if not 0.0 <= self.ssr.relabel_threshold <= 1.0:
            raise ValueError("ssr.relabel_threshold must be in [0, 1].")
        if not 0.0 <= self.ssr.selection_threshold <= 1.0:
            raise ValueError("ssr.selection_threshold must be in [0, 1].")
        if not 1 <= self.ssr.neighbors <= 50_000:
            raise ValueError("ssr.neighbors must be in [1, 50000].")
        if self.ssr.knn_chunks < 1:
            raise ValueError("ssr.knn_chunks must be positive.")
        if self.ssr.mixup_alpha <= 0.0:
            raise ValueError("ssr.mixup_alpha must be positive.")
        if self.ssr.feature_consistency_weight < 0.0:
            raise ValueError("ssr.feature_consistency_weight must not be negative.")
        if self.training.epochs < 1 or self.training.batch_size < 1:
            raise ValueError("training epochs and batch_size must be positive.")
        if self.training.learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive.")
        if self.training.scheduler_eta_min_ratio < 0.0:
            raise ValueError("training.scheduler_eta_min_ratio must not be negative.")
        if self.training.num_workers < 0:
            raise ValueError("training.num_workers must not be negative.")

    def resolve_device(self) -> torch.device:
        if self.runtime.device != "auto":
            return torch.device(self.runtime.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


# Edit this object to define an experiment. There is deliberately no global
# argparse state; the entry point passes this immutable configuration to SSR.
CONFIG = ExperimentConfig()

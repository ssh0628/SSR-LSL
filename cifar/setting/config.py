"""Independent CIFAR-10 SSR/LSL experiment configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal

import torch

# PROJECT_ROOT = Path("/root/project/ssr/cifar").expanduser().resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# - 논문 기본값: batch=128, lr=0.02, chunk=10
# - RTX 5080 (16 GB):
#   batch_size=256, num_workers=8, prefetch_factor=4,
#   SSR/LSL knn_chunks=8
# - H100 NVL (94 GB):
#   batch_size=512, num_workers=16, prefetch_factor=4,
#   SSR/LSL knn_chunks=2

DatasetName = Literal["cifar10"]
NoiseKind = Literal["symmetric", "asymmetric", "idn"]
ModelName = Literal[
    "preact_resnet18",
    "cifar_resnet18",
    "cifar_resnet34",
]
OptimizerName = Literal["sgd", "adamw"]


@dataclass(frozen=True, slots=True)
class DataConfig:
    """CIFAR-10 data and synthetic label-noise settings."""

    dataset: DatasetName = "cifar10"  # CIFAR-10 전용
    root: Path = field(default_factory=lambda: PROJECT_ROOT / "data")  # CIFAR-10 저장 경로
    download: bool = True  # 미보유 데이터 다운로드
    # - 합성 noise: idn / symmetric / asymmetric
    noise_kind: NoiseKind = "idn"
    noise_rate: float = 0.5  # 합성 노이즈 목표 비율
    idn_flip_rate_std: float = 0.1  # IDN truncated-normal 표준편차

    @property
    def num_classes(self) -> int:
        return 10

    def noise_file(self, seed: int) -> Path:
        """현재 noise 설정과 전역 seed에 대응하는 재사용 artifact 경로."""
        rate = f"{self.noise_rate:.4f}".rstrip("0").rstrip(".")
        standard_deviation = (
            f"_std{self.idn_flip_rate_std:g}" if self.noise_kind == "idn" else ""
        )
        filename = (
            f"cifar10_{self.noise_kind}_{rate}{standard_deviation}_seed{seed}.pt"
        )
        return self.root / "noise" / filename

    def validate(self) -> None:
        if self.dataset != "cifar10":
            raise ValueError("This package supports data.dataset='cifar10' only.")
        if not self.root.is_absolute():
            raise ValueError("data.root must be an absolute path.")
        if self.noise_kind not in {"symmetric", "asymmetric", "idn"}:
            raise ValueError("CIFAR-10 noise must be symmetric, asymmetric, or idn.")
        if not 0.0 <= self.noise_rate < 1.0:
            raise ValueError("data.noise_rate must be in [0, 1).")
        if not isfinite(self.idn_flip_rate_std) or self.idn_flip_rate_std <= 0.0:
            raise ValueError("data.idn_flip_rate_std must be positive.")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """분류 backbone 설정."""

    name: ModelName = "preact_resnet18"  # PreActResNet18 / CIFAR ResNet18·34
    # - CIFAR-ResNet18/34: 첫 eval 전 BN 통계 보정
    # - PreActResNet18: 미적용; optimizer update 없음
    calibrate_initial_batch_norm: bool = True

    def validate(self) -> None:
        if self.name not in {
            "preact_resnet18",
            "cifar_resnet18",
            "cifar_resnet34",
        }:
            raise ValueError("Unsupported model.name.")


@dataclass(frozen=True, slots=True)
class SSRConfig:
    """공식 CIFAR SSR 구현에서 유지한 설정."""

    # - relabel 조건: confidence > tau_r
    relabel_threshold: float = 0.55
    # - selection 기준: tau_s=1이면 최다 k-NN vote와 일치
    selection_threshold: float = 1.0
    neighbors: int = 200  # SSR cosine k-NN 이웃 수
    # - SSR k-NN query 분할 수; 메모리 조절용
    knn_chunks: int = 10
    feature_consistency_weight: float = 1.0  # feature-consistency loss 가중치
    mixup_alpha: float = 4.0  # CIFAR mixup Beta(alpha, alpha)

    def validate(self) -> None:
        if not 0.0 <= self.relabel_threshold <= 1.0:
            raise ValueError("ssr.relabel_threshold must be in [0, 1].")
        if not 0.0 <= self.selection_threshold <= 1.0:
            raise ValueError("ssr.selection_threshold must be in [0, 1].")
        if not 1 <= self.neighbors <= 50_000:
            raise ValueError("ssr.neighbors must be in [1, 50000].")
        if self.knn_chunks < 1:
            raise ValueError("ssr.knn_chunks must be positive.")
        if (
            not isfinite(self.feature_consistency_weight)
            or self.feature_consistency_weight < 0.0
        ):
            raise ValueError("ssr.feature_consistency_weight must not be negative.")
        if not isfinite(self.mixup_alpha) or self.mixup_alpha <= 0.0:
            raise ValueError("ssr.mixup_alpha must be positive.")


@dataclass(frozen=True, slots=True)
class StructuralLabelsConfig:
    """CVPR 2024 Learning with Structural Labels 설정."""

    enabled: bool = False  # LSL structural target/loss 활성화
    neighbors: int = 20  # reverse k-NN label 전파 이웃 수
    # - reverse k-NN query 분할 수; 메모리 조절용
    knn_chunks: int = 10
    loss_weight: float = 1.0  # structural mixup CE 가중치

    def validate(self) -> None:
        if not 1 <= self.neighbors <= 50_000:
            raise ValueError("structural_labels.neighbors must be in [1, 50000].")
        if self.knn_chunks < 1:
            raise ValueError("structural_labels.knn_chunks must be positive.")
        if not isfinite(self.loss_weight) or self.loss_weight < 0.0:
            raise ValueError("structural_labels.loss_weight must not be negative.")
        if self.enabled and self.loss_weight == 0.0:
            raise ValueError(
                "Disable structural_labels for the SSR ablation instead of using zero loss weight."
            )


@dataclass(frozen=True, slots=True)
class LabelWaveConfig:
    """ICLR 2024 Label Wave checkpoint 선택 설정."""

    enabled: bool = True  # prediction change 추적 + label_wave.pt 저장
    # - False: checkpoint 저장 + 전체 epoch 학습
    # - True: checkpoint 저장 + 조기 종료
    stop_training: bool = False
    # - prediction-change 이동평균 window
    # - 시작값 3: 논문 Appendix E 참고
    moving_average_window: int = 3
    # - patience: 개선 없는 연속 횟수; 실험 설정값
    patience: int = 20

    def validate(self, training_epochs: int) -> None:
        if self.moving_average_window < 1:
            raise ValueError("label_wave.moving_average_window must be positive.")
        if self.patience < 1:
            raise ValueError("label_wave.patience must be positive.")
        if self.stop_training and not self.enabled:
            raise ValueError(
                "Enable label_wave before setting label_wave.stop_training=True."
            )
        if self.enabled and self.moving_average_window > training_epochs:
            raise ValueError(
                "label_wave.moving_average_window must not exceed training.epochs."
            )


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """논문의 CIFAR 최적화 설정."""

    epochs: int = 300  # 학습 epoch 수; 별도 warm-up 없음
    batch_size: int = 128  # DataLoader mini-batch 크기
    learning_rate: float = 0.02  # 초기 learning rate
    # - encoder 전용 초기 LR
    # - None: head와 같은 learning_rate
    encoder_learning_rate: float | None = None
    optimizer: OptimizerName = "sgd"  # sgd / adamw
    momentum: float = 0.9  # SGD momentum
    weight_decay: float = 5e-4  # SGD L2 weight decay
    # - cosine 최저 LR / 각 parameter group 초기 LR
    scheduler_eta_min_ratio: float = 1.0 / 50.0
    num_workers: int = 4  # DataLoader worker 수
    # - worker당 미리 준비할 batch 수; workers=0이면 미사용
    prefetch_factor: int = 2
    # - epoch 간 worker 유지; selected loader는 매 epoch 재생성
    persistent_workers: bool = True

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("training.epochs must be positive.")
        if self.batch_size < 2:
            raise ValueError(
                "training.batch_size must be at least 2 for the SSR BatchNorm heads."
            )
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive.")
        if self.encoder_learning_rate is not None and (
            not isfinite(self.encoder_learning_rate) or self.encoder_learning_rate <= 0.0
        ):
            raise ValueError("training.encoder_learning_rate must be positive when set.")
        if self.optimizer not in {"sgd", "adamw"}:
            raise ValueError("training.optimizer must be sgd or adamw.")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("training.momentum must be in [0, 1).")
        if not isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("training.weight_decay must not be negative.")
        if not 0.0 <= self.scheduler_eta_min_ratio <= 1.0:
            raise ValueError("training.scheduler_eta_min_ratio must be in [0, 1].")
        if self.num_workers < 0:
            raise ValueError("training.num_workers must not be negative.")
        if self.prefetch_factor < 1:
            raise ValueError("training.prefetch_factor must be positive.")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """알고리즘 결과를 바꾸지 않는 실행·출력 설정."""

    device: str = "auto"  # auto / cuda / mps / cpu 또는 torch device 문자열
    # - config / metric / checkpoint 저장 루트
    output_root: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")

    def validate(self) -> None:
        if not self.output_root.is_absolute():
            raise ValueError("runtime.output_root must be an absolute path.")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """한 번의 noisy-label classification 실험 전체 설정."""

    seed: int = 0  # Python/NumPy/PyTorch/CUDA/noise 공통 seed
    data: DataConfig = field(default_factory=DataConfig)  # 데이터·합성 noise
    model: ModelConfig = field(default_factory=ModelConfig)  # backbone
    ssr: SSRConfig = field(default_factory=SSRConfig)  # SSR relabel / selection / loss
    structural_labels: StructuralLabelsConfig = field(
        default_factory=StructuralLabelsConfig
    )  # LSL reverse k-NN / structural loss
    label_wave: LabelWaveConfig = field(
        default_factory=LabelWaveConfig
    )  # Label Wave 저장 / 조기 종료
    training: TrainingConfig = field(default_factory=TrainingConfig)  # optimizer / epoch
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def run_name(self) -> str:
        """읽기 쉬운 prefix와 전체 설정 hash로 충돌을 막은 실행 이름."""
        rate = f"{self.data.noise_rate:.4f}".rstrip("0").rstrip(".")
        noise = f"{self.data.noise_kind}{rate}"
        algorithm = (
            f"lsl-k{self.structural_labels.neighbors}"
            if self.structural_labels.enabled
            else "ssr"
        )
        if self.label_wave.enabled:
            mode = "stop" if self.label_wave.stop_training else "monitor"
            algorithm += (
                f"-lw-{mode}-k{self.label_wave.moving_average_window}"
                f"-p{self.label_wave.patience}"
            )
        signature = asdict(self)
        signature["runtime"].pop("output_root")
        encoded = json.dumps(
            signature,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.blake2s(encoded, digest_size=4).hexdigest()
        return (
            f"cifar10_{noise}_{self.model.name}_{algorithm}"
            f"_tr{self.ssr.relabel_threshold:g}_ts{self.ssr.selection_threshold:g}"
            f"_seed{self.seed}_{fingerprint}"
        )

    @property
    def run_dir(self) -> Path:
        """config, metric, checkpoint를 저장할 현재 실행 디렉터리."""
        return self.runtime.output_root / self.run_name

    def validate(self) -> None:
        """학습을 시작하기 전에 잘못된 조합을 빠르게 차단."""
        if self.seed < 0:
            raise ValueError("seed must not be negative.")
        self.data.validate()
        self.model.validate()
        self.ssr.validate()
        self.structural_labels.validate()
        self.training.validate()
        self.label_wave.validate(self.training.epochs)
        self.runtime.validate()

    def resolve_device(self) -> torch.device:
        """명시적 device를 존중하고 auto에서는 CUDA, MPS, CPU 순으로 선택."""
        if self.runtime.device != "auto":
            return torch.device(self.runtime.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


# - 실행 설정: CONFIG
# - argparse / 숨은 override 없음
CONFIG = ExperimentConfig()

"""CIFAR-10 SSR/LSL 실험의 단일 설정 소스."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch

# PROJECT_ROOT = Path("/workspace/SSR-LSL").expanduser().resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 논문 기본값(batch=128, lr=0.02, chunk=10)
# RTX 5080 (16 GB):
#   batch_size=256, num_workers=8, prefetch_factor=4,
#   SSR/LSL knn_chunks=8
# H100 NVL (94 GB):
#   batch_size=512, num_workers=16, prefetch_factor=4,
#   SSR/LSL knn_chunks=2

NoiseKind = Literal["symmetric", "asymmetric", "idn"]
ModelName = Literal["preact_resnet18", "cifar_resnet18", "cifar_resnet34"]


@dataclass(frozen=True, slots=True)
class DataConfig:
    """CIFAR-10 데이터와 합성 노이즈 설정."""

    root: Path = field(default_factory=lambda: PROJECT_ROOT / "data")  # CIFAR-10 저장 경로.
    download: bool = True  # 데이터가 없을 때 torchvision으로 내려받을지 여부.
    noise_kind: NoiseKind = "idn"  # "idn", "symmetric", "asymmetric" 중 하나.
    noise_rate: float = 0.5  # 목표 합성 노이즈 비율. 논문 IDN 범위는 0.20~0.50.
    idn_flip_rate_std: float = 0.1  # Xia et al. IDN truncated-normal 표준편차.

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
        if not self.root.is_absolute():
            raise ValueError("data.root must be an absolute path.")
        if self.noise_kind not in {"symmetric", "asymmetric", "idn"}:
            raise ValueError("data.noise_kind must be symmetric, asymmetric, or idn.")
        if not 0.0 <= self.noise_rate < 1.0:
            raise ValueError("data.noise_rate must be in [0, 1).")
        if self.idn_flip_rate_std <= 0.0:
            raise ValueError("data.idn_flip_rate_std must be positive.")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """분류 backbone 설정."""

    name: ModelName = "preact_resnet18"  # 논문 backbone 또는 RLNLC의 CIFAR ResNet-18/34.

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

    # 예측 label로 바꾸기 위한 strict confidence 하한 tau_r.
    relabel_threshold: float = 0.55
    # k-NN consistency 선택 하한 tau_s; 1이면 최다 vote와 일치.
    selection_threshold: float = 1.0
    neighbors: int = 200  # SSR sample selection에 사용하는 cosine k-NN의 K.
    # 결과에 영향 없이 SSR k-NN query를 나누는 메모리 chunk 수.
    knn_chunks: int = 10
    feature_consistency_weight: float = 1.0  # feature-consistency loss 가중치 lambda_fc.
    mixup_alpha: float = 4.0  # CIFAR 실험의 Beta(alpha, alpha) mixup 파라미터.

    def validate(self) -> None:
        if not 0.0 <= self.relabel_threshold <= 1.0:
            raise ValueError("ssr.relabel_threshold must be in [0, 1].")
        if not 0.0 <= self.selection_threshold <= 1.0:
            raise ValueError("ssr.selection_threshold must be in [0, 1].")
        if not 1 <= self.neighbors <= 50_000:
            raise ValueError("ssr.neighbors must be in [1, 50000].")
        if self.knn_chunks < 1:
            raise ValueError("ssr.knn_chunks must be positive.")
        if self.feature_consistency_weight < 0.0:
            raise ValueError("ssr.feature_consistency_weight must not be negative.")
        if self.mixup_alpha <= 0.0:
            raise ValueError("ssr.mixup_alpha must be positive.")


@dataclass(frozen=True, slots=True)
class StructuralLabelsConfig:
    """CVPR 2024 Learning with Structural Labels 설정."""

    enabled: bool = True  # True면 reverse k-NN structural target과 L_st를 SSR에 추가.
    neighbors: int = 20  # reverse k-NN에서 각 source가 label을 전파할 이웃 수 k_st.
    # 결과에 영향 없이 reverse k-NN query를 나누는 메모리 chunk 수.
    knn_chunks: int = 10
    loss_weight: float = 1.0  # structural-label mixup cross-entropy 가중치 lambda_st.

    def validate(self) -> None:
        if not 1 <= self.neighbors <= 50_000:
            raise ValueError("structural_labels.neighbors must be in [1, 50000].")
        if self.knn_chunks < 1:
            raise ValueError("structural_labels.knn_chunks must be positive.")
        if self.loss_weight < 0.0:
            raise ValueError("structural_labels.loss_weight must not be negative.")
        if self.enabled and self.loss_weight == 0.0:
            raise ValueError(
                "Disable structural_labels for the SSR ablation instead of using zero loss weight."
            )


@dataclass(frozen=True, slots=True)
class LabelWaveConfig:
    """ICLR 2024 Label Wave checkpoint 선택 설정."""

    enabled: bool = True  # True면 training prediction change를 추적하고 label_wave.pt 저장.
    # False면 전체 epoch를 유지하며 선택 지점만 관찰; 검증 후 True로 바꿔 실제 조기 종료.
    stop_training: bool = False
    # 최근 k개 prediction-change의 이동평균. 논문 Appendix E에서 k=3 상관이 가장 강함.
    moving_average_window: int = 3
    # 논문은 patience 동작만 정의하고 고정 기본값은 공개하지 않아 실험값으로 노출.
    patience: int = 10

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

    epochs: int = 300  # scratch 학습 epoch 수; 별도 warm-up은 없음.
    batch_size: int = 128  # selected/all/test DataLoader의 mini-batch 크기.
    learning_rate: float = 0.02  # SGD 초기 learning rate.
    momentum: float = 0.9  # SGD momentum.
    weight_decay: float = 5e-4  # SGD L2 weight decay.
    scheduler_eta_min_ratio: float = 1.0 / 50.0  # cosine scheduler 최저 LR / 초기 LR.
    num_workers: int = 4  # 각 DataLoader의 worker process 수.
    # worker마다 미리 준비할 batch 수; num_workers=0이면 미사용.
    prefetch_factor: int = 2
    # epoch 사이 worker를 유지해 재시작 비용을 줄일지 여부.
    persistent_workers: bool = True

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("training epochs and batch_size must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive.")
        if self.momentum < 0.0 or self.weight_decay < 0.0:
            raise ValueError("training momentum and weight_decay must not be negative.")
        if self.scheduler_eta_min_ratio < 0.0:
            raise ValueError("training.scheduler_eta_min_ratio must not be negative.")
        if self.num_workers < 0:
            raise ValueError("training.num_workers must not be negative.")
        if self.prefetch_factor < 1:
            raise ValueError("training.prefetch_factor must be positive.")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """알고리즘 결과를 바꾸지 않는 실행·출력 설정."""

    device: str = "auto"  # "auto", "cuda", "mps", "cpu" 또는 torch device 문자열.
    # config, metric, checkpoint를 저장할 루트.
    output_root: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")

    def validate(self) -> None:
        if not self.output_root.is_absolute():
            raise ValueError("runtime.output_root must be an absolute path.")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """한 번의 CIFAR-10 실험 전체 설정."""

    seed: int = 0  # Python, NumPy, PyTorch, CUDA, noise 생성에 공통으로 쓰는 단일 seed.
    data: DataConfig = field(default_factory=DataConfig)  # 데이터와 label-noise 설정.
    model: ModelConfig = field(default_factory=ModelConfig)  # backbone 선택 설정.
    ssr: SSRConfig = field(default_factory=SSRConfig)  # SSR relabel/selection/loss 설정.
    structural_labels: StructuralLabelsConfig = field(
        default_factory=StructuralLabelsConfig
    )  # LSL reverse k-NN 및 structural loss 토글·설정.
    label_wave: LabelWaveConfig = field(
        default_factory=LabelWaveConfig
    )  # validation GT 없는 checkpoint 선택 및 optional early stopping.
    training: TrainingConfig = field(default_factory=TrainingConfig)  # optimizer와 epoch 설정.
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def run_name(self) -> str:
        """읽기 쉬운 prefix와 전체 설정 hash로 충돌을 막은 실행 이름."""
        rate = f"{self.data.noise_rate:.4f}".rstrip("0").rstrip(".")
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
            f"cifar10_{self.data.noise_kind}{rate}_{self.model.name}_{algorithm}"
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


# 이 객체만 수정하면 전체 실험이 바뀐다.
# argparse나 숨은 전역 override는 사용하지 않는다.
CONFIG = ExperimentConfig()

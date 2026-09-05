"""General image-dataset configuration; CIFAR experiments live in cifar/."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal

import torch

# - 프로젝트 루트: 아래 상수에서 직접 지정 가능
# PROJECT_ROOT = Path("/root/project/ssr").expanduser().resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# - ConvNeXtV2-Tiny / 224px / BF16 AMP 기준
# - RTX 5080 16 GB: batch_size=16, eval_batch_size=64, prefetch_factor=1
# - 현재 서버: H100 NVL 1 GPU / 16 vCPU / RAM 200 GB
# - H100 설정: batch_size=512, eval_batch_size=1024, prefetch_factor=4
# - 학습 worker: 8개 × 두 loader = 16개; evaluation worker=8
# - 처리량 우선 설정; 서버 실측 최적값은 아님
# - LR: GPU에 따른 자동 변경 없음
OptimizerName = Literal["sgd", "adamw"]


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """root 기준 NPY 파일명. 절대 경로도 사용 가능."""

    paths: str  # 이미지 경로 NPY
    labels: str  # 정수 class index NPY

    def validate(self) -> None:
        if not self.paths.strip() or not self.labels.strip():
            raise ValueError("Split paths/labels filenames must not be empty.")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """사용자가 준비한 이미지 경로/라벨 NPY. 추가 noise나 sqrt sampling 없음."""

    root: Path = Path("/root/project/dataset/npy_path/modify_npy")  # NPY 저장 위치
    image_root: Path | None = None  # 상대 이미지 경로 기준; None: root
    name: str = "a1-a7"  # 결과 폴더용 데이터셋 이름
    class_names: tuple[str, ...] = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")  # label 순서
    train: SplitConfig = field(
        default_factory=lambda: SplitConfig("train_path.npy", "train_labels.npy")
    )
    validation: SplitConfig | None = field(
        default_factory=lambda: SplitConfig("val_path.npy", "val_labels.npy")
    )  # None: validation 및 best.pt 생략
    test: SplitConfig | None = field(
        default_factory=lambda: SplitConfig("test_path.npy", "test_labels.npy")
    )  # None: 최종 test 평가 생략
    label_offset: int = 0  # 라벨 시작 번호: 0 또는 1
    image_size: int = 224  # 전체 이미지 resize 크기; ROI/cache 미사용
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # RGB 평균
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)  # RGB 표준편차
    allow_truncated_images: bool = True  # 잘린 이미지 decoder 재시도
    verify_images: bool = False  # 모델 생성 전 전체 이미지 decode 검사
    image_check_workers: int = 8  # 사전 검사 worker 수

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def validate(self) -> None:
        if not self.root.is_absolute():
            raise ValueError("data.root must be an absolute path.")
        if not isinstance(self.train, SplitConfig):
            raise ValueError("data.train must specify paths and labels files.")
        if self.image_root is not None and not self.image_root.is_absolute():
            raise ValueError("data.image_root must be an absolute path.")
        if (
            not self.name
            or not self.name.isascii()
            or not self.name.replace("-", "").replace("_", "").isalnum()
        ):
            raise ValueError("data.name must contain only letters, digits, '-' or '_'.")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("data.class_names must contain at least two unique names.")
        if any(not isinstance(name, str) or not name.strip() for name in self.class_names):
            raise ValueError("data.class_names must be non-empty strings.")
        for split in (self.train, self.validation, self.test):
            if split is not None:
                split.validate()
        if self.image_size < 1:
            raise ValueError("data.image_size must be positive.")
        if not isinstance(self.label_offset, int):
            raise ValueError("data.label_offset must be an integer.")
        if (
            len(self.mean) != 3
            or len(self.std) != 3
            or not all(isfinite(v) for v in (*self.mean, *self.std))
            or any(v <= 0 for v in self.std)
        ):
            raise ValueError("data.mean/std must have three finite channels and positive std.")
        if self.image_check_workers < 1:
            raise ValueError("data.image_check_workers must be positive.")


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """도메인에 맞춰 조절하는 증강. 평가 입력에는 적용하지 않는다."""

    horizontal_flip: float = 0.5  # 좌우 반전 확률
    vertical_flip: float = 0.5  # 상하 반전 확률; 방향 중요 시 0
    weak_rotation: float = 10.0  # weak view 회전 범위 ±degree
    strong_rotation: float = 15.0  # strong view 회전 범위 ±degree
    color_jitter: tuple[float, float, float, float] = (0.1, 0.1, 0.05, 0.02)  # 밝기/대비/채도/색상

    def validate(self) -> None:
        if not 0 <= self.horizontal_flip <= 1 or not 0 <= self.vertical_flip <= 1:
            raise ValueError("augmentation flip probabilities must be in [0, 1].")
        if any(not isfinite(v) or v < 0 for v in (self.weak_rotation, self.strong_rotation)):
            raise ValueError("augmentation rotation must be finite and non-negative.")
        if (
            len(self.color_jitter) != 4
            or any(not isfinite(v) or v < 0 for v in self.color_jitter)
            or self.color_jitter[3] > 0.5
        ):
            raise ValueError("Invalid color_jitter; hue must be in [0, 0.5].")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """timm backbone 이름. pretrained encoder + 새 SSR heads."""

    name: str = "convnextv2_tiny"  # timm 모델명; resnet18 / resnet34 등
    pretrained: bool = True  # pretrained weight 사용 여부
    drop_path_rate: float = 0.2  # stochastic-depth 비율
    calibrate_initial_batch_norm: bool = True  # scratch encoder BN 통계 초기 보정

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("model.name must not be empty.")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError("model.drop_path_rate must be in [0, 1).")


@dataclass(frozen=True, slots=True)
class SSRConfig:
    """SSR relabel, sample selection과 loss 설정."""

    # - relabel 조건: confidence > tau_r
    relabel_threshold: float = 0.9
    # - selection 기준: tau_s=1이면 최다 k-NN vote와 일치
    selection_threshold: float = 1.0
    neighbors: int = 200  # SSR cosine k-NN 이웃 수
    # - SSR k-NN query 분할 수; 메모리 조절용
    knn_chunks: int = 10
    feature_consistency_weight: float = 1.0  # feature-consistency loss 가중치
    mixup_alpha: float = 4.0  # mixup Beta(alpha, alpha)

    def validate(self) -> None:
        if not 0.0 <= self.relabel_threshold <= 1.0:
            raise ValueError("ssr.relabel_threshold must be in [0, 1].")
        if not 0.0 <= self.selection_threshold <= 1.0:
            raise ValueError("ssr.selection_threshold must be in [0, 1].")
        if self.neighbors < 1:
            raise ValueError("ssr.neighbors must be positive.")
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

    enabled: bool = True  # LSL structural target/loss 활성화
    neighbors: int = 20  # reverse k-NN label 전파 이웃 수
    # - reverse k-NN query 분할 수; 메모리 조절용
    knn_chunks: int = 10
    loss_weight: float = 1.0  # structural mixup CE 가중치

    def validate(self) -> None:
        if self.neighbors < 1:
            raise ValueError("structural_labels.neighbors must be positive.")
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
    """일반 이미지 데이터셋 학습 설정."""

    epochs: int = 200  # 학습 epoch 수; 별도 warm-up 없음
    batch_size: int = 512  # 학습 mini-batch; mixup forward 크기=1024
    eval_batch_size: int = 1024  # feature 추출·validation·test 배치; FP32 유지
    amp: bool = True  # CUDA 학습 forward: BF16; CPU/MPS: 미적용
    channels_last: bool = True  # CUDA encoder·이미지 메모리 배치 최적화
    fused_optimizer: bool = True  # CUDA AdamW fused kernel; 나머지 환경 미적용
    learning_rate: float = 1e-3  # classifier/projector/predictor 초기 LR
    # - encoder 전용 초기 LR
    # - None: head와 같은 learning_rate
    encoder_learning_rate: float | None = 3e-5
    optimizer: OptimizerName = "adamw"  # adamw / sgd
    momentum: float = 0.9  # SGD momentum
    weight_decay: float = 0.1  # weight decay
    # - cosine 최저 LR / 각 parameter group 초기 LR
    scheduler_eta_min_ratio: float = 1e-3
    num_workers: int = 8  # 학습 loader당 worker 수; 두 loader 합계 16개
    eval_num_workers: int = 8  # feature 추출·validation·test worker 수
    # - worker당 미리 준비할 batch 수; workers=0이면 미사용
    # - RAM 200 GB 활용; 두 학습 loader에 각각 최대 32 batch 준비
    prefetch_factor: int = 4
    # - epoch 간 worker 유지; selected loader는 매 epoch 재생성
    persistent_workers: bool = True
    log_interval: int = 20  # loss 진행 표시 갱신 간격; step 단위

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("training.epochs must be positive.")
        if self.batch_size < 2:
            raise ValueError(
                "training.batch_size must be at least 2 for the SSR BatchNorm heads."
            )
        if self.eval_batch_size < 1:
            raise ValueError("training.eval_batch_size must be positive.")
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
        if self.num_workers < 0 or self.eval_num_workers < 0:
            raise ValueError("training worker counts must not be negative.")
        if self.prefetch_factor < 1:
            raise ValueError("training.prefetch_factor must be positive.")
        if self.log_interval < 1:
            raise ValueError("training.log_interval must be positive.")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """실행·저장 설정."""

    device: str = "auto"  # auto: CUDA → MPS → CPU
    # - False: cuDNN autotune 활성화; 속도 우선
    # - True: cuDNN deterministic, autotune 해제; 완전한 재현성 보장은 아님
    # - 두 모드 모두 seed=0 유지; AMP 결과는 기존 FP32와 차이 가능
    deterministic: bool = False
    output_root: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")  # 결과 저장 루트
    run_id: str | None = None  # None: 실행 시각 ID; 중복 지정 ID 거부

    def validate(self) -> None:
        if not self.output_root.is_absolute():
            raise ValueError("runtime.output_root must be an absolute path.")
        if self.run_id is not None and (
            not self.run_id
            or self.run_id in {".", ".."}
            or Path(self.run_id).name != self.run_id
            or "/" in self.run_id
            or "\\" in self.run_id
        ):
            raise ValueError("runtime.run_id must be a single directory name.")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """이 설정 하나로 일반 데이터셋 실험 실행."""

    seed: int = 0  # Python/NumPy/PyTorch 공통 seed
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ssr: SSRConfig = field(default_factory=SSRConfig)
    structural_labels: StructuralLabelsConfig = field(default_factory=StructuralLabelsConfig)
    label_wave: LabelWaveConfig = field(default_factory=LabelWaveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def run_name(self) -> str:
        algorithm = "ssr-lsl" if self.structural_labels.enabled else "ssr"
        if self.label_wave.enabled:
            algorithm += "-lw-stop" if self.label_wave.stop_training else "-lw-save"
        signature = asdict(self)
        signature["runtime"].pop("output_root")
        signature["runtime"].pop("run_id")
        encoded = json.dumps(signature, sort_keys=True, default=str).encode("utf-8")
        fingerprint = hashlib.blake2s(encoded, digest_size=4).hexdigest()
        return f"{self.data.name}_{algorithm}_seed{self.seed}_{fingerprint}"

    @property
    def run_dir(self) -> Path:
        directory = self.runtime.output_root / self.run_name
        return directory / self.runtime.run_id if self.runtime.run_id else directory

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must not be negative.")
        self.data.validate()
        self.augmentation.validate()
        self.model.validate()
        self.ssr.validate()
        self.structural_labels.validate()
        self.training.validate()
        self.label_wave.validate(self.training.epochs)
        self.runtime.validate()

    def resolve_device(self) -> torch.device:
        if self.runtime.device != "auto":
            return torch.device(self.runtime.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


CONFIG = ExperimentConfig()

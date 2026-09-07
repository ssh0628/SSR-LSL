"""General image-dataset configuration; CIFAR experiments live in cifar/."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal, get_args

import torch

# - 경로: 아래 값 직접 수정
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 프로젝트
DATASET_ROOT = Path("/root/project/dataset/npy_path/modify_npy")  # 기존 NPY·bbox 입력
OUTPUT_ROOT = PROJECT_ROOT / "outputs"  # 학습 결과

# - H100 NVL: batch 256/1024, workers 16×2/32, prefetch 2; train 512 OOM
# - RTX 5080: train 16 / eval 64 / prefetch 1; 시작값
OptimizerName = Literal["sgd", "adamw"]
MissingBBoxPolicy = Literal["drop", "error", "full"]


def validate_missing_bbox(policy: str) -> None:
    if policy not in get_args(MissingBBoxPolicy):
        raise ValueError(f"data.missing_bbox must be one of {get_args(MissingBBoxPolicy)}.")


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """root 기준 NPY 파일명. 절대 경로도 사용 가능."""

    paths: str  # 이미지 경로 NPY
    labels: str  # 정수 class index NPY
    bboxes: str | None = None  # None: bbox cache 자동 탐색·생성

    def validate(self) -> None:
        if not self.paths.strip() or not self.labels.strip():
            raise ValueError("Split paths/labels filenames must not be empty.")
        if self.bboxes is not None and not self.bboxes.strip():
            raise ValueError("Split bbox filename must not be empty.")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """이미지·라벨·bbox 입력."""

    root: Path = field(default_factory=lambda: DATASET_ROOT)  # NPY 루트
    image_root: Path | None = None  # 상대 이미지 경로 기준; None: root
    annotation_root: Path | None = None  # JSON 루트; None: 이미지 옆
    name: str = "a1-a7"  # 실험 이름
    class_names: tuple[str, ...] = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")  # 라벨 순서
    train: SplitConfig = field(
        default_factory=lambda: SplitConfig("train_path.npy", "train_labels.npy")
    )
    validation: SplitConfig | None = field(
        default_factory=lambda: SplitConfig("val_path.npy", "val_labels.npy")
    )  # None: validation 생략
    test: SplitConfig | None = field(
        default_factory=lambda: SplitConfig("test_path.npy", "test_labels.npy")
    )  # None: test 생략
    label_offset: int = 0  # 라벨 시작 번호
    crop_bbox: bool = True  # 전체 이미지에서 bbox crop
    missing_bbox: MissingBBoxPolicy = "drop"  # 누락: 제외 / 중단 / 전체 이미지
    image_size: int = 224  # crop 후 resize
    bbox_workers: int = 16  # bbox JSON 확인 worker
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # 정규화 평균
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)  # 정규화 표준편차
    allow_truncated_images: bool = True  # 잘린 이미지 재시도
    verify_images: bool = False  # 사전 decode 검사
    image_check_workers: int = 8  # 검사 worker

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
        if self.annotation_root is not None and not self.annotation_root.is_absolute():
            raise ValueError("data.annotation_root must be an absolute path.")
        validate_missing_bbox(self.missing_bbox)
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
        if self.bbox_workers < 1:
            raise ValueError("data.bbox_workers must be positive.")


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """학습 전용 증강."""

    horizontal_flip: float = 0.5  # 좌우 반전 확률
    vertical_flip: float = 0.5  # 상하 반전 확률
    weak_rotation: float = 10.0  # weak 회전 ±도
    strong_rotation: float = 15.0  # strong 회전 ±도
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
    """Backbone 설정."""

    name: str = "convnextv2_tiny"  # timm 모델명
    pretrained: bool = True  # 사전학습 가중치
    drop_path_rate: float = 0.2  # stochastic depth
    calibrate_initial_batch_norm: bool = True  # scratch 모델 BN 보정

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("model.name must not be empty.")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError("model.drop_path_rate must be in [0, 1).")


@dataclass(frozen=True, slots=True)
class SSRConfig:
    """SSR relabel, sample selection과 loss 설정."""

    relabel_threshold: float = 0.9  # confidence > tau_r
    selection_threshold: float = 1.0  # 1: 최다 k-NN vote 일치
    neighbors: int = 200  # k-NN 이웃 수
    knn_chunks: int = 10  # query 분할 수
    feature_consistency_weight: float = 1.0  # consistency loss 가중치
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

    enabled: bool = True  # LSL 사용
    neighbors: int = 20  # reverse k-NN 이웃 수
    knn_chunks: int = 10  # query 분할 수
    loss_weight: float = 1.0  # structural loss 가중치

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

    enabled: bool = True  # GT 없이 Label Wave 저장
    stop_training: bool = False  # True: 조기 종료
    moving_average_window: int = 3  # PC 이동평균 길이
    patience: int = 20  # 연속 미개선 허용 횟수

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

    epochs: int = 300  # 학습·cosine 길이
    batch_size: int = 256  # 학습; mixup 입력은 2배
    eval_batch_size: int = 1024  # FP32 평가
    amp: bool = True  # CUDA 학습 BF16
    channels_last: bool = True  # CUDA 메모리 배치
    fused_optimizer: bool = True  # CUDA fused AdamW
    learning_rate: float = 1e-3  # head 초기 LR
    encoder_learning_rate: float | None = 3e-5  # encoder LR; None: head와 동일
    optimizer: OptimizerName = "adamw"  # adamw / sgd
    momentum: float = 0.9  # SGD momentum
    weight_decay: float = 0.1  # weight decay
    scheduler_eta_min_ratio: float = 1e-3  # 최저 LR / 초기 LR
    num_workers: int = 16  # 학습 loader당 worker; 동시 32개
    eval_num_workers: int = 32  # feature 추출·평가 worker
    prefetch_factor: int = 2  # worker당 선행 batch
    persistent_workers: bool = True  # all-sample worker 유지
    log_interval: int = 20  # 진행 표시 step 간격

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
    deterministic: bool = False  # False: cuDNN autotune
    output_root: Path = field(default_factory=lambda: OUTPUT_ROOT)  # 결과 루트
    run_id: str | None = None  # None: 새 실행 시각 ID

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

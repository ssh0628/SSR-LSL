"""CUDA BF16 학습 정책; CPU/MPS와 평가 경로는 기존 정밀도 유지."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor


def full_precision(device: torch.device | str):
    """외부 autocast 차단; 입력 dtype은 변경하지 않음."""
    device = torch.device(device)
    if device.type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    device: torch.device
    amp: bool = False
    channels_last: bool = False

    @classmethod
    def from_config(cls, training, device: torch.device | str) -> PrecisionPolicy:
        device = torch.device(device)
        amp = device.type == "cuda" and getattr(training, "amp", False)
        if amp:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA BF16 AMP requested, but CUDA is unavailable.")
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(
                        "CUDA device does not support BF16; set training.amp=False."
                    )
        return cls(
            device=device,
            amp=amp,
            channels_last=(
                device.type == "cuda" and getattr(training, "channels_last", False)
            ),
        )

    def autocast(self):
        if self.amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return full_precision(self.device)

    def to_device(self, images: Tensor) -> Tensor:
        memory_format = (
            torch.channels_last
            if self.channels_last and images.ndim == 4
            else torch.contiguous_format
        )
        return images.to(
            self.device, non_blocking=True, memory_format=memory_format
        )

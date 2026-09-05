"""Real-image SSR/LSL step benchmark; no checkpoints, correction, or config writes."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator
from dataclasses import replace
from itertools import islice
from pathlib import Path

import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

from setting.config import CONFIG, ExperimentConfig
from setting.data import build_experiment_data
from setting.model import build_ssr_networks
from ssr.engine import _build_optimizer, _loader_options, seed_everything
from ssr.sampler import ClassBalancedSampler
from ssr.trainer import train_epoch

# - H100 후보 비교; config의 batch_size는 변경하지 않음
BATCH_SIZES = (256, 512, 1024)
WARMUP_STEPS = 3  # worker 시작 / kernel 준비; 시간 측정 제외
# - 기본 prefetch 대기열(8 workers × 4 batch)보다 긴 구간 측정
# - worker/prefetch 증가 시 함께 확대; 데이터 읽기·증강·학습 시간 포함
TIMED_STEPS = 40
_RESULT_PREFIX = "BENCHMARK_RESULT "


class _StepBatches:
    """Share a live loader iterator across warm-up and timed training."""

    def __init__(self, batches: Iterator, steps: int) -> None:
        self.batches = batches
        self.steps = steps

    def __iter__(self) -> Iterator:
        return islice(self.batches, self.steps)

    def __len__(self) -> int:
        return self.steps


def _repeat_batches(loader: DataLoader) -> Iterator:
    if not len(loader):
        raise ValueError("Benchmark requires at least one full training batch.")
    while True:
        yield from loader


def _benchmark_config(config: ExperimentConfig, batch_size: int) -> ExperimentConfig:
    if batch_size < 2:
        raise ValueError("Benchmark batch size must be at least 2.")
    return replace(
        config,
        data=replace(config.data, validation=None, test=None, verify_images=False),
        model=replace(config.model, pretrained=False, calibrate_initial_batch_norm=False),
        training=replace(config.training, batch_size=batch_size),
    )


def benchmark_batch(
    config: ExperimentConfig,
    batch_size: int,
    *,
    warmup_steps: int = WARMUP_STEPS,
    timed_steps: int = TIMED_STEPS,
) -> dict:
    """Run scratch training on real images; labels are benchmark-only surrogates."""
    if warmup_steps < 1 or timed_steps < 1:
        raise ValueError("Benchmark warm-up and timed step counts must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Run benchmark.py on the training server.")

    # Imported lazily so helper tests also run without initializing a CUDA device.
    from ssr.engine import configure_execution

    config = _benchmark_config(config, batch_size)
    config.validate()
    device = config.resolve_device()
    if device.type != "cuda":
        raise ValueError("Benchmark requires runtime.device='auto' or a CUDA device.")
    seed_everything(config.seed)
    data = build_experiment_data(
        config.data,
        structural_labels_enabled=config.structural_labels.enabled,
        augmentation=config.augmentation,
    )
    if len(data.all_train) < batch_size:
        raise ValueError(
            f"Train images ({len(data.all_train)}) fewer than batch size {batch_size}."
        )
    options = _loader_options(config, device)
    selected_loader = DataLoader(
        data.selected_train,
        sampler=ClassBalancedSampler(data.noisy_labels, data.num_classes),
        drop_last=True,
        **options,
    )
    all_loader = DataLoader(data.all_train, shuffle=True, drop_last=True, **options)
    networks = build_ssr_networks(config.model, device, data.num_classes)
    configure_execution(config, networks, device)
    optimizer = _build_optimizer(networks, config)
    labels = data.noisy_labels.to(device)
    # Same CE/FC/LSL compute paths, without a full feature scan or k-NN pass.
    # These one-hot targets do NOT measure corrected-label or model quality.
    structural_targets = (
        one_hot(labels, num_classes=data.num_classes).float()
        if config.structural_labels.enabled else None
    )
    selected_batches = _repeat_batches(selected_loader)
    all_batches = _repeat_batches(all_loader)

    def train_steps(steps: int) -> None:
        train_epoch(
            _StepBatches(selected_batches, steps),
            _StepBatches(all_batches, steps),
            labels,
            structural_targets,
            networks,
            optimizer,
            config,
            device,
            epoch=0,
        )

    torch.cuda.reset_peak_memory_stats(device)
    train_steps(warmup_steps)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    train_steps(timed_steps)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    sample_rate = batch_size * timed_steps / elapsed
    forward_views = 6 if config.structural_labels.enabled else 4
    return {
        "status": "ok",
        "batch_size": batch_size,
        "num_workers_per_loader": config.training.num_workers,
        "gpu": torch.cuda.get_device_name(device),
        "model": config.model.name,
        "lsl": config.structural_labels.enabled,
        "amp": config.training.amp,
        "channels_last": config.training.channels_last,
        "timed_steps": timed_steps,
        "seconds_per_step": elapsed / timed_steps,
        "all_sample_images_per_second": sample_rate,
        "encoder_image_forwards_per_second": sample_rate * forward_views,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }


def _run_candidate(batch_size: int) -> dict:
    """A separate process releases GPU allocations and workers even after OOM."""
    result = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve()), "--batch-size", str(batch_size)],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_RESULT_PREFIX):
            return json.loads(line[len(_RESULT_PREFIX):])
    return {
        "status": "error",
        "batch_size": batch_size,
        "message": f"Benchmark process exited {result.returncode} without a result. "
        "Check the stderr output above (including DataLoader/shared-memory errors).",
    }


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--batch-size":
        batch_size = int(sys.argv[2])
        try:
            result = benchmark_batch(CONFIG, batch_size)
        except torch.cuda.OutOfMemoryError as error:
            result = {"status": "oom", "batch_size": batch_size, "message": str(error)}
        except Exception as error:
            traceback.print_exc()
            result = {"status": "error", "batch_size": batch_size, "message": str(error)}
        print(_RESULT_PREFIX + json.dumps(result), flush=True)
        return 1 if result["status"] == "error" else 0
    if len(sys.argv) != 1:
        raise SystemExit("Usage: python benchmark.py (edit BATCH_SIZES at file top)")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required. Run benchmark.py on the training server.")

    print(
        "Real training images + configured augmentations + SSR/LSL training steps.\n"
        "Scratch weights / surrogate targets: no accuracy or whole-epoch estimate.\n"
        "No pretrained download, preflight scan, k-NN, checkpoint, or config changes.\n"
        "Run without another training process on the same GPU.\n"
        f"Warm-up {WARMUP_STEPS} steps; timed {TIMED_STEPS} steps; "
        f"workers {CONFIG.training.num_workers} per loader.\n"
        "Timing includes loader waits; peak memory includes warm-up.\n"
        "Timed steps should exceed workers × prefetch_factor to expose sustained I/O.\n"
        "sample/s counts all-sample batches, not unique images; encoder-view/s "
        "also counts selected and repeated views.\n",
        flush=True,
    )
    results = []
    for batch_size in BATCH_SIZES:
        print(f"Benchmark batch_size={batch_size} ...", flush=True)
        result = _run_candidate(batch_size)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["status"] == "error":
            return 1

    print("\n batch  step(ms)  sample/s  encoder-view/s  allocated(GiB)  reserved(GiB)")
    for result in results:
        if result["status"] == "oom":
            print(f"{result['batch_size']:6d}  OOM")
            continue
        print(
            f"{result['batch_size']:6d}  {result['seconds_per_step'] * 1000:8.1f}  "
            f"{result['all_sample_images_per_second']:8.1f}  "
            f"{result['encoder_image_forwards_per_second']:14.1f}  "
            f"{result['peak_allocated_gib']:14.2f}  {result['peak_reserved_gib']:13.2f}"
        )
    print("Choose batch size after comparing speed, memory headroom, and validation quality.")
    return 0 if any(result["status"] == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

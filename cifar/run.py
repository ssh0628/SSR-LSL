"""Run the experiment queue recorded in ssr+lsl.xlsx."""

from __future__ import annotations

import gc
import json
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import torch

if __package__ in {None, ""}:
    # Support both python -m cifar.run and python cifar/run.py.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifar.setting.config import CONFIG, ExperimentConfig
from cifar.ssr.engine import run as run_experiment


# Run 1-3의 비교가 끝났으므로 동일한 Run 3 설정에 Label Wave만 켠다.
RUN_IDS_TO_RUN = (4,)

# last.pt가 있는 완료 실험은 다시 계산하지 않는다.
SKIP_COMPLETED = True

# 중단된 실행의 기존 로그를 덮어쓰지 않는다.
# 처음부터 다시 돌릴 때만 True.
RESTART_INCOMPLETE = False

# 한 실험이 실패하면 다음 실험까지 계속할지 여부.
CONTINUE_ON_ERROR = False


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Spreadsheet Run ID and its immutable training configuration."""

    run_id: int
    config_id: str
    description: str
    config: ExperimentConfig


@dataclass(frozen=True, slots=True)
class RunResult:
    run_id: int
    config_id: str
    status: str
    run_dir: str
    best_accuracy: float | None = None
    error: str | None = None


def build_experiments(base: ExperimentConfig = CONFIG) -> tuple[ExperimentSpec, ...]:
    """Build the baseline runs and the Label Wave monitoring run."""
    common = replace(
        base,
        seed=0,
        data=replace(
            base.data,
            noise_kind="idn",
            noise_rate=0.5,
            idn_flip_rate_std=0.1,
        ),
        model=replace(base.model, name="cifar_resnet34"),
        ssr=replace(
            base.ssr,
            relabel_threshold=0.55,
            selection_threshold=1.0,
            neighbors=200,
            feature_consistency_weight=1.0,
            mixup_alpha=4.0,
        ),
        structural_labels=replace(
            base.structural_labels,
            enabled=False,
            neighbors=20,
            loss_weight=1.0,
        ),
        label_wave=replace(
            base.label_wave,
            enabled=False,
            stop_training=False,
        ),
        training=replace(
            base.training,
            epochs=300,
            batch_size=128,
            learning_rate=0.02,
            momentum=0.9,
            weight_decay=5e-4,
            scheduler_eta_min_ratio=1.0 / 50.0,
            num_workers=4,
        ),
    )
    run_1 = replace(
        common,
        ssr=replace(common.ssr, relabel_threshold=0.9),
    )
    run_3 = replace(
        common,
        structural_labels=replace(common.structural_labels, enabled=True),
    )
    run_4 = replace(
        run_3,
        label_wave=replace(
            run_3.label_wave,
            enabled=True,
            stop_training=False,
            moving_average_window=3,
            patience=20,
        ),
    )
    return (
        ExperimentSpec(1, "C01", "SSR / tau_r=0.90", run_1),
        ExperimentSpec(2, "C02", "SSR / tau_r=0.55", common),
        ExperimentSpec(3, "C03", "SSR+LSL / tau_r=0.55", run_3),
        ExperimentSpec(
            4,
            "C04",
            "SSR+LSL / tau_r=0.55 / Label Wave monitor",
            run_4,
        ),
    )


def select_experiments(
    experiments: Iterable[ExperimentSpec],
    run_ids: Iterable[int],
) -> tuple[ExperimentSpec, ...]:
    """Select runs in the requested order and reject ambiguous queues."""
    by_id: dict[int, ExperimentSpec] = {}
    for experiment in experiments:
        if experiment.run_id in by_id:
            raise ValueError(f"Duplicate experiment Run ID: {experiment.run_id}")
        by_id[experiment.run_id] = experiment

    selected: list[ExperimentSpec] = []
    seen: set[int] = set()
    for run_id in run_ids:
        if run_id in seen:
            raise ValueError(f"Duplicate Run ID in queue: {run_id}")
        if run_id not in by_id:
            raise ValueError(f"Unknown experiment Run ID: {run_id}")
        selected.append(by_id[run_id])
        seen.add(run_id)
    if not selected:
        raise ValueError("RUN_IDS_TO_RUN must contain at least one Run ID.")
    return tuple(selected)


def _has_run_artifacts(run_dir: Path) -> bool:
    return any(
        (run_dir / filename).exists()
        for filename in ("config.json", "metrics.jsonl", "best.pt", "last.pt")
    )


def _is_completed(run_dir: Path) -> bool:
    return (run_dir / "last.pt").is_file()


def _release_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _append_batch_result(path: Path, result: RunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **asdict(result),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(values, ensure_ascii=False) + "\n")


def run_batch(
    experiments: Iterable[ExperimentSpec],
    *,
    skip_completed: bool = SKIP_COMPLETED,
    restart_incomplete: bool = RESTART_INCOMPLETE,
    continue_on_error: bool = CONTINUE_ON_ERROR,
    training_fn: Callable[[ExperimentConfig], float] = run_experiment,
) -> tuple[RunResult, ...]:
    """Run experiments sequentially, isolating logs and clearing device caches."""
    queue = tuple(experiments)
    if not queue:
        raise ValueError("The experiment queue must not be empty.")

    output_root = queue[0].config.runtime.output_root
    status_path = output_root / "batch_status.jsonl"
    results: list[RunResult] = []
    print(f"experiment_queue={[experiment.run_id for experiment in queue]}")

    for position, experiment in enumerate(queue, start=1):
        config = experiment.config
        config.validate()
        run_dir = config.run_dir
        prefix = f"[{position}/{len(queue)}] Run {experiment.run_id} ({experiment.config_id})"

        if _is_completed(run_dir) and skip_completed:
            result = RunResult(
                experiment.run_id,
                experiment.config_id,
                "skipped_completed",
                str(run_dir),
            )
            results.append(result)
            _append_batch_result(status_path, result)
            print(f"{prefix} skip: {run_dir}")
            continue

        if _has_run_artifacts(run_dir) and not restart_incomplete:
            raise RuntimeError(
                f"{prefix} has an incomplete or existing run directory: {run_dir}\n"
                "Move the directory aside, or set RESTART_INCOMPLETE=True to overwrite "
                "its training logs."
            )

        print(f"{prefix} start: {experiment.description}")
        print(f"output={run_dir}")
        try:
            best_accuracy = training_fn(config)
        except Exception as error:
            result = RunResult(
                experiment.run_id,
                experiment.config_id,
                "failed",
                str(run_dir),
                error=f"{type(error).__name__}: {error}",
            )
            results.append(result)
            _append_batch_result(status_path, result)
            print(f"{prefix} failed: {result.error}")
            if not continue_on_error:
                raise
        else:
            result = RunResult(
                experiment.run_id,
                experiment.config_id,
                "completed",
                str(run_dir),
                best_accuracy=best_accuracy,
            )
            results.append(result)
            _append_batch_result(status_path, result)
            print(f"{prefix} complete: best_accuracy={best_accuracy:.4f}")
        finally:
            _release_device_cache()

    return tuple(results)


def main() -> None:
    queue = select_experiments(build_experiments(), RUN_IDS_TO_RUN)
    run_batch(queue)


if __name__ == "__main__":
    main()

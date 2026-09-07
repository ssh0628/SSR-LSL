"""Final held-out evaluation; never used to choose training checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from log.checkpoint import CHECKPOINT_FILENAMES, CheckpointManager
from log.common import JsonlWriter
from setting.model import SSRNetworks
from ssr.metrics import evaluate_classification


def evaluate_checkpoints(
    checkpoints: CheckpointManager,
    networks: SSRNetworks,
    loader: DataLoader | None,
    device: torch.device,
    *,
    channels_last: bool,
    class_names: tuple[str, ...],
    evaluator: Callable = evaluate_classification,
) -> None:
    """Write one row per saved name, evaluate shared epoch weights only once.

    A completed epoch identifies one network state within this run. Different
    selection criteria can choose the same epoch and share its test evaluation.
    Stored weights/selection stay unchanged even if final evaluation fails.
    """
    evaluated: dict[int, dict] = {}
    with JsonlWriter(checkpoints.run_dir / "checkpoint_results.jsonl") as writer:
        for filename in CHECKPOINT_FILENAMES:
            record = checkpoints.records.get(filename)
            if record is None:
                writer.write({
                    "filename": filename, "status": "not_created",
                    "reason": (
                        "validation unavailable" if filename.startswith("best_")
                        else "Label Wave disabled or no candidate observed"
                    ),
                })
                continue
            epoch = record["completed_epochs"]
            started = perf_counter()
            reused = epoch in evaluated
            if loader is not None and not reused:
                state = torch.load(
                    checkpoints.run_dir / filename, map_location="cpu", weights_only=True,
                )
                networks.encoder.load_state_dict(state["encoder"])
                networks.classifier.load_state_dict(state["classifier"])
                del state  # Do not retain CPU optimizer tensors across checkpoints.
                evaluated[epoch] = evaluator(
                    loader, networks, device, description=f"Test epoch {epoch}",
                    channels_last=channels_last, class_names=class_names,
                )
            test = evaluated.get(epoch)
            writer.write({
                **record, "status": "evaluated" if test is not None else "test_disabled",
                "test": test, "test_reused_same_epoch": reused,
                "evaluation_seconds": perf_counter() - started,
                "metric_units": "rates in [0, 1]",
            })
            if test is not None:
                print(
                    f"checkpoint={filename} epoch={epoch} "
                    f"test_acc={test['accuracy']:.2%} "
                    f"test_bal_acc={test['balanced_accuracy']:.2%} "
                    f"test_macro_f1={test['macro_f1']:.2%}", flush=True,
                )

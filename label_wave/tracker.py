"""ICLR 2024 Label Wave prediction-change tracker."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class LabelWaveObservation:
    """Label Wave state after observing one epoch-boundary prediction vector."""

    epoch: int
    prediction_changes: int | None
    moving_average: float | None
    best_moving_average: float | None
    selected_epoch: int | None
    patience_count: int
    is_candidate: bool
    should_stop: bool


class LabelWaveTracker:
    """Track the first stable local minimum of prediction changes.

    The implementation follows Algorithm 1 from the Label Wave paper:
    prediction changes are averaged over the latest ``window`` epochs, a new
    checkpoint is selected for every strict running minimum, and training is
    considered complete after ``patience`` consecutive non-improvements.
    """

    def __init__(self, *, window: int, patience: int) -> None:
        if window < 1:
            raise ValueError("window must be positive.")
        if patience < 1:
            raise ValueError("patience must be positive.")
        self.window = window
        self.patience = patience
        self._recent_changes: deque[int] = deque(maxlen=window)
        self._previous_predictions: Tensor | None = None
        self._best_moving_average: float | None = None
        self._selected_epoch: int | None = None
        self._patience_count = 0
        self._decision_reached = False
        self._last_epoch: int | None = None

    @property
    def selected_epoch(self) -> int | None:
        return self._selected_epoch

    def update(self, predictions: Tensor, *, epoch: int) -> LabelWaveObservation:
        """Observe predictions after ``epoch`` completed training epochs."""
        if epoch < 0:
            raise ValueError("epoch must not be negative.")
        if self._last_epoch is not None and epoch <= self._last_epoch:
            raise ValueError("epoch must increase strictly between observations.")
        if predictions.ndim != 1 or predictions.numel() == 0:
            raise ValueError("predictions must be a non-empty rank-1 tensor.")

        current = predictions.detach().to(device="cpu", dtype=torch.long).clone()
        self._last_epoch = epoch
        if self._previous_predictions is None:
            self._previous_predictions = current
            return self._observation(epoch, None, None, False, False)
        if current.shape != self._previous_predictions.shape:
            raise ValueError("prediction vector size changed between epochs.")

        prediction_changes = int(current.ne(self._previous_predictions).sum().item())
        self._previous_predictions = current
        self._recent_changes.append(prediction_changes)
        if len(self._recent_changes) < self.window:
            return self._observation(
                epoch,
                prediction_changes,
                None,
                False,
                False,
            )

        moving_average = sum(self._recent_changes) / self.window
        if self._decision_reached:
            return self._observation(
                epoch,
                prediction_changes,
                moving_average,
                False,
                False,
            )

        is_candidate = (
            self._best_moving_average is None
            or moving_average < self._best_moving_average
        )
        if is_candidate:
            self._best_moving_average = moving_average
            self._selected_epoch = epoch
            self._patience_count = 0
        else:
            self._patience_count += 1

        should_stop = self._patience_count >= self.patience
        if should_stop:
            self._decision_reached = True
        return self._observation(
            epoch,
            prediction_changes,
            moving_average,
            is_candidate,
            should_stop,
        )

    def _observation(
        self,
        epoch: int,
        prediction_changes: int | None,
        moving_average: float | None,
        is_candidate: bool,
        should_stop: bool,
    ) -> LabelWaveObservation:
        return LabelWaveObservation(
            epoch=epoch,
            prediction_changes=prediction_changes,
            moving_average=moving_average,
            best_moving_average=self._best_moving_average,
            selected_epoch=self._selected_epoch,
            patience_count=self._patience_count,
            is_candidate=is_candidate,
            should_stop=should_stop,
        )

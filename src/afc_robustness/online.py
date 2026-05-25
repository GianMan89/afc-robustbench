"""Prefix-based online AFC evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from afc_robustness.domain import AlarmEpisode
from afc_robustness.models.base import AFCModel
from afc_robustness.representations import episode_to_series

ProgressReference = Literal["event_count", "relative_time"]


@dataclass(frozen=True)
class OnlineEvaluationConfig:
    """Configuration for online prefix evaluation."""

    progress_grid: tuple[float, ...] = tuple(np.round(np.linspace(0.1, 1.0, 10), 3))
    update_interval: float = 1.0
    update_start: float | None = None
    progress_reference: ProgressReference = "event_count"
    include_zero_update: bool = True
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.update_start is not None and self.update_start < 0:
            raise ValueError("update_start must be non-negative when provided")
        if self.progress_reference not in {"event_count", "relative_time"}:
            raise ValueError("progress_reference must be 'event_count' or 'relative_time'")
        for rho in self.progress_grid:
            if not 0 < rho <= 1:
                raise ValueError("progress_grid values must be in (0, 1]")


def prefix_by_time(episode: AlarmEpisode, time: float) -> AlarmEpisode:
    """Return the causal prefix available at a given time."""

    u = min(max(float(time), 0.0), episode.horizon)
    events = [event for event in episode.events if event.timestamp <= u + 1e-12]
    return episode.with_events(events, horizon=u)


def native_update_times(
    horizon: float,
    *,
    update_interval: float,
    include_zero_update: bool = True,
    update_start: float | None = None,
) -> np.ndarray:
    """Return a time-driven native update grid.

    If ``update_start`` is provided, native updates start at that time.
    Otherwise, the legacy behavior is used: 0.0 when ``include_zero_update``
    is true, and ``update_interval`` when it is false.
    """

    if horizon <= 0:
        return np.array([0.0])
    start = float(update_start) if update_start is not None else (0.0 if include_zero_update else update_interval)
    if start > horizon:
        return np.array([float(horizon)])
    times = np.arange(start, horizon + 1e-12, update_interval, dtype=float)
    if times.size == 0 or abs(times[-1] - horizon) > 1e-9:
        times = np.append(times, horizon)
    return np.unique(np.clip(times, 0.0, horizon))


def map_predictions_to_progress_grid(
    native_progress: np.ndarray,
    native_predictions: np.ndarray,
    progress_grid: tuple[float, ...],
) -> dict[float, int]:
    """Map native online predictions to a common progress grid causally.

    For each grid point, the latest prediction with native progress not greater
    than the grid point is selected. If no such prediction exists, the first
    native prediction is used as the manuscript's pragmatic no-gap convention.
    """

    if len(native_predictions) == 0:
        raise ValueError("native_predictions must not be empty")
    order = np.argsort(native_progress)
    native_progress = np.asarray(native_progress)[order]
    native_predictions = np.asarray(native_predictions)[order]
    mapped: dict[float, int] = {}
    for rho in progress_grid:
        idx = np.searchsorted(native_progress, rho, side="right") - 1
        if idx < 0:
            idx = 0
        mapped[float(rho)] = int(native_predictions[idx])
    return mapped


@dataclass
class OnlineEvaluator:
    """Run an AFC model on repaired perturbed episodes under online observation."""

    config: OnlineEvaluationConfig

    def evaluate_episode(self, model: AFCModel, episode: AlarmEpisode) -> dict[float, int]:
        times = native_update_times(
            episode.horizon,
            update_interval=self.config.update_interval,
            include_zero_update=self.config.include_zero_update,
            update_start=self.config.update_start,
        )
        native_progress: list[float] = []
        native_predictions: list[int] = []
        n_events_total = max(episode.n_events, 1)
        horizon = max(float(episode.horizon), 0.0)
        n_time_total = max(1, int(np.floor(horizon / self.config.dt + 1e-9)) + 1)

        for u in times:
            u = float(u)
            prefix = prefix_by_time(episode, u)
            X_prefix = episode_to_series(prefix, dt=self.config.dt)[None, :, :]

            event_count_progress = min(prefix.n_events / n_events_total, 1.0)
            relative_time_progress = 1.0 if horizon <= 0 else min(u / horizon, 1.0)
            sample_count_progress = min(X_prefix.shape[2] / n_time_total, 1.0)
            progress = (
                relative_time_progress
                if self.config.progress_reference == "relative_time"
                else event_count_progress
            )

            pred = int(
                model.predict_with_context(
                    X_prefix,
                    progress=progress,
                    progress_reference=self.config.progress_reference,
                    event_count_progress=event_count_progress,
                    relative_time_progress=relative_time_progress,
                    sample_count_progress=sample_count_progress,
                    update_time=u,
                    episode_horizon=horizon,
                    n_events_observed=prefix.n_events,
                    n_events_total=episode.n_events,
                )[0]
            )

            native_progress.append(progress)
            native_predictions.append(pred)

        return map_predictions_to_progress_grid(
            np.asarray(native_progress, dtype=float),
            np.asarray(native_predictions, dtype=int),
            self.config.progress_grid,
        )

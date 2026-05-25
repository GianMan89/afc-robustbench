"""Conversions between binary alarm series, event streams, sets, and sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from afc_robustness.domain import AlarmEpisode, AlarmEvent, EventType

InitialActivePolicy = Literal["act", "ignore"]


@dataclass(frozen=True)
class SeriesConversionConfig:
    """Configuration for series/event conversion."""

    dt: float = 1.0
    initial_active_policy: InitialActivePolicy = "act"
    binary_threshold: float = 0.5


def _as_binary_matrix(matrix: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-D with shape (n_tags, n_time_steps)")
    return (arr > threshold).astype(np.int8, copy=False)


def series_to_episode(
    matrix: np.ndarray,
    *,
    tag_names: Sequence[str] | None = None,
    sample_id: str = "",
    dt: float = 1.0,
    time_points: Sequence[float] | None = None,
    initial_active_policy: InitialActivePolicy = "act",
    binary_threshold: float = 0.5,
) -> AlarmEpisode:
    """Convert a binary alarm-state matrix to an event-stream episode.

    Parameters
    ----------
    matrix:
        Binary matrix with shape ``(n_tags, n_time_steps)``.
    tag_names:
        Optional tag names. If omitted, integer tag names are used.
    time_points:
        Optional sampling times. If supplied, times are shifted so that the
        first sample has local time zero.
    initial_active_policy:
        ``"act"`` inserts an activation event at time zero for tags that are
        active in the first sample. ``"ignore"`` does not insert such events.
    """

    x = _as_binary_matrix(matrix, threshold=binary_threshold)
    n_tags, n_time = x.shape
    if n_time == 0:
        raise ValueError("matrix must contain at least one time step")

    if time_points is None:
        times = np.arange(n_time, dtype=float) * float(dt)
    else:
        times = np.asarray(time_points, dtype=float)
        if times.ndim != 1 or len(times) != n_time:
            raise ValueError("time_points must be one-dimensional and match n_time_steps")
        times = times - times[0]
        if np.any(np.diff(times) < -1e-12):
            raise ValueError("time_points must be nondecreasing")

    events: list[AlarmEvent] = []
    order = 0
    for tag in range(n_tags):
        row = x[tag]
        if row[0] == 1 and initial_active_policy == "act":
            events.append(AlarmEvent(tag, EventType.ACT, float(times[0]), order))
            order += 1

        transitions = np.diff(row.astype(np.int8))
        for idx, diff in enumerate(transitions, start=1):
            if diff == 1:
                events.append(AlarmEvent(tag, EventType.ACT, float(times[idx]), order))
                order += 1
            elif diff == -1:
                events.append(AlarmEvent(tag, EventType.RTN, float(times[idx]), order))
                order += 1

    horizon = float(times[-1]) if n_time > 1 else 0.0
    return AlarmEpisode(events, horizon=horizon, n_tags=n_tags, tag_names=tag_names, sample_id=sample_id)


def episode_to_series(
    episode: AlarmEpisode,
    *,
    dt: float = 1.0,
    max_length: int | None = None,
    binary_dtype: type = np.int8,
) -> np.ndarray:
    """Convert an event-stream episode to a binary alarm-state matrix.

    Active intervals are reconstructed as ``ACT`` to the next ``RTN``. A leading
    ``RTN`` for a tag is interpreted as an alarm that was already active at the
    episode start and returns to normal at that timestamp.
    """

    if max_length is None:
        n_time = max(1, int(np.floor(episode.horizon / dt + 1e-9)) + 1)
    else:
        n_time = max(1, int(max_length))
    grid = np.arange(n_time, dtype=float) * float(dt)
    x = np.zeros((episode.n_tags, n_time), dtype=binary_dtype)

    by_tag: list[list[AlarmEvent]] = [[] for _ in range(episode.n_tags)]
    for event in sorted(episode.events, key=lambda e: (e.timestamp, e.order)):
        by_tag[event.tag].append(event)

    def fill(tag: int, start: float, end: float | None) -> None:
        if end is None:
            mask = (grid >= start - 1e-12) & (grid <= episode.horizon + 1e-12)
        else:
            mask = (grid >= start - 1e-12) & (grid < end - 1e-12)
        x[tag, mask] = 1

    for tag, events in enumerate(by_tag):
        if not events:
            continue
        idx = 0
        active = False
        start = 0.0

        if events[0].event_type == EventType.RTN:
            fill(tag, 0.0, events[0].timestamp)
            active = False
            idx = 1

        for event in events[idx:]:
            if event.event_type == EventType.ACT and not active:
                active = True
                start = event.timestamp
            elif event.event_type == EventType.RTN and active:
                fill(tag, start, event.timestamp)
                active = False
        if active:
            fill(tag, start, None)

    return x


def activation_sequence_from_series(matrix: np.ndarray) -> list[tuple[int, int]]:
    """Return ordered activation events ``(tag, time_index)`` from a series matrix."""

    x = _as_binary_matrix(matrix)
    events: list[tuple[int, int]] = []
    n_tags, n_time = x.shape
    for tag in range(n_tags):
        if x[tag, 0] == 1:
            events.append((tag, 0))
        rising = np.where(np.diff(x[tag].astype(np.int8)) == 1)[0] + 1
        for time_idx in rising:
            events.append((tag, int(time_idx)))
    events.sort(key=lambda item: (item[1], item[0]))
    return events


def activation_sequence_from_episode(episode: AlarmEpisode) -> list[tuple[int, float]]:
    """Return ordered activation events ``(tag, timestamp)`` from an episode."""

    return [
        (event.tag, event.timestamp)
        for event in sorted(episode.events, key=lambda e: (e.timestamp, e.order))
        if event.event_type == EventType.ACT
    ]


def active_set_vector(matrix: np.ndarray) -> np.ndarray:
    """Binary vector indicating which tags are active at least once."""

    x = _as_binary_matrix(matrix)
    return (x.max(axis=1) > 0).astype(np.int8)


def active_set_from_episode(episode: AlarmEpisode) -> set[int]:
    """Set of tags with at least one activation event in the episode."""

    return {event.tag for event in episode.events if event.event_type == EventType.ACT}

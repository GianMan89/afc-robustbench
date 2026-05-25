"""Perturbation functions for robustness benchmarking of alarm-event streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

import numpy as np

from afc_robustness.domain import AlarmEpisode, AlarmEvent, EventType
from afc_robustness.repair import TraceRepair


class Perturbation(Protocol):
    """Protocol implemented by all perturbation operators."""

    name: str

    def apply(
        self,
        episode: AlarmEpisode,
        severity: float,
        rng: np.random.Generator,
    ) -> AlarmEpisode:
        ...


class BasePerturbation:
    """Base class with common severity validation."""

    name: str = "base"

    @staticmethod
    def _validate_severity(severity: float) -> float:
        s = float(severity)
        if not 0 <= s < 1:
            raise ValueError("severity must satisfy 0 <= severity < 1")
        return s


@dataclass(frozen=True)
class MissingEvents(BasePerturbation):
    """Uniformly delete a severity-dependent number of events."""

    name: str = "missing_events"

    def apply(self, episode: AlarmEpisode, severity: float, rng: np.random.Generator) -> AlarmEpisode:
        s = self._validate_severity(severity)
        n = episode.n_events
        q = int(np.floor(s * n))
        if q <= 0:
            return episode
        remove = set(rng.choice(n, size=q, replace=False).tolist())
        events = [event for idx, event in enumerate(episode.events) if idx not in remove]
        return episode.with_events(events)


@dataclass(frozen=True)
class TagDropout(BasePerturbation):
    """Remove all events of a severity-dependent number of alarm tags."""

    name: str = "tag_dropout"

    def apply(self, episode: AlarmEpisode, severity: float, rng: np.random.Generator) -> AlarmEpisode:
        s = self._validate_severity(severity)
        k = int(np.floor(s * episode.n_tags))
        if k <= 0:
            return episode
        tags = set(rng.choice(episode.n_tags, size=k, replace=False).tolist())
        events = [event for event in episode.events if event.tag not in tags]
        return episode.with_events(events)


@dataclass(frozen=True)
class SpuriousEvents(BasePerturbation):
    """Insert random spurious events into the episode window."""

    name: str = "spurious_events"

    def apply(self, episode: AlarmEpisode, severity: float, rng: np.random.Generator) -> AlarmEpisode:
        s = self._validate_severity(severity)
        q = int(np.floor(s * episode.n_events))
        if q <= 0:
            return episode
        max_order = max((event.order for event in episode.events), default=-1)
        inserted: list[AlarmEvent] = []
        for offset in range(q):
            tag = int(rng.integers(0, episode.n_tags))
            event_type = EventType.ACT if rng.integers(0, 2) == 0 else EventType.RTN
            timestamp = float(rng.uniform(0.0, max(episode.horizon, 0.0)))
            inserted.append(AlarmEvent(tag, event_type, timestamp, max_order + offset + 1))
        return episode.with_events(tuple(episode.events) + tuple(inserted))


@dataclass(frozen=True)
class BurstSpuriousEvents(BasePerturbation):
    """Insert spurious events concentrated within a random temporal burst."""

    half_width: float = 2.0
    name: str = "spurious_burst"

    def apply(self, episode: AlarmEpisode, severity: float, rng: np.random.Generator) -> AlarmEpisode:
        s = self._validate_severity(severity)
        q = int(np.floor(s * episode.n_events))
        if q <= 0:
            return episode
        center = float(rng.uniform(0.0, max(episode.horizon, 0.0)))
        lo = max(0.0, center - self.half_width)
        hi = min(episode.horizon, center + self.half_width)
        max_order = max((event.order for event in episode.events), default=-1)
        inserted: list[AlarmEvent] = []
        for offset in range(q):
            tag = int(rng.integers(0, episode.n_tags))
            event_type = EventType.ACT if rng.integers(0, 2) == 0 else EventType.RTN
            timestamp = float(rng.uniform(lo, hi)) if hi > lo else lo
            inserted.append(AlarmEvent(tag, event_type, timestamp, max_order + offset + 1))
        return episode.with_events(tuple(episode.events) + tuple(inserted))


@dataclass(frozen=True)
class TimingUncertainty(BasePerturbation):
    """Randomly shift timestamps of a severity-dependent event subset."""

    max_shift: float = 3.0
    name: str = "timing_uncertainty"

    def apply(self, episode: AlarmEpisode, severity: float, rng: np.random.Generator) -> AlarmEpisode:
        s = self._validate_severity(severity)
        n = episode.n_events
        q = int(np.floor(s * n))
        if q <= 0:
            return episode
        selected = set(rng.choice(n, size=q, replace=False).tolist())
        events: list[AlarmEvent] = []
        for idx, event in enumerate(episode.events):
            if idx in selected:
                eps = float(rng.uniform(-s * self.max_shift, s * self.max_shift))
                timestamp = min(max(event.timestamp + eps, 0.0), episode.horizon)
                events.append(event.with_timestamp(timestamp))
            else:
                events.append(event)
        return episode.with_events(events)


@dataclass(frozen=True)
class LateEventDelay(BasePerturbation):
    """Pipeline-induced delayed onset by discarding the first fraction of events."""

    name: str = "late_event_delay"

    def apply(self, episode: AlarmEpisode, severity: float, rng: np.random.Generator) -> AlarmEpisode:
        del rng
        s = self._validate_severity(severity)
        n = episode.n_events
        if n <= 1:
            return episode
        delta = min(int(np.floor(s * n)), n - 1)
        if delta <= 0:
            return episode
        sorted_events = tuple(sorted(episode.events, key=lambda e: (e.timestamp, e.order)))
        theta = sorted_events[delta].timestamp
        retained = [
            AlarmEvent(event.tag, event.event_type, event.timestamp - theta, idx)
            for idx, event in enumerate(sorted_events[delta:])
        ]
        horizon = max(0.0, episode.horizon - theta)
        return episode.with_events(retained, horizon=horizon)


@dataclass(frozen=True)
class LateTimeDelay(BasePerturbation):
    """Pipeline-induced delayed onset by discarding the first fraction of duration."""

    name: str = "late_time_delay"

    def apply(self, episode: AlarmEpisode, severity: float, rng: np.random.Generator) -> AlarmEpisode:
        del rng
        s = self._validate_severity(severity)
        if episode.horizon <= 0:
            return episode
        theta = min(s * episode.horizon, episode.last_event_time if episode.events else episode.horizon)
        if theta <= 0:
            return episode
        retained_raw = [event for event in episode.events if event.timestamp >= theta - 1e-12]
        retained = [
            AlarmEvent(event.tag, event.event_type, max(0.0, event.timestamp - theta), idx)
            for idx, event in enumerate(retained_raw)
        ]
        horizon = max(0.0, episode.horizon - theta)
        return episode.with_events(retained, horizon=horizon)


@dataclass(frozen=True)
class MixedPerturbation(BasePerturbation):
    """Ordered composition of base perturbations with repair between stages."""

    stages: tuple[Perturbation, ...]
    repair: TraceRepair = field(default_factory=TraceRepair)
    name: str = "mixed"

    def __init__(
        self,
        stages: Iterable[Perturbation],
        name: str | None = None,
        repair: TraceRepair | None = None,
    ) -> None:
        stage_tuple = tuple(stages)
        if not stage_tuple:
            raise ValueError("mixed perturbation requires at least one stage")
        object.__setattr__(self, "stages", stage_tuple)
        object.__setattr__(self, "repair", TraceRepair() if repair is None else repair)
        if name is None:
            name = "mixed_" + "__".join(stage.name for stage in stage_tuple)
        object.__setattr__(self, "name", name)

    def apply(self, episode: AlarmEpisode, severity: float, rng: np.random.Generator) -> AlarmEpisode:
        s = self._validate_severity(severity)
        current = episode
        for stage in self.stages:
            current = stage.apply(self.repair.apply(current), s, rng)
        return current


PERTURBATION_REGISTRY: dict[str, type[BasePerturbation]] = {
    "missing_events": MissingEvents,
    "tag_dropout": TagDropout,
    "spurious_events": SpuriousEvents,
    "spurious_burst": BurstSpuriousEvents,
    "timing_uncertainty": TimingUncertainty,
    "late_event_delay": LateEventDelay,
    "late_time_delay": LateTimeDelay,
}


def build_perturbation(spec: str | dict[str, Any]) -> Perturbation:
    """Build a perturbation object from a string or configuration dictionary."""

    if isinstance(spec, str):
        if spec not in PERTURBATION_REGISTRY:
            raise KeyError(f"unknown perturbation type: {spec}")
        return PERTURBATION_REGISTRY[spec]()

    ptype = spec.get("type", spec.get("name"))
    if ptype == "mixed":
        stages = [build_perturbation(stage) for stage in spec["stages"]]
        return MixedPerturbation(stages=stages, name=spec.get("name"))

    if ptype not in PERTURBATION_REGISTRY:
        raise KeyError(f"unknown perturbation type: {ptype}")
    cls = PERTURBATION_REGISTRY[ptype]
    kwargs = {k: v for k, v in spec.items() if k not in {"type", "name"}}
    obj = cls(**kwargs)
    if "name" in spec and spec["name"] != getattr(obj, "name", None):
        object.__setattr__(obj, "name", spec["name"])
    return obj

"""Domain objects for alarm-event streams and extracted alarm-flood episodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterable, Sequence


class EventType(str, Enum):
    """Supported alarm transition types."""

    ACT = "ACT"
    RTN = "RTN"


@dataclass(frozen=True)
class AlarmEvent:
    """One time-stamped alarm event.

    Parameters
    ----------
    tag:
        Integer alarm-tag index in ``0, ..., n_tags - 1``.
    event_type:
        ``EventType.ACT`` for activation or ``EventType.RTN`` for return-to-normal.
    timestamp:
        Non-negative local episode timestamp.
    order:
        Stable ordering index used to break ties between equal timestamps.
    """

    tag: int
    event_type: EventType | str
    timestamp: float
    order: int = 0

    def __post_init__(self) -> None:
        if self.tag < 0:
            raise ValueError("tag must be non-negative")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        object.__setattr__(self, "event_type", EventType(self.event_type))
        object.__setattr__(self, "timestamp", float(self.timestamp))
        object.__setattr__(self, "order", int(self.order))

    def with_timestamp(self, timestamp: float) -> "AlarmEvent":
        return replace(self, timestamp=float(timestamp))

    def with_order(self, order: int) -> "AlarmEvent":
        return replace(self, order=int(order))


@dataclass(frozen=True)
class AlarmEpisode:
    """One extracted alarm-flood episode.

    The episode contains a finite event sequence, an observation horizon, and
    the number of configured alarm tags. Timestamps are local to the extracted
    episode start.
    """

    events: tuple[AlarmEvent, ...]
    horizon: float
    n_tags: int
    tag_names: tuple[str, ...] = ()
    sample_id: str = ""

    def __init__(
        self,
        events: Iterable[AlarmEvent],
        horizon: float,
        n_tags: int,
        tag_names: Sequence[str] | None = None,
        sample_id: str = "",
        sort: bool = True,
    ) -> None:
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        if n_tags <= 0:
            raise ValueError("n_tags must be positive")
        event_tuple = tuple(events)
        for event in event_tuple:
            if event.tag >= n_tags:
                raise ValueError(f"event tag {event.tag} outside n_tags={n_tags}")
            if event.timestamp > horizon + 1e-9:
                raise ValueError(
                    f"event timestamp {event.timestamp} exceeds episode horizon {horizon}"
                )
        if sort:
            event_tuple = tuple(sorted(event_tuple, key=lambda e: (e.timestamp, e.order)))
        if tag_names is None:
            tag_tuple = tuple(str(i) for i in range(n_tags))
        else:
            tag_tuple = tuple(tag_names)
            if len(tag_tuple) != n_tags:
                raise ValueError("tag_names length must equal n_tags")
        object.__setattr__(self, "events", event_tuple)
        object.__setattr__(self, "horizon", float(horizon))
        object.__setattr__(self, "n_tags", int(n_tags))
        object.__setattr__(self, "tag_names", tag_tuple)
        object.__setattr__(self, "sample_id", str(sample_id))

    @property
    def n_events(self) -> int:
        return len(self.events)

    @property
    def last_event_time(self) -> float:
        return self.events[-1].timestamp if self.events else 0.0

    def sorted(self) -> "AlarmEpisode":
        return AlarmEpisode(
            self.events,
            horizon=self.horizon,
            n_tags=self.n_tags,
            tag_names=self.tag_names,
            sample_id=self.sample_id,
            sort=True,
        )

    def with_events(
        self,
        events: Iterable[AlarmEvent],
        horizon: float | None = None,
        sample_id: str | None = None,
        sort: bool = True,
    ) -> "AlarmEpisode":
        return AlarmEpisode(
            events,
            horizon=self.horizon if horizon is None else horizon,
            n_tags=self.n_tags,
            tag_names=self.tag_names,
            sample_id=self.sample_id if sample_id is None else sample_id,
            sort=sort,
        )

    def empty_prefix(self, horizon: float = 0.0) -> "AlarmEpisode":
        return AlarmEpisode(
            (),
            horizon=horizon,
            n_tags=self.n_tags,
            tag_names=self.tag_names,
            sample_id=self.sample_id,
        )

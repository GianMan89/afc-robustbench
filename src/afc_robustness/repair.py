"""Trace repair for perturbed alarm-event streams."""

from __future__ import annotations

from dataclasses import dataclass

from afc_robustness.domain import AlarmEpisode, AlarmEvent, EventType


@dataclass(frozen=True)
class TraceRepair:
    """Deletion-based repair enforcing tag-wise ACT/RTN consistency.

    The repair operator sorts by timestamp with stable tie-breaking and scans
    events in chronological order. For each tag, retained events must alternate
    between ``ACT`` and ``RTN``. A leading ``RTN`` is optionally retained and
    interpreted as a tag that was active at the episode start.
    """

    allow_leading_rtn: bool = True

    def __call__(self, episode: AlarmEpisode) -> AlarmEpisode:
        return self.apply(episode)

    def apply(self, episode: AlarmEpisode) -> AlarmEpisode:
        active = [False] * episode.n_tags
        seen = [False] * episode.n_tags
        repaired: list[AlarmEvent] = []

        for event in sorted(episode.events, key=lambda e: (e.timestamp, e.order)):
            tag = event.tag
            keep = False
            if event.event_type == EventType.ACT:
                if not active[tag]:
                    keep = True
                    active[tag] = True
                    seen[tag] = True
            else:  # RTN
                if active[tag]:
                    keep = True
                    active[tag] = False
                    seen[tag] = True
                elif self.allow_leading_rtn and not seen[tag]:
                    keep = True
                    active[tag] = False
                    seen[tag] = True

            if keep:
                repaired.append(event.with_order(len(repaired)))

        return episode.with_events(repaired, sort=True)

import numpy as np

from afc_robustness.perturbations import (
    LateEventDelay,
    MissingEvents,
    MixedPerturbation,
    SpuriousEvents,
    TimingUncertainty,
)
from afc_robustness.repair import TraceRepair
from afc_robustness.representations import series_to_episode


def make_episode():
    X = np.zeros((3, 10), dtype=int)
    X[0, 1:5] = 1
    X[1, 3:8] = 1
    X[2, 6:] = 1
    return series_to_episode(X, dt=1.0)


def test_zero_severity_identity_for_missing():
    ep = make_episode()
    out = MissingEvents().apply(ep, 0.0, np.random.default_rng(1))
    assert out.events == ep.events
    assert out.horizon == ep.horizon


def test_spurious_events_increases_unrepaired_length():
    ep = make_episode()
    out = SpuriousEvents().apply(ep, 0.5, np.random.default_rng(1))
    assert out.n_events >= ep.n_events


def test_timing_uncertainty_keeps_horizon_and_event_count():
    ep = make_episode()
    out = TimingUncertainty(max_shift=2.0).apply(ep, 0.5, np.random.default_rng(1))
    assert out.n_events == ep.n_events
    assert out.horizon == ep.horizon
    assert all(0.0 <= e.timestamp <= ep.horizon for e in out.events)


def test_late_event_delay_keeps_nonempty_if_possible():
    ep = make_episode()
    out = LateEventDelay().apply(ep, 0.9, np.random.default_rng(1))
    assert out.n_events >= 1
    assert out.horizon <= ep.horizon
    assert out.events[0].timestamp == 0.0


def test_mixed_perturbation_with_repair_is_valid():
    ep = make_episode()
    mixed = MixedPerturbation([SpuriousEvents(), MissingEvents()])
    out = TraceRepair().apply(mixed.apply(ep, 0.5, np.random.default_rng(1)))
    # Validity proxy: no two consecutive retained events of a tag have same type,
    # except that a leading RTN is possible.
    by_tag = {}
    for event in out.events:
        prev = by_tag.get(event.tag)
        assert prev is None or prev != event.event_type
        by_tag[event.tag] = event.event_type

import numpy as np

from afc_robustness.domain import AlarmEpisode, AlarmEvent, EventType
from afc_robustness.repair import TraceRepair
from afc_robustness.representations import episode_to_series, series_to_episode


def test_series_episode_roundtrip_simple_interval():
    X = np.zeros((2, 6), dtype=int)
    X[0, 1:4] = 1
    X[1, 0:2] = 1
    episode = series_to_episode(X, dt=1.0, tag_names=["A", "B"])
    X_back = episode_to_series(episode, dt=1.0, max_length=6)
    assert np.array_equal(X, X_back)


def test_repair_removes_duplicate_act_and_allows_leading_rtn():
    episode = AlarmEpisode(
        [
            AlarmEvent(0, EventType.RTN, 1.0, 0),
            AlarmEvent(0, EventType.RTN, 2.0, 1),
            AlarmEvent(0, EventType.ACT, 3.0, 2),
            AlarmEvent(0, EventType.ACT, 4.0, 3),
            AlarmEvent(0, EventType.RTN, 5.0, 4),
        ],
        horizon=6.0,
        n_tags=1,
    )
    repaired = TraceRepair().apply(episode)
    assert [event.event_type for event in repaired.events] == [
        EventType.RTN,
        EventType.ACT,
        EventType.RTN,
    ]

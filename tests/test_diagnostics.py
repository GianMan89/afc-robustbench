import numpy as np

from afc_robustness.data import AlarmSeriesDataset
from afc_robustness.diagnostics import (
    aggregate_diagnostic_metrics,
    compare_episodes,
    compute_update_table,
    dataset_perturbation_metrics,
    prefix_boundary_table,
    prefix_inference_table,
    prefix_model_descriptions,
    prefix_training_plan,
)
from afc_robustness.online import OnlineEvaluationConfig


def _dataset():
    X = np.zeros((4, 3, 8), dtype=np.int8)
    X[0, 0, 1:4] = 1
    X[0, 1, 3:6] = 1
    X[1, 0, 2:5] = 1
    X[1, 2, 4:7] = 1
    X[2, 1, 1:3] = 1
    X[2, 2, 4:6] = 1
    X[3, 1, 2:5] = 1
    X[3, 2, 5:7] = 1
    y = np.array([0, 0, 1, 1])
    return AlarmSeriesDataset(
        X=X,
        y=y,
        class_names=("a", "b"),
        sample_ids=("a/0", "a/1", "b/0", "b/1"),
        tag_names=("T0", "T1", "T2"),
        lengths=np.array([8, 8, 8, 8]),
        dt=1.0,
        name="toy",
    )


def test_compare_and_dataset_metrics():
    dataset = _dataset()
    clean = dataset.episode(0)
    same = dataset.episode(0)
    metrics = compare_episodes(clean, same, dt=1.0)
    assert metrics["act_tag_set_jaccard"] == 1.0
    assert metrics["all_sequence_lcs_similarity"] == 1.0

    df = dataset_perturbation_metrics(
        dataset,
        [{"name": "missing", "type": "missing_events"}],
        severity_grid=[0.0, 0.5],
        n_draws=1,
        sample_indices=[0, 1],
        random_seed=1,
    )
    assert {"scenario", "severity", "variant", "act_count_delta"}.issubset(df.columns)
    agg = aggregate_diagnostic_metrics(df)
    assert not agg.empty
    assert "act_count_delta_mean" in agg.columns


def test_update_and_prefix_tables():
    dataset = _dataset()
    episode = dataset.episode(0)
    online = OnlineEvaluationConfig(progress_grid=(0.25, 0.5, 1.0), update_interval=1.0)
    updates = compute_update_table(episode, online)
    assert not updates.empty
    assert updates["event_count_progress"].max() <= 1.0

    bounds = prefix_boundary_table(episode, [0.25, 0.5, 1.0], reference="event_count")
    assert list(bounds["prefix_progress"]) == [0.25, 0.5, 1.0]
    assert bounds["n_events_used"].iloc[-1] == episode.n_events

    specs = [
        {
            "name": "wdi_1nn",
            "params_grid": [
                {
                    "template_threshold": 0.5,
                    "training_mode": "prefix",
                    "prefix_grid": "progress_grid",
                    "prefix_reference": "event_count",
                    "prefix_train_reference": "same",
                }
            ],
        }
    ]
    desc = prefix_model_descriptions(specs, online)
    assert not desc.empty
    plan = prefix_training_plan(dataset, [0, 1], specs, online)
    assert not plan.empty
    inference = prefix_inference_table(episode, specs, online)
    assert not inference.empty
    assert "selected_prefix_progress" in inference.columns

import pandas as pd

from afc_robustness.metrics import aggregate_all


def test_aggregate_all_shapes():
    predictions = pd.DataFrame(
        [
            {
                "dataset": "d",
                "fold": 0,
                "repeat": 0,
                "method": "m",
                "scenario": "p",
                "severity": 0.0,
                "draw": 0,
                "sample_id": "s1",
                "y_true": 0,
                "progress": 0.5,
                "y_pred": 0,
                "correct": 1,
            },
            {
                "dataset": "d",
                "fold": 0,
                "repeat": 0,
                "method": "m",
                "scenario": "p",
                "severity": 0.0,
                "draw": 0,
                "sample_id": "s1",
                "y_true": 0,
                "progress": 1.0,
                "y_pred": 1,
                "correct": 0,
            },
        ]
    )
    tables = aggregate_all(predictions)
    assert tables["degradation_profiles"]["score"].tolist() == [1.0, 0.0]
    assert tables["progress_auc"]["progress_auc"].iloc[0] == 0.5
    assert tables["scenario_scores"]["scenario_score"].iloc[0] == 0.5

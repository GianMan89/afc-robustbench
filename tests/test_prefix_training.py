import numpy as np

from afc_robustness.data import AlarmSeriesDataset
from afc_robustness.models.factory import make_model
from afc_robustness.online import OnlineEvaluationConfig, OnlineEvaluator
from afc_robustness.experiment import fit_model_with_context


def _toy_dataset():
    X = np.zeros((6, 3, 12), dtype=np.int8)
    y = np.array([0, 0, 0, 1, 1, 1])
    X[:3, 0, 1:6] = 1
    X[3:, 1, 2:9] = 1
    lengths = np.full(6, 12)
    return AlarmSeriesDataset(
        X=X,
        y=y,
        class_names=("a", "b"),
        sample_ids=tuple(f"s{i}" for i in range(6)),
        tag_names=("t0", "t1", "t2"),
        lengths=lengths,
        dt=1.0,
        name="toy",
    )


def test_prefix_training_wrapper_trains_one_model_per_grid_point():
    dataset = _toy_dataset()
    evaluator = OnlineEvaluator(
        OnlineEvaluationConfig(progress_grid=(0.5, 1.0), update_interval=3.0)
    )
    model = make_model(
        "wdi_1nn",
        {
            "template_threshold": 0.5,
            "training_mode": "prefix",
            "prefix_grid": "progress_grid",
            "prefix_reference": "online",
        },
    )
    fit_model_with_context(model, dataset, np.arange(dataset.n_episodes), evaluator, "act")
    assert model.name == "WDI-1NN-prefix"
    assert model.prefix_grid_ == (0.5, 1.0)
    assert sorted(model.models_) == [0.5, 1.0]

    episode = dataset.episode(0, initial_active_policy="act")
    trajectory = evaluator.evaluate_episode(model, episode)
    assert set(trajectory) == {0.5, 1.0}
    assert all(pred in {0, 1} for pred in trajectory.values())

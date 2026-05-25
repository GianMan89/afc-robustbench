import numpy as np

from afc_robustness.models import ACMSVM, CASIM, EAC1NN, Jaccard1NN, MBWLogisticRegression, WDI1NN


def make_data():
    rng = np.random.default_rng(0)
    X = np.zeros((12, 6, 20), dtype=int)
    y = np.repeat([0, 1, 2], 4)
    for i, cls in enumerate(y):
        start = 2 + cls * 3
        X[i, cls : cls + 2, start : start + 5] = 1
        if rng.random() < 0.4:
            X[i, rng.integers(0, 6), rng.integers(0, 18) : rng.integers(18, 20)] = 1
    return X, y


def test_models_predict_proba_shapes():
    X, y = make_data()
    models = [
        WDI1NN(template_threshold=0.4),
        Jaccard1NN(),
        EAC1NN(attenuation=0.001),
        MBWLogisticRegression(C=1.0, max_iter=200),
        ACMSVM(C=1.0),
        CASIM(num_features=32, n_estimators=1, backend="lite", random_state=0),
    ]
    for model in models:
        model.fit(X, y)
        proba = model.predict_proba(X[:2, :, :10])
        assert proba.shape == (2, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

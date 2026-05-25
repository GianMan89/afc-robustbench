"""CASIM wrapper with optional exact sktime/MultiRocket backend and lightweight fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from afc_robustness.models.base import AFCModel, ensure_3d

Backend = Literal["auto", "sktime", "lite"]


class _RandomConvolutionFeatures:
    """Small deterministic random-convolution feature extractor for smoke tests.

    This is a development fallback, not a substitute for the exact CASIM backend
    used for final manuscript-grade experiments.
    """

    def __init__(self, n_kernels: int = 128, random_state: int = 42) -> None:
        self.n_kernels = int(n_kernels)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray) -> "_RandomConvolutionFeatures":
        X = ensure_3d(X)
        rng = np.random.default_rng(self.random_state)
        _, n_tags, n_time = X.shape
        lengths = np.array([length for length in (3, 5, 7, 9) if length <= max(n_time, 3)])
        if len(lengths) == 0:
            lengths = np.array([min(3, n_time)])
        self.kernels_: list[dict[str, Any]] = []
        for _ in range(self.n_kernels):
            length = int(rng.choice(lengths))
            n_channels = int(rng.integers(1, min(n_tags, 3) + 1))
            channels = rng.choice(n_tags, size=n_channels, replace=False)
            weights = rng.normal(size=length)
            weights = weights - weights.mean()
            dilation = int(rng.integers(1, max(2, n_time // max(length, 1) + 1)))
            self.kernels_.append(
                {"channels": channels, "weights": weights, "dilation": dilation, "length": length}
            )
        return self

    def _apply_kernel(self, sample: np.ndarray, kernel: dict[str, Any]) -> np.ndarray:
        channels = kernel["channels"]
        weights = kernel["weights"]
        dilation = kernel["dilation"]
        length = kernel["length"]
        idx = np.arange(length) * dilation
        max_start = sample.shape[1] - int(idx[-1])
        if max_start <= 0:
            signal = sample[channels].mean(axis=0)
            return np.array([float(np.dot(signal[: min(len(signal), length)], weights[: min(len(signal), length)]))])
        signal = sample[channels].mean(axis=0)
        vals = np.empty(max_start, dtype=float)
        for start in range(max_start):
            vals[start] = float(np.dot(signal[start + idx], weights))
        return vals

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = ensure_3d(X)
        features = np.zeros((X.shape[0], 4 * len(self.kernels_)), dtype=float)
        for i, sample in enumerate(X.astype(float)):
            row = []
            for kernel in self.kernels_:
                vals = self._apply_kernel(sample, kernel)
                row.extend([vals.max(), vals.mean(), vals.min(), np.mean(vals > 0.0)])
            features[i] = row
        return np.nan_to_num(features)


@dataclass
class CASIM(AFCModel):
    """CASIM classifier.

    ``backend='sktime'`` uses the MultiRocket/Arsenal implementation if the
    optional ``sktime`` dependency is installed. ``backend='lite'`` uses a small
    deterministic random-convolution fallback intended for smoke tests.
    """

    num_features: int = 672
    n_estimators: int = 25
    n_jobs_multirocket: int = 1
    random_state: int = 42
    alphas: Any = None
    backend: Backend = "auto"
    name: str = "CASIM"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CASIM":
        X = ensure_3d(X)
        self.train_length_ = X.shape[2]
        self.classes_ = np.unique(y)
        backend = self.backend
        if backend in {"auto", "sktime"}:
            try:
                from afc_robustness.models.CASIM_arsenal import Arsenal  # noqa: N806

                alphas = np.logspace(-3, 3, 10) if self.alphas is None else self.alphas
                self.model_ = Arsenal(
                    num_features=self.num_features,
                    n_jobs_multirocket=self.n_jobs_multirocket,
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    alphas=alphas,
                )
                self.model_.fit(X, y)
                self.backend_ = "sktime"
                return self
            except Exception as exc:
                if backend == "sktime":
                    raise ImportError(
                        "CASIM sktime backend requested but could not be initialized. "
                        "Install with `pip install -e .[casim]`."
                    ) from exc

        self.extractor_ = _RandomConvolutionFeatures(
            n_kernels=max(8, self.num_features // 8), random_state=self.random_state
        ).fit(X)
        features = self.extractor_.transform(X)
        base = make_pipeline(
            StandardScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        )
        self.model_ = CalibratedClassifierCV(base, cv=3 if len(y) >= 9 else 2)
        self.model_.fit(features, y)
        self.classes_ = self.model_.classes_
        self.backend_ = "lite"
        return self

    def _pad_to_train_length(self, X: np.ndarray) -> np.ndarray:
        X = ensure_3d(X)
        if not hasattr(self, "train_length_"):
            return X
        if X.shape[2] == self.train_length_:
            return X
        if X.shape[2] > self.train_length_:
            return X[:, :, : self.train_length_]
        pad = self.train_length_ - X.shape[2]
        return np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="constant")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = self._pad_to_train_length(X)
        if getattr(self, "backend_", None) == "sktime":
            if hasattr(self.model_, "predict_proba"):
                return np.asarray(self.model_.predict_proba(X))
            return np.asarray(self.model_._predict_proba(X))
        features = self.extractor_.transform(X)
        return self.model_.predict_proba(features)

    def get_params(self) -> dict[str, Any]:
        return {
            "num_features": self.num_features,
            "n_estimators": self.n_estimators,
            "backend": self.backend,
        }

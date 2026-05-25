"""Modified bag-of-words vectorization with logistic regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from afc_robustness.models.base import AFCModel, ensure_3d
from afc_robustness.representations import activation_sequence_from_series


@dataclass
class MBWLogisticRegression(AFCModel):
    """Modified bag-of-words features followed by logistic regression."""

    C: float = 1.0
    penalty: str | None = "l2"
    solver: str = "lbfgs"
    fit_intercept: bool = True
    max_iter: int = 1000
    use_idf: bool = True
    use_time_weight: bool = True
    time_epsilon: float = 1.0
    name: str = "MBW-LR"

    def _counts_and_first_times(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = ensure_3d(X)
        n_samples, n_tags, n_time = X.shape
        counts = np.zeros((n_samples, n_tags), dtype=float)
        first = np.full((n_samples, n_tags), np.nan, dtype=float)
        for i, sample in enumerate(X):
            for tag, time_idx in activation_sequence_from_series(sample):
                counts[i, tag] += 1.0
                if np.isnan(first[i, tag]):
                    first[i, tag] = float(time_idx + self.time_epsilon)
        if n_time <= 0:
            n_time = 1
        first = np.where(np.isnan(first), np.inf, first)
        return counts, first

    def _fit_idf(self, counts: np.ndarray) -> np.ndarray:
        df = (counts > 0).sum(axis=0)
        return np.log((counts.shape[0] + 1.0) / (df + 1.0)) + 1.0

    def _transform(self, X: np.ndarray, *, fit: bool = False) -> np.ndarray:
        X = ensure_3d(X)
        counts, first = self._counts_and_first_times(X)
        total = counts.sum(axis=1, keepdims=True)
        tf = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)

        if fit or not hasattr(self, "idf_"):
            self.idf_ = self._fit_idf(counts)
        idf = self.idf_ if self.use_idf else np.ones(counts.shape[1], dtype=float)

        if self.use_time_weight:
            horizon = max(X.shape[2] - 1 + self.time_epsilon, self.time_epsilon)
            time_weight = np.zeros_like(tf)
            finite = np.isfinite(first)
            time_weight[finite] = np.log(
                (horizon + self.time_epsilon) / np.maximum(first[finite], self.time_epsilon)
            )
            time_weight = np.maximum(time_weight, 0.0)
        else:
            time_weight = np.ones_like(tf)

        features = tf * idf[None, :] * time_weight
        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MBWLogisticRegression":
        features = self._transform(X, fit=True)
        penalty = None if self.penalty in {None, "none", "None"} else self.penalty
        kwargs = {
            "C": self.C,
            "solver": self.solver,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
        }
        if penalty != "l2":
            kwargs["penalty"] = penalty
        self.clf_ = LogisticRegression(**kwargs)
        self.clf_.fit(features, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        features = self._transform(X, fit=False)
        return self.clf_.predict_proba(features)

    def get_params(self) -> dict[str, Any]:
        return {
            "C": self.C,
            "penalty": self.penalty,
            "solver": self.solver,
            "fit_intercept": self.fit_intercept,
            "use_idf": self.use_idf,
            "use_time_weight": self.use_time_weight,
        }

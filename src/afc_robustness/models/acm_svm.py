"""Alarm co-activation matrix with support vector machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from afc_robustness.models.base import AFCModel, ensure_3d


@dataclass
class ACMSVM(AFCModel):
    """Co-activation-matrix features with an SVM classifier."""

    C: float = 1.0
    kernel: str = "rbf"
    gamma: str | float = "scale"
    probability: bool = True
    random_state: int = 42
    name: str = "ACM-SVM"

    @staticmethod
    def coactivation_features(X: np.ndarray) -> np.ndarray:
        X = ensure_3d(X)
        A = (X == 1).astype(np.float64)
        n_samples, n_tags, _ = A.shape
        intersections = np.einsum("svt,swt->svw", A, A, optimize=True)
        sums = A.sum(axis=2)
        unions = sums[:, :, None] + sums[:, None, :] - intersections
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard = np.where(unions > 0, intersections / unions, 0.0)
        iu = np.triu_indices(n_tags, k=1)
        if len(iu[0]) == 0:
            return np.zeros((n_samples, 1), dtype=float)
        return jaccard[:, iu[0], iu[1]]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ACMSVM":
        features = self.coactivation_features(X)
        self.clf_ = make_pipeline(
            StandardScaler(),
            SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                probability=self.probability,
                random_state=self.random_state,
            ),
        )
        self.clf_.fit(features, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        features = self.coactivation_features(X)
        return self.clf_.predict_proba(features)

    def get_params(self) -> dict[str, Any]:
        return {"C": self.C, "kernel": self.kernel, "gamma": self.gamma}

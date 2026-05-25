"""Jaccard active-set 1-nearest-neighbor classifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from afc_robustness.models.base import AFCModel, ensure_3d, stable_softmax_from_scores
from afc_robustness.representations import active_set_vector


@dataclass
class Jaccard1NN(AFCModel):
    """1-NN over active alarm sets with Jaccard distance."""

    name: str = "JAC-1NN"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Jaccard1NN":
        X = ensure_3d(X)
        self.classes_ = np.unique(y)
        self.y_train_ = np.asarray(y)
        self.X_active_ = np.stack([active_set_vector(sample) for sample in X]).astype(bool)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = ensure_3d(X)
        active = np.stack([active_set_vector(sample) for sample in X]).astype(bool)
        min_dist = np.full((X.shape[0], len(self.classes_)), np.inf, dtype=float)
        for i, sample in enumerate(active):
            inter = np.logical_and(self.X_active_, sample).sum(axis=1)
            union = np.logical_or(self.X_active_, sample).sum(axis=1)
            similarity = np.where(union > 0, inter / union, 1.0)
            distance = 1.0 - similarity
            for cidx, cls in enumerate(self.classes_):
                class_dist = distance[self.y_train_ == cls]
                min_dist[i, cidx] = class_dist.min() if class_dist.size else np.inf
        return stable_softmax_from_scores(min_dist, higher_is_better=False)

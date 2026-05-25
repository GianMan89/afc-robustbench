"""Exponentially attenuated component 1-nearest-neighbor classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from afc_robustness.models.base import AFCModel, ensure_3d, stable_softmax_from_scores
from afc_robustness.representations import activation_sequence_from_series

TimeScale = Literal["event_index", "time_index"]
Distance = Literal["euclidean", "cosine"]


@dataclass
class EAC1NN(AFCModel):
    """Exponentially attenuated component representation with 1-NN classification.

    Each activation contributes ``exp(-attenuation * t)`` to its tag component,
    where ``t`` is either the event index in the activation sequence or the
    discrete time index.
    """

    attenuation: float = 0.001
    time_scale: TimeScale = "event_index"
    distance: Distance = "euclidean"
    normalize: bool = True
    name: str = "EAC-1NN"

    def _transform_one(self, sample: np.ndarray) -> np.ndarray:
        n_tags = sample.shape[0]
        vec = np.zeros(n_tags, dtype=float)
        seq = activation_sequence_from_series(sample)
        for event_index, (tag, time_idx) in enumerate(seq):
            t = event_index if self.time_scale == "event_index" else time_idx
            vec[tag] += float(np.exp(-self.attenuation * t))
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = ensure_3d(X)
        return np.stack([self._transform_one(sample) for sample in X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EAC1NN":
        self.classes_ = np.unique(y)
        self.y_train_ = np.asarray(y)
        self.X_feat_ = self._transform(X)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feat = self._transform(X)
        min_dist = np.full((feat.shape[0], len(self.classes_)), np.inf, dtype=float)
        for i, row in enumerate(feat):
            if self.distance == "cosine":
                denom = np.linalg.norm(self.X_feat_, axis=1) * max(np.linalg.norm(row), 1e-12)
                sim = np.divide(self.X_feat_ @ row, denom, out=np.zeros_like(denom), where=denom > 0)
                dist = 1.0 - sim
            else:
                dist = np.linalg.norm(self.X_feat_ - row, axis=1)
            for cidx, cls in enumerate(self.classes_):
                class_dist = dist[self.y_train_ == cls]
                min_dist[i, cidx] = class_dist.min() if class_dist.size else np.inf
        return stable_softmax_from_scores(min_dist, higher_is_better=False)

    def get_params(self) -> dict[str, object]:
        return {
            "attenuation": self.attenuation,
            "time_scale": self.time_scale,
            "distance": self.distance,
            "normalize": self.normalize,
        }

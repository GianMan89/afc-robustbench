"""Base interfaces for AFC models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


class AFCModel(ABC):
    """Minimal interface expected by the online benchmark.

    ``fit`` and ``predict_proba`` are the only methods that concrete AFC
    classifiers must implement. ``fit_with_context`` and ``predict_with_context``
    are optional extension hooks used by meta-estimators, for example the
    prefix-trained wrapper. Ordinary classifiers inherit the default behavior.
    """

    name: str = "AFCModel"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "AFCModel":
        """Fit the model on full or prefix alarm-series matrices."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities with columns aligned to ``self.classes_``."""

    def fit_with_context(self, X: np.ndarray, y: np.ndarray, **_: Any) -> "AFCModel":
        """Fit with optional experiment context.

        The default implementation intentionally ignores context and calls
        ``fit``. Meta-estimators can override this method to use dataset-level
        information such as original episode lengths, event streams, or the
        online progress grid.
        """

        return self.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "classes_"):
            raise AttributeError("model is not fitted; missing classes_ attribute")
        proba = self.predict_proba(X)
        return np.asarray(self.classes_)[np.argmax(proba, axis=1)]

    def predict_with_context(self, X: np.ndarray, **_: Any) -> np.ndarray:
        """Predict with optional online-prefix context.

        Ordinary classifiers ignore the context. Prefix-trained wrappers use it
        to choose the stage-specific classifier corresponding to the currently
        observed progress.
        """

        return self.predict(X)

    def get_params(self) -> dict[str, Any]:
        return {}


def ensure_3d(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError("X must have shape (n_samples, n_tags, n_time_steps)")
    return arr


def stable_softmax_from_scores(scores: np.ndarray, *, higher_is_better: bool = True) -> np.ndarray:
    logits = np.asarray(scores, dtype=float)
    if not higher_is_better:
        logits = -logits
    logits = logits - np.nanmax(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    denom = exp_logits.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return exp_logits / denom


@dataclass(frozen=True)
class ModelSpec:
    """Experiment configuration for one AFC model family."""

    name: str
    params_grid: tuple[dict[str, Any], ...]

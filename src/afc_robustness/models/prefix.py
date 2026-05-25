"""Prefix-trained meta-estimator for online AFC models."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np

from afc_robustness.data import AlarmSeriesDataset
from afc_robustness.domain import AlarmEpisode
from afc_robustness.models.base import AFCModel, ensure_3d
from afc_robustness.representations import episode_to_series

PrefixReference = Literal["online", "event_count", "relative_time", "sample_count"]
ConcretePrefixReference = Literal["event_count", "relative_time", "sample_count"]
PrefixSelection = Literal["left", "right", "nearest"]
PrefixTrainHorizon = Literal["prefix", "full"]


def _unique_sorted_grid(values: Sequence[float]) -> tuple[float, ...]:
    grid = sorted({round(float(v), 12) for v in values})
    if not grid:
        raise ValueError("prefix_grid must contain at least one value")
    for value in grid:
        if not 0.0 < value <= 1.0:
            raise ValueError("prefix_grid values must be in (0, 1]")
    return tuple(grid)


def _pad_prefix_matrices(
    matrices: list[np.ndarray],
    *,
    n_tags: int,
    target_length: int | None = None,
    min_length: int = 1,
) -> np.ndarray:
    """Pad variable-length prefix matrices to one 3-D training tensor."""

    if not matrices:
        raise ValueError("cannot pad an empty matrix list")
    inferred = max(int(mat.shape[1]) for mat in matrices)
    length = max(int(min_length), int(target_length) if target_length is not None else inferred)
    Xp = np.zeros((len(matrices), n_tags, length), dtype=np.int8)
    for i, mat in enumerate(matrices):
        if mat.shape[0] != n_tags:
            raise ValueError("all prefix matrices must have the same number of alarm tags")
        copy_len = min(length, int(mat.shape[1]))
        if copy_len > 0:
            Xp[i, :, :copy_len] = mat[:, :copy_len]
    return Xp


@dataclass
class PrefixTrainedAFCModel(AFCModel):
    """Train one base AFC classifier per online-prefix progress level.

    The wrapper is intentionally model-agnostic: the base classifier is still a
    normal ``AFCModel``. The wrapper only controls which prefix-specific clone is
    fitted and which clone is used for a given online prediction.
    """

    base_model_cls: Callable[..., AFCModel]
    base_params: dict[str, Any]
    prefix_grid: Sequence[float] | str | None = None
    prefix_reference: PrefixReference = "online"
    prefix_train_reference: ConcretePrefixReference | Literal["same"] = "same"
    prefix_selection: PrefixSelection = "left"
    prefix_train_horizon: PrefixTrainHorizon = "prefix"
    prefix_min_time_steps: int = 1
    include_full_prefix: bool = True
    name: str = "PrefixTrainedAFCModel"

    def __post_init__(self) -> None:
        self.base_params = dict(self.base_params)
        self.prefix_reference = str(self.prefix_reference)  # type: ignore[assignment]
        self.prefix_train_reference = str(self.prefix_train_reference)  # type: ignore[assignment]
        self.prefix_selection = str(self.prefix_selection)  # type: ignore[assignment]
        self.prefix_train_horizon = str(self.prefix_train_horizon)  # type: ignore[assignment]
        if self.prefix_reference not in {"online", "event_count", "relative_time", "sample_count"}:
            raise ValueError("prefix_reference must be online, event_count, relative_time, or sample_count")
        if self.prefix_train_reference not in {"same", "event_count", "relative_time", "sample_count"}:
            raise ValueError("prefix_train_reference must be same, event_count, relative_time, or sample_count")
        if self.prefix_selection not in {"left", "right", "nearest"}:
            raise ValueError("prefix_selection must be left, right, or nearest")
        if self.prefix_train_horizon not in {"prefix", "full"}:
            raise ValueError("prefix_train_horizon must be prefix or full")
        self.prefix_min_time_steps = max(1, int(self.prefix_min_time_steps))

        base_name = getattr(self._make_base_model(), "name", self.base_model_cls.__name__)
        self.name = f"{base_name}-prefix"

    def _make_base_model(self) -> AFCModel:
        return self.base_model_cls(**copy.deepcopy(self.base_params))

    def _resolve_grid(self, online_config: Any | None = None) -> tuple[float, ...]:
        if self.prefix_grid is None or self.prefix_grid == "progress_grid":
            if online_config is not None and hasattr(online_config, "progress_grid"):
                grid = tuple(float(v) for v in online_config.progress_grid)
            else:
                grid = tuple(np.round(np.linspace(0.1, 1.0, 10), 3))
        elif isinstance(self.prefix_grid, str):
            if self.prefix_grid in {"deciles", "default"}:
                grid = tuple(np.round(np.linspace(0.1, 1.0, 10), 3))
            else:
                raise ValueError(
                    "prefix_grid must be a sequence of floats, 'progress_grid', 'deciles', or None"
                )
        else:
            grid = tuple(float(v) for v in self.prefix_grid)

        if self.include_full_prefix and 1.0 not in {round(v, 12) for v in grid}:
            grid = (*grid, 1.0)
        return _unique_sorted_grid(grid)

    def _resolve_training_reference(self, online_config: Any | None = None) -> ConcretePrefixReference:
        if self.prefix_train_reference != "same":
            return self.prefix_train_reference  # type: ignore[return-value]
        if self.prefix_reference == "online":
            if online_config is not None and hasattr(online_config, "progress_reference"):
                return str(online_config.progress_reference)  # type: ignore[return-value]
            return "relative_time"
        return self.prefix_reference  # type: ignore[return-value]

    def _prefix_episode(self, episode: AlarmEpisode, rho: float, reference: ConcretePrefixReference) -> AlarmEpisode:
        rho = min(max(float(rho), 0.0), 1.0)
        if reference == "relative_time":
            horizon = min(episode.horizon, rho * episode.horizon)
            events = [event for event in episode.events if event.timestamp <= horizon + 1e-12]
            return episode.with_events(events, horizon=horizon)

        if reference == "event_count":
            if episode.n_events == 0:
                return episode.empty_prefix(horizon=0.0)
            k = max(1, min(episode.n_events, int(np.ceil(rho * episode.n_events))))
            events = episode.events[:k]
            horizon = episode.horizon if rho >= 1.0 - 1e-12 else events[-1].timestamp
            return episode.with_events(events, horizon=horizon)

        raise ValueError("_prefix_episode only supports relative_time and event_count")

    def _prefix_tensor_from_dataset(
        self,
        dataset: AlarmSeriesDataset,
        indices: np.ndarray,
        rho: float,
        reference: ConcretePrefixReference,
        *,
        initial_active_policy: str,
        fallback_X: np.ndarray,
    ) -> np.ndarray:
        if reference == "sample_count":
            matrices = []
            for idx in indices:
                mat = dataset.matrix(int(idx), padded=False)
                length = max(1, int(np.ceil(float(rho) * mat.shape[1])))
                matrices.append(mat[:, :length])
        else:
            matrices = []
            for idx in indices:
                episode = dataset.episode(int(idx), initial_active_policy=initial_active_policy)
                prefix = self._prefix_episode(episode, float(rho), reference)
                matrices.append(episode_to_series(prefix, dt=dataset.dt))

        target_length = fallback_X.shape[2] if self.prefix_train_horizon == "full" else None
        return _pad_prefix_matrices(
            matrices,
            n_tags=dataset.n_tags,
            target_length=target_length,
            min_length=self.prefix_min_time_steps,
        )

    def _prefix_tensor_from_matrix(self, X: np.ndarray, rho: float) -> np.ndarray:
        X = ensure_3d(X)
        length = max(self.prefix_min_time_steps, int(np.ceil(float(rho) * X.shape[2])))
        length = min(length, X.shape[2]) if self.prefix_train_horizon == "prefix" else X.shape[2]
        return X[:, :, :length]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrefixTrainedAFCModel":
        """Context-free fallback: train prefixes by slicing the matrix time axis."""

        return self.fit_with_context(X, y)

    def fit_with_context(self, X: np.ndarray, y: np.ndarray, **context: Any) -> "PrefixTrainedAFCModel":
        X = ensure_3d(X)
        y = np.asarray(y)
        dataset = context.get("dataset")
        indices = context.get("indices")
        online_config = context.get("online_config")
        initial_active_policy = str(context.get("initial_active_policy", "act"))

        self.prefix_grid_ = self._resolve_grid(online_config)
        self.prefix_reference_ = self._resolve_training_reference(online_config)
        self.models_: dict[float, AFCModel] = {}
        self.train_lengths_: dict[float, int] = {}

        use_dataset_context = dataset is not None and indices is not None
        indices_arr = np.asarray(indices, dtype=int) if indices is not None else None

        for rho in self.prefix_grid_:
            if use_dataset_context:
                X_prefix = self._prefix_tensor_from_dataset(
                    dataset,
                    indices_arr,  # type: ignore[arg-type]
                    rho,
                    self.prefix_reference_,
                    initial_active_policy=initial_active_policy,
                    fallback_X=X,
                )
            else:
                X_prefix = self._prefix_tensor_from_matrix(X, rho)

            model = self._make_base_model()
            model.fit(X_prefix, y)
            self.models_[float(rho)] = model
            self.train_lengths_[float(rho)] = int(X_prefix.shape[2])

        self.classes_ = np.unique(y)
        self.default_prefix_ = max(self.prefix_grid_)
        return self

    def _progress_from_context(self, **context: Any) -> float:
        if self.prefix_reference == "online":
            return float(context.get("progress", 1.0))
        if self.prefix_reference == "event_count":
            return float(context.get("event_count_progress", context.get("progress", 1.0)))
        if self.prefix_reference == "relative_time":
            return float(context.get("relative_time_progress", context.get("progress", 1.0)))
        if self.prefix_reference == "sample_count":
            return float(context.get("sample_count_progress", context.get("progress", 1.0)))
        return float(context.get("progress", 1.0))

    def _select_prefix(self, progress: float) -> float:
        if not hasattr(self, "prefix_grid_"):
            raise AttributeError("prefix-trained model is not fitted")
        grid = np.asarray(self.prefix_grid_, dtype=float)
        progress = min(max(float(progress), 0.0), 1.0)

        if self.prefix_selection == "nearest":
            return float(grid[int(np.argmin(np.abs(grid - progress)))])
        if self.prefix_selection == "right":
            idx = int(np.searchsorted(grid, progress, side="left"))
            idx = min(idx, len(grid) - 1)
            return float(grid[idx])

        # Causal default: use the most mature model whose training progress does
        # not exceed the observed progress. Before the first grid point, fall back
        # to the earliest available prefix model.
        idx = int(np.searchsorted(grid, progress, side="right")) - 1
        idx = max(0, min(idx, len(grid) - 1))
        return float(grid[idx])

    def _selected_model(self, progress: float | None = None) -> AFCModel:
        if progress is None:
            rho = getattr(self, "default_prefix_", 1.0)
        else:
            rho = self._select_prefix(progress)
        return self.models_[float(rho)]

    def predict_with_context(self, X: np.ndarray, **context: Any) -> np.ndarray:
        progress = self._progress_from_context(**context)
        model = self._selected_model(progress)
        return model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities from the full-prefix classifier when no context exists."""

        return self._selected_model(None).predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._selected_model(None).predict(X)

    def get_params(self) -> dict[str, Any]:
        return {
            **copy.deepcopy(self.base_params),
            "training_mode": "prefix",
            "prefix_grid": list(self.prefix_grid_) if hasattr(self, "prefix_grid_") else self.prefix_grid,
            "prefix_reference": self.prefix_reference,
            "prefix_train_reference": self.prefix_train_reference,
            "prefix_selection": self.prefix_selection,
            "prefix_train_horizon": self.prefix_train_horizon,
            "prefix_min_time_steps": self.prefix_min_time_steps,
        }

"""Dataset loading utilities for binary alarm-series CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from afc_robustness.domain import AlarmEpisode
from afc_robustness.representations import series_to_episode

DEFAULT_TIME_COLUMNS = ("Minutes", "minutes", "time", "Time", "timestamp", "Timestamp", "t", "T")


@dataclass(frozen=True)
class SeriesCsvRecord:
    """One loaded alarm-series CSV file."""

    matrix: np.ndarray
    tag_names: tuple[str, ...]
    time_points: np.ndarray | None
    sample_id: str


@dataclass(frozen=True)
class AlarmSeriesDataset:
    """Padded binary alarm-series dataset.

    ``X`` has shape ``(n_episodes, n_tags, n_time_steps_max)``. ``lengths`` stores
    the unpadded length of each episode.
    """

    X: np.ndarray
    y: np.ndarray
    class_names: tuple[str, ...]
    sample_ids: tuple[str, ...]
    tag_names: tuple[str, ...]
    lengths: np.ndarray
    dt: float = 1.0
    name: str = "dataset"

    def __post_init__(self) -> None:
        if self.X.ndim != 3:
            raise ValueError("X must have shape (n_episodes, n_tags, n_time_steps)")
        if len(self.y) != self.X.shape[0]:
            raise ValueError("y length must match number of episodes")
        if len(self.sample_ids) != self.X.shape[0]:
            raise ValueError("sample_ids length must match number of episodes")
        if len(self.tag_names) != self.X.shape[1]:
            raise ValueError("tag_names length must match number of alarm tags")
        if len(self.lengths) != self.X.shape[0]:
            raise ValueError("lengths length must match number of episodes")

    @property
    def n_episodes(self) -> int:
        return self.X.shape[0]

    @property
    def n_tags(self) -> int:
        return self.X.shape[1]

    @property
    def n_time_steps(self) -> int:
        return self.X.shape[2]

    def subset(self, indices: Sequence[int]) -> "AlarmSeriesDataset":
        idx = np.asarray(indices, dtype=int)
        return AlarmSeriesDataset(
            X=self.X[idx],
            y=self.y[idx],
            class_names=self.class_names,
            sample_ids=tuple(self.sample_ids[i] for i in idx),
            tag_names=self.tag_names,
            lengths=self.lengths[idx],
            dt=self.dt,
            name=self.name,
        )

    def matrix(self, index: int, *, padded: bool = False) -> np.ndarray:
        if padded:
            return self.X[index]
        return self.X[index, :, : self.lengths[index]]

    def episode(self, index: int, *, initial_active_policy: str = "act") -> AlarmEpisode:
        matrix = self.matrix(index, padded=False)
        length = int(self.lengths[index])
        time_points = np.arange(length, dtype=float) * self.dt
        return series_to_episode(
            matrix,
            tag_names=self.tag_names,
            sample_id=self.sample_ids[index],
            dt=self.dt,
            time_points=time_points,
            initial_active_policy=initial_active_policy,  # type: ignore[arg-type]
        )


def load_run_csv(
    csv_path: str | Path,
    *,
    time_columns: Sequence[str] = DEFAULT_TIME_COLUMNS,
    transpose: bool = False,
    binary_threshold: float = 0.5,
) -> SeriesCsvRecord:
    """Load one alarm-series CSV as ``(n_tags, n_time_steps)``.

    A column with a name in ``time_columns`` is treated as the time axis and
    removed from the alarm matrix. Values are coerced to numeric and binarized.
    """

    path = Path(csv_path)
    df = pd.read_csv(path)
    time_points: np.ndarray | None = None

    time_col = next((col for col in time_columns if col in df.columns), None)
    if time_col is not None:
        time_points = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        df = df.drop(columns=[time_col])
        if len(time_points) > 0:
            time_points = time_points - time_points[0]

    numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    arr = (numeric.to_numpy(dtype=float) > binary_threshold).astype(np.int8)
    tag_names = tuple(str(col) for col in numeric.columns)

    if transpose:
        # Input CSV is already organized as rows=tags, columns=time. In this
        # case the CSV column names usually denote time indices, so keep the
        # matrix orientation and replace tag names by row identifiers.
        tag_names = tuple(str(i) for i in range(arr.shape[0]))
    else:
        # Default CSV layout: rows=time, columns=tags.
        arr = arr.T

    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"invalid alarm matrix in {path}")

    return SeriesCsvRecord(
        matrix=arr,
        tag_names=tag_names,
        time_points=time_points,
        sample_id=path.stem,
    )


def _pad_matrices(matrices: list[np.ndarray], max_length: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([m.shape[1] for m in matrices], dtype=int)
    n_tags = matrices[0].shape[0]
    target_len = int(max_length) if max_length is not None else int(lengths.max())
    X = np.zeros((len(matrices), n_tags, target_len), dtype=np.int8)
    clipped_lengths = np.minimum(lengths, target_len)
    for i, mat in enumerate(matrices):
        if mat.shape[0] != n_tags:
            raise ValueError("all matrices must have the same number of alarm tags")
        length = int(clipped_lengths[i])
        X[i, :, :length] = mat[:, :length]
    return X, clipped_lengths


def load_dataset_from_folders(
    base_path: str | Path,
    *,
    name: str | None = None,
    dt: float = 1.0,
    max_length: int | None = None,
    binary_threshold: float = 0.5,
) -> AlarmSeriesDataset:
    """Load a dataset organized as one class subfolder per label."""

    root = Path(base_path)
    if not root.exists():
        raise FileNotFoundError(root)

    class_dirs = sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    if not class_dirs:
        raise ValueError(f"no class subfolders found under {root}")

    matrices: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    reference_tags: tuple[str, ...] | None = None

    for label, class_dir in enumerate(class_dirs):
        csv_files = sorted(class_dir.glob("*.csv"))
        if not csv_files:
            continue
        for csv_file in csv_files:
            record = load_run_csv(csv_file, binary_threshold=binary_threshold)
            if reference_tags is None:
                reference_tags = record.tag_names
            elif record.tag_names != reference_tags:
                raise ValueError(
                    f"tag columns in {csv_file} do not match the first loaded file; "
                    "align the columns before running the benchmark"
                )
            matrices.append(record.matrix)
            labels.append(label)
            sample_ids.append(f"{class_dir.name}/{csv_file.stem}")

    if not matrices:
        raise ValueError(f"no CSV files found under {root}")

    X, lengths = _pad_matrices(matrices, max_length=max_length)
    y = np.asarray(labels, dtype=int)
    return AlarmSeriesDataset(
        X=X,
        y=y,
        class_names=tuple(p.name for p in class_dirs),
        sample_ids=tuple(sample_ids),
        tag_names=reference_tags or tuple(str(i) for i in range(X.shape[1])),
        lengths=lengths,
        dt=dt,
        name=name or root.name,
    )


def describe_dataset(dataset: AlarmSeriesDataset) -> pd.DataFrame:
    """Return a compact class-level dataset summary."""

    rows = []
    for label, class_name in enumerate(dataset.class_names):
        mask = dataset.y == label
        rows.append(
            {
                "class_label": label,
                "class_name": class_name,
                "n_episodes": int(mask.sum()),
                "min_length": int(dataset.lengths[mask].min()) if mask.any() else 0,
                "max_length": int(dataset.lengths[mask].max()) if mask.any() else 0,
                "mean_length": float(dataset.lengths[mask].mean()) if mask.any() else 0.0,
            }
        )
    return pd.DataFrame(rows)

"""Configuration-driven robustness benchmark runner."""

from __future__ import annotations

import contextlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from tqdm import tqdm

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None

from afc_robustness.data import AlarmSeriesDataset, load_dataset_from_folders
from afc_robustness.metrics import PREDICTION_COLUMNS, aggregate_all
from afc_robustness.models.factory import make_model
from afc_robustness.online import OnlineEvaluationConfig, OnlineEvaluator
from afc_robustness.perturbations import Perturbation, build_perturbation
from afc_robustness.repair import TraceRepair


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse bool-like YAML values robustly."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _log(enabled: bool, message: str) -> None:
    """Print compact status messages only when verbose mode is enabled."""
    if enabled:
        print(message)


@contextlib.contextmanager
def _tqdm_joblib(tqdm_object: tqdm):
    """Update a tqdm progress bar when joblib parallel batches finish."""
    old_callback = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(old_callback):
        def __call__(self, *args: Any, **kwargs: Any):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

@dataclass(frozen=True)
class BenchmarkConfig:
    """Parsed experiment configuration.

    Relative paths in the YAML file are resolved against the repository root
    inferred from ``config_path``. This makes notebooks robust to being executed
    from ``notebooks/`` or from the repository root.
    """

    raw: dict[str, Any]
    config_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BenchmarkConfig":
        """Load a YAML configuration and remember its file location."""
        import yaml

        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as f:
            return cls(raw=yaml.safe_load(f), config_path=config_path)

    @property
    def project_root(self) -> Path:
        """Repository root used to resolve relative paths."""
        explicit_root = self.raw.get("project_root")
        if explicit_root is not None:
            root = Path(explicit_root).expanduser()
            if not root.is_absolute() and self.config_path is not None:
                root = self.config_path.parent / root
            return root.resolve()

        start = self.config_path.parent if self.config_path is not None else Path.cwd()
        for candidate in [start, *start.parents]:
            if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
                return candidate.resolve()
        return start.resolve()
    
    @property
    def parallel_config(self) -> dict[str, Any]:
        """Parallel execution settings.

        Top-level ``parallel`` is preferred. For convenience, ``benchmark.parallel``
        and ``benchmark.n_jobs`` are also supported.
        """
        benchmark = dict(self.raw.get("benchmark", {}))
        parallel = dict(benchmark.get("parallel", {}))
        parallel.update(dict(self.raw.get("parallel", {})))

        if "n_jobs" not in parallel and "n_jobs" in benchmark:
            parallel["n_jobs"] = benchmark["n_jobs"]

        return parallel

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a possibly relative path against ``project_root``."""
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.project_root / path).resolve()

    @property
    def random_seed(self) -> int:
        return int(self.raw.get("random_seed", 42))
    
    @property
    def verbose(self) -> bool:
        """Whether to show compact benchmark progress information."""
        benchmark = self.raw.get("benchmark", {})
        return _as_bool(self.raw.get("verbose", benchmark.get("verbose", False)))

    @property
    def output_dir(self) -> Path:
        return self.resolve_path(self.raw.get("output_dir", "results/default"))

    @property
    def dataset_config(self) -> dict[str, Any]:
        dataset = dict(self.raw["dataset"])
        if "path" in dataset:
            dataset["path"] = str(self.resolve_path(dataset["path"]))
        return dataset

    @property
    def cv_config(self) -> dict[str, Any]:
        return dict(self.raw.get("cv", {}))

    @property
    def benchmark_config(self) -> dict[str, Any]:
        return dict(self.raw.get("benchmark", {}))

    @property
    def model_specs(self) -> list[dict[str, Any]]:
        return list(self.raw.get("models", []))

    @property
    def perturbation_specs(self) -> list[dict[str, Any]]:
        return list(self.raw.get("perturbations", []))


def expand_param_grid(grid: list[dict[str, Any]] | dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    """Expand a compact parameter grid into a list of dictionaries."""

    if grid is None:
        return [{}]
    if isinstance(grid, list):
        return [dict(item) for item in grid] if grid else [{}]
    if isinstance(grid, dict):
        keys = list(grid)
        values = [grid[key] if isinstance(grid[key], list) else [grid[key]] for key in keys]
        return [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*values)]
    raise TypeError("params_grid must be a list of dicts, dict of lists, or None")


def load_dataset_from_config(config: BenchmarkConfig) -> AlarmSeriesDataset:
    ds = config.dataset_config
    return load_dataset_from_folders(
        ds["path"],
        name=ds.get("name"),
        dt=float(ds.get("dt", 1.0)),
        max_length=ds.get("max_length"),
        binary_threshold=float(ds.get("binary_threshold", 0.5)),
    )


def build_online_config(config: BenchmarkConfig, dataset: AlarmSeriesDataset) -> OnlineEvaluationConfig:
    bcfg = config.benchmark_config
    progress_grid = tuple(float(v) for v in bcfg.get("progress_grid", np.round(np.linspace(0.1, 1.0, 10), 3)))
    return OnlineEvaluationConfig(
        progress_grid=progress_grid,
        update_interval=float(bcfg.get("update_interval", dataset.dt)),
        update_start=(None if bcfg.get("update_start") is None else float(bcfg.get("update_start"))),
        progress_reference=bcfg.get("progress_reference", "event_count"),
        include_zero_update=bool(bcfg.get("include_zero_update", True)),
        dt=float(bcfg.get("dt", dataset.dt)),
    )


def fit_model_with_context(
    model,
    dataset: AlarmSeriesDataset,
    train_indices: np.ndarray,
    evaluator: OnlineEvaluator,
    initial_active_policy: str,
):
    """Fit a model using dataset context when the model supports it.

    Ordinary classifiers ignore the extra context and behave exactly as before.
    Prefix-trained meta-estimators use the context to construct training prefixes
    that are aligned with the online evaluation configuration.
    """

    idx = np.asarray(train_indices, dtype=int)
    return model.fit_with_context(
        dataset.X[idx],
        dataset.y[idx],
        dataset=dataset,
        indices=idx,
        online_config=evaluator.config,
        initial_active_policy=initial_active_policy,
    )


def _evaluate_clean_validation(
    model,
    dataset: AlarmSeriesDataset,
    val_indices: np.ndarray,
    evaluator: OnlineEvaluator,
    initial_active_policy: str,
) -> float:
    y_true: list[int] = []
    y_pred: list[int] = []
    for idx in val_indices:
        episode = dataset.episode(int(idx), initial_active_policy=initial_active_policy)
        traj = evaluator.evaluate_episode(model, episode)
        for pred in traj.values():
            y_true.append(int(dataset.y[idx]))
            y_pred.append(int(pred))
    return float(accuracy_score(y_true, y_pred)) if y_true else 0.0


def select_hyperparameters(
    model_name: str,
    params_grid: list[dict[str, Any]],
    dataset: AlarmSeriesDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    evaluator: OnlineEvaluator,
    initial_active_policy: str,
) -> tuple[dict[str, Any], float]:
    """Select hyperparameters on clean validation trajectories."""

    best_params: dict[str, Any] = {}
    best_score = -np.inf
    for params in params_grid:
        model = make_model(model_name, params)
        fit_model_with_context(
            model,
            dataset,
            train_indices,
            evaluator,
            initial_active_policy,
        )
        score = _evaluate_clean_validation(
            model, dataset, val_indices, evaluator, initial_active_policy
        )
        if score > best_score:
            best_score = score
            best_params = dict(params)
    return best_params, float(best_score)


def _run_fold_model_job(
    *,
    config: BenchmarkConfig,
    dataset: AlarmSeriesDataset,
    split_no: int,
    n_splits: int,
    development_indices: np.ndarray,
    test_indices: np.ndarray,
    model_spec: dict[str, Any],
    severity_grid: list[float],
    n_draws: int,
    val_fraction: float,
    use_validation: bool,
    inner_max_num_threads: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one fold/model task.

    The task owns model selection, final fitting, and all perturbation evaluations
    for one model on one outer CV split.
    """
    if inner_max_num_threads is not None and threadpool_limits is not None:
        with threadpool_limits(limits=inner_max_num_threads):
            return _run_fold_model_job_impl(
                config=config,
                dataset=dataset,
                split_no=split_no,
                n_splits=n_splits,
                development_indices=development_indices,
                test_indices=test_indices,
                model_spec=model_spec,
                severity_grid=severity_grid,
                n_draws=n_draws,
                val_fraction=val_fraction,
                use_validation=use_validation,
            )

    return _run_fold_model_job_impl(
        config=config,
        dataset=dataset,
        split_no=split_no,
        n_splits=n_splits,
        development_indices=development_indices,
        test_indices=test_indices,
        model_spec=model_spec,
        severity_grid=severity_grid,
        n_draws=n_draws,
        val_fraction=val_fraction,
        use_validation=use_validation,
    )


def _run_fold_model_job_impl(
    *,
    config: BenchmarkConfig,
    dataset: AlarmSeriesDataset,
    split_no: int,
    n_splits: int,
    development_indices: np.ndarray,
    test_indices: np.ndarray,
    model_spec: dict[str, Any],
    severity_grid: list[float],
    n_draws: int,
    val_fraction: float,
    use_validation: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Implementation of one independent fold/model benchmark task."""
    bcfg = config.benchmark_config
    repeat = split_no // n_splits
    fold = split_no % n_splits

    initial_active_policy = str(bcfg.get("initial_active_policy", "act"))
    repair = TraceRepair(allow_leading_rtn=bool(bcfg.get("allow_leading_rtn", True)))
    evaluator = OnlineEvaluator(build_online_config(config, dataset))
    perturbations: list[Perturbation] = [
        build_perturbation(spec) for spec in config.perturbation_specs
    ]

    if use_validation:
        train_indices, val_indices = train_test_split(
            development_indices,
            test_size=val_fraction,
            stratify=dataset.y[development_indices],
            random_state=config.random_seed + split_no,
        )
    else:
        train_indices = development_indices
        val_indices = np.array([], dtype=int)

    model_name = str(model_spec["name"])
    params_grid = expand_param_grid(model_spec.get("params_grid"))
    method_label = getattr(make_model(model_name, params_grid[0]), "name", model_name)

    if use_validation:
        selected_params, val_score = select_hyperparameters(
            model_name,
            params_grid,
            dataset,
            train_indices,
            val_indices,
            evaluator,
            initial_active_policy,
        )
        selection_mode = "clean_validation"
    else:
        selected_params = dict(params_grid[0])
        val_score = np.nan
        selection_mode = "first_grid_entry_no_validation"

    selected_rows = [
        {
            "dataset": dataset.name,
            "fold": fold,
            "repeat": repeat,
            "method": method_label,
            "model_key": model_name,
            "validation_score": val_score,
            "selection_mode": selection_mode,
            "selected_params": json.dumps(selected_params, sort_keys=True),
        }
    ]

    model = make_model(model_name, selected_params)
    fit_model_with_context(
        model,
        dataset,
        train_indices,
        evaluator,
        initial_active_policy,
    )

    prediction_rows: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(perturbations):
        for severity in severity_grid:
            for draw in range(n_draws):
                seed_base = (
                    config.random_seed
                    + 1_000_003 * split_no
                    + 10_007 * draw
                    + 1_009 * int(round(severity * 1000))
                    + 101 * scenario_index
                )
                rng = np.random.default_rng(seed_base)

                for sample_index in test_indices:
                    sample_index_int = int(sample_index)
                    clean_episode = dataset.episode(
                        sample_index_int,
                        initial_active_policy=initial_active_policy,
                    )
                    perturbed = scenario.apply(clean_episode, severity, rng)
                    repaired = repair.apply(perturbed)
                    trajectory = evaluator.evaluate_episode(model, repaired)

                    y_true = int(dataset.y[sample_index_int])
                    sample_id = dataset.sample_ids[sample_index_int]

                    for progress, y_hat in trajectory.items():
                        prediction_rows.append(
                            {
                                "dataset": dataset.name,
                                "fold": fold,
                                "repeat": repeat,
                                "method": method_label,
                                "scenario": scenario.name,
                                "severity": severity,
                                "draw": draw,
                                "sample_id": sample_id,
                                "y_true": y_true,
                                "progress": progress,
                                "y_pred": int(y_hat),
                                "correct": int(y_hat == y_true),
                            }
                        )

    return prediction_rows, selected_rows


@dataclass
class BenchmarkRunner:
    """Main experiment runner."""

    config: BenchmarkConfig

    def run(self, dataset: AlarmSeriesDataset | None = None) -> dict[str, pd.DataFrame]:
        dataset = load_dataset_from_config(self.config) if dataset is None else dataset

        verbose = self.config.verbose
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.config.raw, f, indent=2)

        bcfg = self.config.benchmark_config
        severity_grid = [
            float(s)
            for s in bcfg.get(
                "severity_grid",
                np.round(np.arange(0, 1.0, 0.1), 3),
            )
        ]
        n_draws = int(bcfg.get("n_draws", 5))

        cv = self.config.cv_config
        n_splits = int(cv.get("n_splits", 5))
        n_repeats = int(cv.get("n_repeats", 10))
        val_fraction = float(cv.get("validation_fraction", 0.2))

        if not 0.0 <= val_fraction < 1.0:
            raise ValueError(
                "cv.validation_fraction must satisfy 0.0 <= validation_fraction < 1.0. "
                "Use 0.0 to disable validation and hyperparameter tuning."
            )

        use_validation = val_fraction > 0.0

        pcfg = self.config.parallel_config
        n_jobs = int(pcfg.get("n_jobs", 1))
        if n_jobs == 0:
            raise ValueError("parallel.n_jobs must not be 0. Use 1 for sequential execution or -1 for all cores.")

        backend = str(pcfg.get("backend", "loky"))
        pre_dispatch = str(pcfg.get("pre_dispatch", "2*n_jobs"))
        inner_max_num_threads_raw = pcfg.get("inner_max_num_threads", 1)
        inner_max_num_threads = (
            None
            if inner_max_num_threads_raw is None
            else int(inner_max_num_threads_raw)
        )

        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=self.config.random_seed,
        )

        split_iter = list(splitter.split(dataset.X, dataset.y))
        perturbation_specs = self.config.perturbation_specs
        model_specs = self.config.model_specs
        n_classes = len(np.unique(dataset.y))

        fold_model_jobs = [
            {
                "config": self.config,
                "dataset": dataset,
                "split_no": split_no,
                "n_splits": n_splits,
                "development_indices": development_indices,
                "test_indices": test_indices,
                "model_spec": model_spec,
                "severity_grid": severity_grid,
                "n_draws": n_draws,
                "val_fraction": val_fraction,
                "use_validation": use_validation,
                "inner_max_num_threads": inner_max_num_threads,
            }
            for split_no, (development_indices, test_indices) in enumerate(split_iter)
            for model_spec in model_specs
        ]

        n_eval_blocks = (
            len(fold_model_jobs)
            * len(perturbation_specs)
            * len(severity_grid)
            * n_draws
        )

        _log(
            verbose,
            (
                f"Running robustness benchmark for dataset='{dataset.name}' "
                f"with {len(dataset.y)} episodes, {n_classes} classes, "
                f"{len(split_iter)} CV folds, {len(model_specs)} models, "
                f"{len(perturbation_specs)} perturbation scenarios, "
                f"{len(severity_grid)} severity levels, and {n_draws} draw(s)."
            ),
        )
        _log(verbose, f"Output directory: {output_dir}")
        _log(
            verbose,
            (
                f"Parallel execution: n_jobs={n_jobs}, backend='{backend}', "
                f"inner_max_num_threads={inner_max_num_threads}."
            ),
        )

        if use_validation:
            _log(
                verbose,
                f"Hyperparameter selection: clean validation split with validation_fraction={val_fraction}.",
            )
        else:
            _log(
                verbose,
                "Hyperparameter selection disabled: using first params_grid entry for each model.",
            )

        _log(
            verbose,
            (
                f"Scheduled {len(fold_model_jobs)} fold/model task(s), "
                f"covering {n_eval_blocks} scenario/severity/draw evaluation block(s)."
            ),
        )

        if n_jobs == 1:
            results = []
            iterator = tqdm(
                fold_model_jobs,
                desc="Fold/model tasks",
                disable=not verbose,
                dynamic_ncols=True,
            )
            for job in iterator:
                model_name = str(job["model_spec"]["name"])
                split_no = int(job["split_no"])
                repeat = split_no // n_splits
                fold = split_no % n_splits

                if verbose:
                    iterator.set_postfix(
                        {
                            "fold": f"{repeat + 1}.{fold + 1}",
                            "model": model_name,
                        },
                        refresh=False,
                    )

                results.append(_run_fold_model_job(**job))
        else:
            progress = tqdm(
                total=len(fold_model_jobs),
                desc="Fold/model tasks",
                disable=not verbose,
                dynamic_ncols=True,
            )
            with _tqdm_joblib(progress):
                results = Parallel(
                    n_jobs=n_jobs,
                    backend=backend,
                    pre_dispatch=pre_dispatch,
                )(
                    delayed(_run_fold_model_job)(**job)
                    for job in fold_model_jobs
                )

        prediction_rows: list[dict[str, Any]] = []
        selected_rows: list[dict[str, Any]] = []

        for prediction_part, selected_part in results:
            prediction_rows.extend(prediction_part)
            selected_rows.extend(selected_part)

        predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
        selected = pd.DataFrame(selected_rows)

        predictions.to_csv(output_dir / "raw_predictions.csv", index=False)
        selected.to_csv(output_dir / "selected_hyperparameters.csv", index=False)

        tables = aggregate_all(predictions)
        tables["raw_predictions"] = predictions
        tables["selected_hyperparameters"] = selected

        for name, table in tables.items():
            if name in {"raw_predictions", "selected_hyperparameters"}:
                continue
            table.to_csv(output_dir / f"{name}.csv", index=False)

        _log(verbose, "Benchmark finished.")
        _log(verbose, f"Wrote raw predictions to: {output_dir / 'raw_predictions.csv'}")
        _log(verbose, f"Wrote selected hyperparameters to: {output_dir / 'selected_hyperparameters.csv'}")

        return tables

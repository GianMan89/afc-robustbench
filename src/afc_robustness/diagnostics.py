"""Diagnostics for perturbation effects and online-prefix mechanics.

This module is deliberately separate from the benchmark runner. It helps inspect
what perturbations do to alarm-event streams before those streams are passed to
AFC classifiers. The functions support two complementary views:

* dataset-level perturbation metrics aggregated over episodes, severities, and
  Monte-Carlo draws;
* episode-level timelines that show events, online update times, prefix-training
  boundaries, and prefix-classifier selection during inference.

The metrics distinguish activation-only changes from ACT/RTN event-stream
changes. This is important because set- and sequence-based AFC methods often use
activation information, whereas series-based methods are also affected by RTN
transitions because they determine alarm-state durations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from afc_robustness.data import AlarmSeriesDataset
from afc_robustness.domain import AlarmEpisode, EventType
from afc_robustness.models.factory import make_model
from afc_robustness.models.prefix import PrefixTrainedAFCModel
from afc_robustness.online import (
    OnlineEvaluationConfig,
    native_update_times,
    prefix_by_time,
)
from afc_robustness.perturbations import Perturbation, build_perturbation
from afc_robustness.repair import TraceRepair
from afc_robustness.representations import episode_to_series

EventMode = Literal["activations", "all"]
PrefixReference = Literal["event_count", "relative_time", "sample_count"]


@dataclass(frozen=True)
class PerturbationTrace:
    """Clean, unrepaired perturbed, and repaired perturbed versions of one episode."""

    clean: AlarmEpisode
    perturbed: AlarmEpisode
    repaired: AlarmEpisode
    scenario: str
    severity: float
    seed: int


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else np.nan
    return float(numerator) / float(denominator)


def _set_jaccard(a: set[Any], b: set[Any]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _counter_jaccard(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    intersection = sum(min(a.get(key, 0), b.get(key, 0)) for key in keys)
    union = sum(max(a.get(key, 0), b.get(key, 0)) for key in keys)
    return float(intersection) / float(union) if union else 1.0


def _lcs_length(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Return the longest common subsequence length using O(min(m,n)) memory."""

    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    previous = [0] * (len(short) + 1)
    for item_long in long:
        current = [0]
        for j, item_short in enumerate(short, start=1):
            if item_long == item_short:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def _lcs_similarity(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Sequence-order similarity in [0,1], normalized by the longer sequence."""

    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0
    return _lcs_length(a, b) / denom


def _event_subset(episode: AlarmEpisode, mode: EventMode) -> list:
    events = list(sorted(episode.events, key=lambda e: (e.timestamp, e.order)))
    if mode == "activations":
        events = [event for event in events if event.event_type == EventType.ACT]
    elif mode != "all":
        raise ValueError("mode must be 'activations' or 'all'")
    return events


def _event_counter(
    episode: AlarmEpisode,
    *,
    mode: EventMode,
    include_time: bool = False,
    time_bin_width: float = 1.0,
) -> Counter:
    """Build a multiset of events for robust similarity diagnostics.

    Without time, activation-only counters use ``tag`` and all-event counters use
    ``(tag, event_type)``. With time, timestamps are discretized to bins. Timing
    perturbations therefore become visible in the time-sensitive similarities.
    """

    counter: Counter = Counter()
    width = max(float(time_bin_width), 1e-12)
    for event in _event_subset(episode, mode):
        if mode == "activations":
            token: tuple[Any, ...] = (event.tag,)
        else:
            token = (event.tag, event.event_type.value)
        if include_time:
            token = (*token, int(np.floor(event.timestamp / width + 1e-12)))
        counter[token] += 1
    return counter


def _event_sequence(episode: AlarmEpisode, *, mode: EventMode) -> list[Any]:
    if mode == "activations":
        return [event.tag for event in _event_subset(episode, mode)]
    return [(event.tag, event.event_type.value) for event in _event_subset(episode, mode)]


def _event_tag_set(episode: AlarmEpisode, *, mode: EventMode) -> set[int]:
    return {event.tag for event in _event_subset(episode, mode)}


def _event_type_count(episode: AlarmEpisode, event_type: EventType) -> int:
    return sum(1 for event in episode.events if event.event_type == event_type)


def _state_metrics(clean: AlarmEpisode, perturbed: AlarmEpisode, *, dt: float) -> dict[str, float]:
    """Compare reconstructed binary alarm-state matrices on local episode grids.

    For late-detection perturbations, the perturbed episode is re-referenced to a
    shifted start time. The state metrics are therefore local-window metrics, not
    absolute physical-time metrics. The horizon ratio is reported to make this
    boundary effect explicit.
    """

    max_horizon = max(clean.horizon, perturbed.horizon)
    n_time = max(1, int(np.floor(max_horizon / dt + 1e-9)) + 1)
    X_clean = episode_to_series(clean, dt=dt, max_length=n_time).astype(bool)
    X_pert = episode_to_series(perturbed, dt=dt, max_length=n_time).astype(bool)
    diff = X_clean != X_pert
    inter = np.logical_and(X_clean, X_pert).sum()
    union = np.logical_or(X_clean, X_pert).sum()
    clean_active = X_clean.sum()
    pert_active = X_pert.sum()
    return {
        "state_hamming_fraction": float(diff.mean()) if diff.size else 0.0,
        "state_active_cell_jaccard": float(inter / union) if union else 1.0,
        "state_clean_active_cells": int(clean_active),
        "state_perturbed_active_cells": int(pert_active),
        "state_active_cell_ratio": _safe_ratio(float(pert_active), float(clean_active)),
    }


def event_table(episode: AlarmEpisode) -> pd.DataFrame:
    """Return a readable event table for one alarm-flood episode."""

    rows = []
    for idx, event in enumerate(sorted(episode.events, key=lambda e: (e.timestamp, e.order))):
        rows.append(
            {
                "event_index": idx,
                "timestamp": float(event.timestamp),
                "tag": int(event.tag),
                "tag_name": episode.tag_names[event.tag] if episode.tag_names else str(event.tag),
                "event_type": event.event_type.value,
                "order": int(event.order),
            }
        )
    return pd.DataFrame(rows)


def compare_episodes(
    clean: AlarmEpisode,
    perturbed: AlarmEpisode,
    *,
    dt: float = 1.0,
    time_bin_width: float | None = None,
) -> dict[str, Any]:
    """Compute activation-only and ACT/RTN perturbation metrics for two episodes."""

    width = float(dt if time_bin_width is None else time_bin_width)
    clean_act = _event_type_count(clean, EventType.ACT)
    pert_act = _event_type_count(perturbed, EventType.ACT)
    clean_rtn = _event_type_count(clean, EventType.RTN)
    pert_rtn = _event_type_count(perturbed, EventType.RTN)

    out: dict[str, Any] = {
        "sample_id": clean.sample_id,
        "n_tags": clean.n_tags,
        "clean_horizon": float(clean.horizon),
        "perturbed_horizon": float(perturbed.horizon),
        "horizon_delta": float(perturbed.horizon - clean.horizon),
        "horizon_ratio": _safe_ratio(perturbed.horizon, clean.horizon),
        "clean_n_events": int(clean.n_events),
        "perturbed_n_events": int(perturbed.n_events),
        "event_count_delta": int(perturbed.n_events - clean.n_events),
        "event_count_ratio": _safe_ratio(perturbed.n_events, clean.n_events),
        "clean_n_act": int(clean_act),
        "perturbed_n_act": int(pert_act),
        "act_count_delta": int(pert_act - clean_act),
        "act_count_ratio": _safe_ratio(pert_act, clean_act),
        "clean_n_rtn": int(clean_rtn),
        "perturbed_n_rtn": int(pert_rtn),
        "rtn_count_delta": int(pert_rtn - clean_rtn),
        "rtn_count_ratio": _safe_ratio(pert_rtn, clean_rtn),
        "act_tag_set_jaccard": _set_jaccard(
            _event_tag_set(clean, mode="activations"),
            _event_tag_set(perturbed, mode="activations"),
        ),
        "all_event_tag_set_jaccard": _set_jaccard(
            _event_tag_set(clean, mode="all"),
            _event_tag_set(perturbed, mode="all"),
        ),
        "act_tag_multiset_jaccard": _counter_jaccard(
            _event_counter(clean, mode="activations"),
            _event_counter(perturbed, mode="activations"),
        ),
        "all_tag_type_multiset_jaccard": _counter_jaccard(
            _event_counter(clean, mode="all"),
            _event_counter(perturbed, mode="all"),
        ),
        "act_tag_time_multiset_jaccard": _counter_jaccard(
            _event_counter(clean, mode="activations", include_time=True, time_bin_width=width),
            _event_counter(perturbed, mode="activations", include_time=True, time_bin_width=width),
        ),
        "all_tag_type_time_multiset_jaccard": _counter_jaccard(
            _event_counter(clean, mode="all", include_time=True, time_bin_width=width),
            _event_counter(perturbed, mode="all", include_time=True, time_bin_width=width),
        ),
        "act_sequence_lcs_similarity": _lcs_similarity(
            _event_sequence(clean, mode="activations"),
            _event_sequence(perturbed, mode="activations"),
        ),
        "all_sequence_lcs_similarity": _lcs_similarity(
            _event_sequence(clean, mode="all"),
            _event_sequence(perturbed, mode="all"),
        ),
    }
    out.update(_state_metrics(clean, perturbed, dt=dt))
    return out


def perturb_episode_trace(
    episode: AlarmEpisode,
    perturbation: Perturbation | dict[str, Any] | str,
    *,
    severity: float,
    seed: int = 42,
    repair: TraceRepair | None = None,
) -> PerturbationTrace:
    """Apply one perturbation and return clean, unrepaired, and repaired episodes."""

    operator = build_perturbation(perturbation) if not hasattr(perturbation, "apply") else perturbation
    rng = np.random.default_rng(seed)
    perturbed = operator.apply(episode, float(severity), rng)  # type: ignore[union-attr]
    repaired = (TraceRepair() if repair is None else repair).apply(perturbed)
    return PerturbationTrace(
        clean=episode,
        perturbed=perturbed,
        repaired=repaired,
        scenario=getattr(operator, "name", str(perturbation)),
        severity=float(severity),
        seed=int(seed),
    )


def dataset_perturbation_metrics(
    dataset: AlarmSeriesDataset,
    perturbation_specs: Sequence[dict[str, Any] | str],
    *,
    severity_grid: Sequence[float],
    n_draws: int = 5,
    sample_indices: Sequence[int] | None = None,
    random_seed: int = 42,
    dt: float | None = None,
    initial_active_policy: str = "act",
    allow_leading_rtn: bool = True,
    include_unrepaired: bool = False,
    time_bin_width: float | None = None,
    verbose: bool = False,
    progress: bool | None = None,
    progress_unit: Literal["sample", "block"] = "sample",
) -> pd.DataFrame:
    """Generate perturbation-diagnostic metrics for a dataset.

    The returned table has one row per scenario, severity, draw, sample, and
    variant. The default variant is the repaired perturbed episode, because that
    is what the benchmark passes to classifiers. Set ``include_unrepaired=True``
    to also inspect effects before trace repair.

    Parameters
    ----------
    verbose:
        If ``True``, print a compact execution summary before the loop and a
        completion message afterwards. No intermediate results are printed.
    progress:
        If ``True``, show a single ``tqdm`` progress bar. If ``None``, the value
        follows ``verbose``.
    progress_unit:
        ``"sample"`` updates the bar after every processed episode and is more
        informative for long runs. ``"block"`` updates once per
        scenario/severity/draw combination and has slightly lower overhead.
    """

    if progress_unit not in {"sample", "block"}:
        raise ValueError("progress_unit must be either 'sample' or 'block'")

    show_progress = bool(verbose if progress is None else progress)

    if sample_indices is None:
        indices = np.arange(dataset.n_episodes, dtype=int)
    else:
        indices = np.asarray(sample_indices, dtype=int)

    severity_values = [float(value) for value in severity_grid]
    operators = [build_perturbation(spec) for spec in perturbation_specs]
    repair = TraceRepair(allow_leading_rtn=allow_leading_rtn)
    dt_value = dataset.dt if dt is None else float(dt)

    n_variants = 2 if include_unrepaired else 1
    n_blocks = len(operators) * len(severity_values) * int(n_draws)
    n_samples = len(indices)
    expected_rows = n_blocks * n_samples * n_variants

    if verbose:
        print(
            "Perturbation diagnostics: "
            f"dataset='{dataset.name}', scenarios={len(operators)}, "
            f"severities={len(severity_values)}, draws={int(n_draws)}, "
            f"samples={n_samples}, variants={n_variants}, "
            f"expected_rows={expected_rows:,}."
        )
        print(
            "Progress unit: "
            f"{progress_unit}; include_unrepaired={include_unrepaired}; "
            f"initial_active_policy='{initial_active_policy}'; "
            f"allow_leading_rtn={allow_leading_rtn}."
        )

    pbar_total = n_blocks * n_samples if progress_unit == "sample" else n_blocks
    pbar = None
    if show_progress:
        from tqdm.auto import tqdm

        pbar = tqdm(
            total=pbar_total,
            desc="Perturbation diagnostics",
            unit=progress_unit,
            dynamic_ncols=True,
            leave=True,
        )

    rows: list[dict[str, Any]] = []
    try:
        for scenario_index, operator in enumerate(operators):
            for severity_value in severity_values:
                for draw in range(int(n_draws)):
                    seed = (
                        int(random_seed)
                        + 1009 * int(round(severity_value * 1000))
                        + 10007 * draw
                        + 101 * scenario_index
                    )
                    rng = np.random.default_rng(seed)

                    if pbar is not None:
                        pbar.set_postfix(
                            {
                                "scenario": operator.name,
                                "severity": severity_value,
                                "draw": draw + 1,
                            },
                            refresh=False,
                        )

                    for sample_index in indices:
                        sample_index_int = int(sample_index)
                        clean = dataset.episode(
                            sample_index_int,
                            initial_active_policy=initial_active_policy,
                        )
                        perturbed = operator.apply(clean, severity_value, rng)
                        repaired = repair.apply(perturbed)
                        variants: list[tuple[str, AlarmEpisode]] = [("repaired", repaired)]
                        if include_unrepaired:
                            variants.insert(0, ("unrepaired", perturbed))

                        for variant_name, candidate in variants:
                            metrics = compare_episodes(
                                clean,
                                candidate,
                                dt=dt_value,
                                time_bin_width=time_bin_width,
                            )
                            metrics.update(
                                {
                                    "dataset": dataset.name,
                                    "scenario": operator.name,
                                    "severity": severity_value,
                                    "draw": int(draw),
                                    "seed": int(seed),
                                    "sample_index": sample_index_int,
                                    "sample_id": dataset.sample_ids[sample_index_int],
                                    "class_label": int(dataset.y[sample_index_int]),
                                    "class_name": dataset.class_names[int(dataset.y[sample_index_int])],
                                    "variant": variant_name,
                                }
                            )
                            rows.append(metrics)

                        if pbar is not None and progress_unit == "sample":
                            pbar.update(1)

                    if pbar is not None and progress_unit == "block":
                        pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    if verbose:
        print(f"Perturbation diagnostics finished: generated {len(rows):,} rows.")

    return pd.DataFrame(rows)


def aggregate_diagnostic_metrics(
    metrics: pd.DataFrame,
    *,
    group_by: Sequence[str] = ("scenario", "severity", "variant"),
) -> pd.DataFrame:
    """Aggregate diagnostic metrics with mean, std, and robust quantiles."""

    if metrics.empty:
        return pd.DataFrame()
    numeric_cols = [
        col
        for col in metrics.select_dtypes(include=[np.number]).columns
        if col not in set(group_by) | {"draw", "seed", "sample_index", "class_label"}
    ]
    frames: list[pd.DataFrame] = []
    grouped = metrics.groupby(list(group_by), dropna=False)
    for stat_name, frame in [
        ("mean", grouped[numeric_cols].mean()),
        ("std", grouped[numeric_cols].std(ddof=1)),
        ("p05", grouped[numeric_cols].quantile(0.05)),
        ("p50", grouped[numeric_cols].quantile(0.50)),
        ("p95", grouped[numeric_cols].quantile(0.95)),
    ]:
        renamed = frame.add_suffix(f"_{stat_name}")
        frames.append(renamed)
    return pd.concat(frames, axis=1).reset_index()


def compute_update_table(episode: AlarmEpisode, config: OnlineEvaluationConfig) -> pd.DataFrame:
    """Return native online update times and the prefix available at each update."""

    times = native_update_times(
        episode.horizon,
        update_interval=config.update_interval,
        include_zero_update=config.include_zero_update,
    )
    n_events_total = max(episode.n_events, 1)
    horizon = max(float(episode.horizon), 0.0)
    n_time_total = max(1, int(np.floor(horizon / config.dt + 1e-9)) + 1)

    rows = []
    for update_index, update_time in enumerate(times):
        prefix = prefix_by_time(episode, float(update_time))
        X_prefix = episode_to_series(prefix, dt=config.dt)
        event_count_progress = min(prefix.n_events / n_events_total, 1.0)
        relative_time_progress = 1.0 if horizon <= 0 else min(float(update_time) / horizon, 1.0)
        sample_count_progress = min(X_prefix.shape[1] / n_time_total, 1.0)
        benchmark_progress = (
            relative_time_progress
            if config.progress_reference == "relative_time"
            else event_count_progress
        )
        act_events = [e for e in prefix.events if e.event_type == EventType.ACT]
        rtn_events = [e for e in prefix.events if e.event_type == EventType.RTN]
        act_tags = sorted({e.tag for e in act_events})
        all_tags = sorted({e.tag for e in prefix.events})
        rows.append(
            {
                "update_index": int(update_index),
                "update_time": float(update_time),
                "benchmark_progress": float(benchmark_progress),
                "event_count_progress": float(event_count_progress),
                "relative_time_progress": float(relative_time_progress),
                "sample_count_progress": float(sample_count_progress),
                "n_events_observed": int(prefix.n_events),
                "n_events_total": int(episode.n_events),
                "n_act_observed": int(len(act_events)),
                "n_rtn_observed": int(len(rtn_events)),
                "n_unique_act_tags_observed": int(len(act_tags)),
                "n_unique_event_tags_observed": int(len(all_tags)),
                "act_tags_observed": tuple(act_tags),
                "event_tags_observed": tuple(all_tags),
                "act_tag_names_observed": tuple(episode.tag_names[tag] for tag in act_tags),
                "event_tag_names_observed": tuple(episode.tag_names[tag] for tag in all_tags),
            }
        )
    return pd.DataFrame(rows)


def _prefix_episode_by_reference(
    episode: AlarmEpisode,
    rho: float,
    reference: PrefixReference,
    *,
    dt: float = 1.0,
) -> AlarmEpisode:
    rho = min(max(float(rho), 0.0), 1.0)
    if reference == "event_count":
        if episode.n_events == 0:
            return episode.empty_prefix(horizon=0.0)
        k = max(1, min(episode.n_events, int(np.ceil(rho * episode.n_events))))
        events = episode.events[:k]
        horizon = episode.horizon if rho >= 1.0 - 1e-12 else events[-1].timestamp
        return episode.with_events(events, horizon=horizon)
    if reference == "relative_time":
        horizon = min(episode.horizon, rho * episode.horizon)
        events = [event for event in episode.events if event.timestamp <= horizon + 1e-12]
        return episode.with_events(events, horizon=horizon)
    if reference == "sample_count":
        n_time = max(1, int(np.floor(episode.horizon / dt + 1e-9)) + 1)
        length = max(1, int(np.ceil(rho * n_time)))
        horizon = min(episode.horizon, (length - 1) * dt)
        events = [event for event in episode.events if event.timestamp <= horizon + 1e-12]
        return episode.with_events(events, horizon=horizon)
    raise ValueError("reference must be event_count, relative_time, or sample_count")


def _expand_param_grid(grid: list[dict[str, Any]] | dict[str, list[Any]] | None) -> list[dict[str, Any]]:
    if grid is None:
        return [{}]
    if isinstance(grid, list):
        return [dict(item) for item in grid] if grid else [{}]
    if isinstance(grid, dict):
        import itertools

        keys = list(grid)
        values = [grid[key] if isinstance(grid[key], list) else [grid[key]] for key in keys]
        return [dict(zip(keys, combo, strict=False)) for combo in itertools.product(*values)]
    raise TypeError("params_grid must be a list of dicts, dict of lists, or None")


def prefix_model_descriptions(
    model_specs: Sequence[dict[str, Any]],
    online_config: OnlineEvaluationConfig,
) -> pd.DataFrame:
    """Describe prefix-trained model settings present in the YAML model specs."""

    rows: list[dict[str, Any]] = []
    for model_spec in model_specs:
        model_name = str(model_spec["name"])
        params_grid = _expand_param_grid(model_spec.get("params_grid"))
        for param_index, params in enumerate(params_grid):
            if params.get("training_mode") != "prefix":
                continue
            model = make_model(model_name, params)
            if not isinstance(model, PrefixTrainedAFCModel):
                continue
            grid = model._resolve_grid(online_config)  # diagnostic introspection
            train_reference = model._resolve_training_reference(online_config)
            rows.append(
                {
                    "model_key": model_name,
                    "method": model.name,
                    "param_index": int(param_index),
                    "prefix_grid": tuple(float(v) for v in grid),
                    "prefix_reference": model.prefix_reference,
                    "prefix_train_reference": train_reference,
                    "prefix_selection": model.prefix_selection,
                    "prefix_train_horizon": model.prefix_train_horizon,
                    "prefix_min_time_steps": int(model.prefix_min_time_steps),
                }
            )
    return pd.DataFrame(rows)


def prefix_training_plan(
    dataset: AlarmSeriesDataset,
    indices: Sequence[int],
    model_specs: Sequence[dict[str, Any]],
    online_config: OnlineEvaluationConfig,
    *,
    initial_active_policy: str = "act",
) -> pd.DataFrame:
    """Summarize which clean training prefixes are used for prefix classifiers."""

    descriptions = prefix_model_descriptions(model_specs, online_config)
    if descriptions.empty:
        return pd.DataFrame()
    idx_arr = np.asarray(indices, dtype=int)
    rows: list[dict[str, Any]] = []
    for desc in descriptions.to_dict("records"):
        for rho in desc["prefix_grid"]:
            per_episode: list[dict[str, Any]] = []
            for idx in idx_arr:
                episode = dataset.episode(int(idx), initial_active_policy=initial_active_policy)
                prefix = _prefix_episode_by_reference(
                    episode,
                    float(rho),
                    desc["prefix_train_reference"],
                    dt=dataset.dt,
                )
                matrix = episode_to_series(prefix, dt=dataset.dt)
                per_episode.append(
                    {
                        "n_events": prefix.n_events,
                        "n_act": _event_type_count(prefix, EventType.ACT),
                        "n_rtn": _event_type_count(prefix, EventType.RTN),
                        "n_unique_act_tags": len(_event_tag_set(prefix, mode="activations")),
                        "horizon": prefix.horizon,
                        "n_time_steps": matrix.shape[1],
                    }
                )
            tmp = pd.DataFrame(per_episode)
            row = {**desc, "prefix_progress": float(rho), "n_episodes": int(len(idx_arr))}
            for col in ["n_events", "n_act", "n_rtn", "n_unique_act_tags", "horizon", "n_time_steps"]:
                row[f"{col}_min"] = float(tmp[col].min())
                row[f"{col}_mean"] = float(tmp[col].mean())
                row[f"{col}_max"] = float(tmp[col].max())
            rows.append(row)
    return pd.DataFrame(rows)


def prefix_boundary_table(
    episode: AlarmEpisode,
    prefix_grid: Sequence[float],
    *,
    reference: PrefixReference = "event_count",
    dt: float = 1.0,
) -> pd.DataFrame:
    """Return event/time boundaries used to construct prefix-specific classifiers."""

    rows = []
    for rho in sorted({float(v) for v in prefix_grid}):
        prefix = _prefix_episode_by_reference(episode, rho, reference, dt=dt)
        act_tags = sorted(_event_tag_set(prefix, mode="activations"))
        rows.append(
            {
                "prefix_progress": float(rho),
                "boundary_time": float(prefix.horizon),
                "n_events_used": int(prefix.n_events),
                "n_act_used": int(_event_type_count(prefix, EventType.ACT)),
                "n_rtn_used": int(_event_type_count(prefix, EventType.RTN)),
                "n_unique_act_tags_used": int(len(act_tags)),
                "act_tags_used": tuple(act_tags),
                "act_tag_names_used": tuple(episode.tag_names[tag] for tag in act_tags),
            }
        )
    return pd.DataFrame(rows)


def _select_prefix_from_grid(progress: float, grid: Sequence[float], selection: str) -> float:
    values = np.asarray(sorted({float(v) for v in grid}), dtype=float)
    progress = min(max(float(progress), 0.0), 1.0)
    if selection == "nearest":
        return float(values[int(np.argmin(np.abs(values - progress)))])
    if selection == "right":
        idx = int(np.searchsorted(values, progress, side="left"))
        return float(values[min(idx, len(values) - 1)])
    idx = int(np.searchsorted(values, progress, side="right")) - 1
    return float(values[max(0, min(idx, len(values) - 1))])


def prefix_inference_table(
    episode: AlarmEpisode,
    model_specs: Sequence[dict[str, Any]],
    online_config: OnlineEvaluationConfig,
) -> pd.DataFrame:
    """Show which prefix-trained classifier would be selected at each update."""

    descriptions = prefix_model_descriptions(model_specs, online_config)
    if descriptions.empty:
        return pd.DataFrame()
    updates = compute_update_table(episode, online_config)
    rows: list[dict[str, Any]] = []
    for desc in descriptions.to_dict("records"):
        grid = desc["prefix_grid"]
        for update in updates.to_dict("records"):
            if desc["prefix_reference"] == "online":
                progress = update["benchmark_progress"]
            elif desc["prefix_reference"] == "event_count":
                progress = update["event_count_progress"]
            elif desc["prefix_reference"] == "relative_time":
                progress = update["relative_time_progress"]
            elif desc["prefix_reference"] == "sample_count":
                progress = update["sample_count_progress"]
            else:
                progress = update["benchmark_progress"]
            selected = _select_prefix_from_grid(progress, grid, desc["prefix_selection"])
            rows.append(
                {
                    **{k: update[k] for k in updates.columns if k not in {"act_tags_observed", "event_tags_observed", "act_tag_names_observed", "event_tag_names_observed"}},
                    "model_key": desc["model_key"],
                    "method": desc["method"],
                    "prefix_reference": desc["prefix_reference"],
                    "prefix_selection": desc["prefix_selection"],
                    "observed_selection_progress": float(progress),
                    "selected_prefix_progress": float(selected),
                    "act_tag_names_observed": update["act_tag_names_observed"],
                    "event_tag_names_observed": update["event_tag_names_observed"],
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers. These functions intentionally use only matplotlib defaults.
# ---------------------------------------------------------------------------


def plot_metric_vs_severity(
    metrics: pd.DataFrame,
    metric: str,
    *,
    variant: str = "repaired",
    scenarios: Sequence[str] | None = None,
    ax: Any | None = None,
    title: str | None = None,
):
    """Line plot of a diagnostic metric against severity, grouped by scenario."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    data = metrics.copy()
    if "variant" in data.columns:
        data = data[data["variant"] == variant]
    if scenarios is not None:
        data = data[data["scenario"].isin(list(scenarios))]
    grouped = data.groupby(["scenario", "severity"], dropna=False)[metric]
    summary = grouped.agg(["mean", "std"]).reset_index()
    for scenario, part in summary.groupby("scenario", sort=False):
        part = part.sort_values("severity")
        ax.plot(part["severity"], part["mean"], marker="o", label=str(scenario))
        if part["std"].notna().any():
            lower = part["mean"] - part["std"]
            upper = part["mean"] + part["std"]
            ax.fill_between(part["severity"], lower, upper, alpha=0.15)
    ax.set_xlabel("Perturbation severity")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax


def plot_metric_heatmap(
    metrics: pd.DataFrame,
    metric: str,
    *,
    variant: str = "repaired",
    ax: Any | None = None,
    title: str | None = None,
):
    """Heatmap of mean metric values by scenario and severity."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    data = metrics.copy()
    if "variant" in data.columns:
        data = data[data["variant"] == variant]
    pivot = data.pivot_table(index="scenario", columns="severity", values=metric, aggfunc="mean")
    image = ax.imshow(pivot.to_numpy(), aspect="auto")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns], rotation=45, ha="right")
    ax.set_xlabel("Perturbation severity")
    ax.set_title(title or metric)
    plt.colorbar(image, ax=ax, label=metric)
    return ax


def plot_episode_timeline(
    episode: AlarmEpisode,
    *,
    update_table: pd.DataFrame | None = None,
    prefix_boundaries: pd.DataFrame | None = None,
    event_mode: EventMode = "all",
    max_tags: int | None = 40,
    annotate_updates: bool = False,
    ax: Any | None = None,
    title: str | None = None,
):
    """Plot ACT/RTN events over time and optionally mark online update times."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    table = event_table(episode)
    if event_mode == "activations":
        table = table[table["event_type"] == "ACT"]

    if max_tags is not None and table["tag"].nunique() > max_tags:
        selected_tags = table["tag"].value_counts().head(max_tags).index.tolist()
        table = table[table["tag"].isin(selected_tags)]
    else:
        selected_tags = sorted(table["tag"].unique().tolist()) if not table.empty else []

    tag_to_y = {tag: pos for pos, tag in enumerate(selected_tags)}
    for event_type, marker in [("ACT", "^"), ("RTN", "v")]:
        part = table[table["event_type"] == event_type]
        if part.empty:
            continue
        ax.scatter(
            part["timestamp"],
            [tag_to_y[tag] for tag in part["tag"]],
            marker=marker,
            label=event_type,
        )

    if update_table is not None and not update_table.empty:
        for _, row in update_table.iterrows():
            ax.axvline(float(row["update_time"]), linestyle=":", linewidth=0.8, alpha=0.35)
            if annotate_updates:
                ax.text(
                    float(row["update_time"]),
                    len(selected_tags),
                    str(int(row["update_index"])),
                    rotation=90,
                    va="bottom",
                    ha="center",
                    fontsize=7,
                )

    if prefix_boundaries is not None and not prefix_boundaries.empty:
        for _, row in prefix_boundaries.iterrows():
            ax.axvline(float(row["boundary_time"]), linestyle="--", linewidth=1.0, alpha=0.6)
            ax.text(
                float(row["boundary_time"]),
                -0.8,
                f"{float(row['prefix_progress']):.2g}",
                rotation=90,
                va="top",
                ha="center",
                fontsize=8,
            )

    ax.set_yticks(np.arange(len(selected_tags)))
    ax.set_yticklabels([episode.tag_names[tag] for tag in selected_tags])
    ax.set_xlabel("Local episode time")
    ax.set_ylabel("Alarm tag")
    ax.set_title(title or f"Episode {episode.sample_id}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    return ax


def plot_alarm_state_matrix(
    episode: AlarmEpisode,
    *,
    update_table: pd.DataFrame | None = None,
    max_tags: int | None = 60,
    dt: float = 1.0,
    ax: Any | None = None,
    title: str | None = None,
):
    """Show the reconstructed binary alarm-state matrix and update locations."""

    import matplotlib.pyplot as plt

    X = episode_to_series(episode, dt=dt).astype(float)
    tag_indices = np.arange(episode.n_tags)
    if max_tags is not None and episode.n_tags > max_tags:
        activity = X.sum(axis=1)
        tag_indices = np.argsort(activity)[::-1][:max_tags]
        tag_indices = np.sort(tag_indices)
        X = X[tag_indices]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(X, aspect="auto", interpolation="nearest", origin="lower")
    if update_table is not None and not update_table.empty:
        for _, row in update_table.iterrows():
            ax.axvline(float(row["update_time"]) / dt, linestyle=":", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Alarm tag")
    ax.set_yticks(np.arange(len(tag_indices)))
    ax.set_yticklabels([episode.tag_names[int(tag)] for tag in tag_indices])
    ax.set_title(title or f"Alarm-state matrix: {episode.sample_id}")
    return ax


def plot_perturbation_comparison(
    trace: PerturbationTrace,
    *,
    online_config: OnlineEvaluationConfig | None = None,
    event_mode: EventMode = "all",
    max_tags: int | None = 40,
):
    """Plot clean, unrepaired, and repaired event timelines for one perturbation."""

    import matplotlib.pyplot as plt

    episodes = [("clean", trace.clean), ("perturbed", trace.perturbed), ("repaired", trace.repaired)]
    fig, axes = plt.subplots(len(episodes), 1, figsize=(11, 4 * len(episodes)), sharex=False)
    if len(episodes) == 1:
        axes = [axes]
    for ax, (label, episode) in zip(axes, episodes, strict=False):
        updates = compute_update_table(episode, online_config) if online_config is not None else None
        plot_episode_timeline(
            episode,
            update_table=updates,
            event_mode=event_mode,
            max_tags=max_tags,
            ax=ax,
            title=f"{label}: {trace.scenario}, severity={trace.severity}",
        )
    fig.tight_layout()
    return fig, axes


def plot_prefix_training_coverage(
    boundary_table: pd.DataFrame,
    *,
    ax: Any | None = None,
    title: str | None = None,
):
    """Plot prefix boundary time and event count as a function of prefix progress."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    data = boundary_table.sort_values("prefix_progress")
    ax.step(data["prefix_progress"], data["boundary_time"], where="post", marker="o", label="boundary time")
    ax.set_xlabel("Prefix progress")
    ax.set_ylabel("Boundary time")
    ax2 = ax.twinx()
    ax2.step(data["prefix_progress"], data["n_events_used"], where="post", marker="s", label="events used")
    ax2.set_ylabel("Number of events used")
    ax.set_title(title or "Prefix-training coverage")
    ax.grid(True, alpha=0.3)
    return ax


def plot_prefix_inference_selection(
    inference_table: pd.DataFrame,
    *,
    model_key: str | None = None,
    ax: Any | None = None,
    title: str | None = None,
):
    """Plot observed progress and selected prefix-classifier progress over time."""

    import matplotlib.pyplot as plt

    data = inference_table.copy()
    if model_key is not None:
        data = data[data["model_key"] == model_key]
    if data.empty:
        raise ValueError("no prefix-inference rows available for plotting")
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    for model, part in data.groupby("model_key", sort=False):
        part = part.sort_values("update_time")
        ax.step(
            part["update_time"],
            part["selected_prefix_progress"],
            where="post",
            marker="o",
            label=f"selected prefix: {model}",
        )
    first = data.sort_values("update_time")
    ax.plot(first["update_time"], first["observed_selection_progress"], linestyle="--", label="observed progress")
    ax.set_xlabel("Local episode time")
    ax.set_ylabel("Progress")
    ax.set_ylim(0, 1.05)
    ax.set_title(title or "Prefix classifier selected during inference")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    return ax

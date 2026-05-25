"""Aggregation and robustness metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = [
    "dataset",
    "fold",
    "repeat",
    "method",
    "scenario",
    "severity",
    "draw",
    "sample_id",
    "y_true",
    "progress",
    "y_pred",
    "correct",
]


def compute_degradation_profiles(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute empirical degradation profile M_m,p(s,rho)."""

    group_cols = ["dataset", "fold", "repeat", "method", "scenario", "severity", "progress"]
    profiles = (
        predictions.groupby(group_cols, dropna=False)["correct"]
        .agg(score="mean", n="count")
        .reset_index()
        .sort_values(group_cols)
    )
    return profiles


def compute_progress_auc(profiles: pd.DataFrame) -> pd.DataFrame:
    """Compute discrete area under progress--performance curve A_m,p(s)."""

    group_cols = ["dataset", "fold", "repeat", "method", "scenario", "severity"]
    auc = (
        profiles.groupby(group_cols, dropna=False)["score"]
        .mean()
        .reset_index(name="progress_auc")
        .sort_values(group_cols)
    )
    return auc


def compute_scenario_scores(progress_auc: pd.DataFrame) -> pd.DataFrame:
    """Compute scenario-level robustness scores R_m,p."""

    group_cols = ["dataset", "fold", "repeat", "method", "scenario"]
    scores = (
        progress_auc.groupby(group_cols, dropna=False)["progress_auc"]
        .mean()
        .reset_index(name="scenario_score")
        .sort_values(group_cols)
    )
    return scores


def compute_overall_scores(scenario_scores: pd.DataFrame) -> pd.DataFrame:
    """Compute average, product, and worst-scenario robustness scores."""

    rows = []
    group_cols = ["dataset", "fold", "repeat", "method"]
    for keys, group in scenario_scores.groupby(group_cols, dropna=False):
        values = group["scenario_score"].to_numpy(dtype=float)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,), strict=False))
        row.update(
            {
                "R_avg": float(np.mean(values)) if len(values) else np.nan,
                "R_prod": float(np.prod(values)) if len(values) else np.nan,
                "R_min": float(np.min(values)) if len(values) else np.nan,
                "n_scenarios": int(len(values)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def summarize_overall_scores(overall_by_fold: pd.DataFrame) -> pd.DataFrame:
    """Mean/std summary over repeated folds."""

    group_cols = ["dataset", "method"]
    value_cols = ["R_avg", "R_prod", "R_min"]
    summary = overall_by_fold.groupby(group_cols, dropna=False)[value_cols].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index().sort_values(group_cols)


def aggregate_all(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute all benchmark aggregation tables."""

    profiles = compute_degradation_profiles(predictions)
    progress_auc = compute_progress_auc(profiles)
    scenario_scores = compute_scenario_scores(progress_auc)
    overall_by_fold = compute_overall_scores(scenario_scores)
    overall_summary = summarize_overall_scores(overall_by_fold)
    return {
        "degradation_profiles": profiles,
        "progress_auc": progress_auc,
        "scenario_scores": scenario_scores,
        "overall_scores_by_fold": overall_by_fold,
        "overall_scores_summary": overall_summary,
    }

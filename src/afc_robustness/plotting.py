"""Plotting utilities for benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_progress_auc_by_severity(
    progress_auc: pd.DataFrame,
    output_dir: str | Path,
    *,
    file_prefix: str = "progress_auc",
) -> list[Path]:
    """Create one degradation curve plot per perturbation scenario."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    summary = (
        progress_auc.groupby(["dataset", "method", "scenario", "severity"], dropna=False)[
            "progress_auc"
        ]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    for (dataset, scenario), group in summary.groupby(["dataset", "scenario"], dropna=False):
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for method, method_group in group.groupby("method", dropna=False):
            method_group = method_group.sort_values("severity")
            ax.plot(method_group["severity"], method_group["mean"], marker="o", label=str(method))
        ax.set_title(f"{dataset}: {scenario}")
        ax.set_xlabel("Perturbation severity")
        ax.set_ylabel("Mean progress AUC")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = output / f"{file_prefix}_{dataset}_{scenario}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_overall_scores(
    overall_summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    metric: str = "R_avg_mean",
    file_name: str = "overall_scores.png",
) -> Path:
    """Create a bar chart for an overall robustness metric."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = overall_summary.sort_values(metric, ascending=False)
    labels = df["method"].astype(str).tolist()
    values = df[metric].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(labels, values)
    ax.set_ylabel(metric)
    ax.set_ylim(0.0, 1.02 if metric.startswith("R_") and "prod" not in metric else max(values) * 1.1)
    ax.set_title("Overall robustness summary")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = output / file_name
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path

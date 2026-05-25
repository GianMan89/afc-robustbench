"""Command-line interface for the AFC robustness benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from afc_robustness.data import describe_dataset, load_dataset_from_folders
from afc_robustness.experiment import BenchmarkConfig, BenchmarkRunner
from afc_robustness.plotting import plot_overall_scores, plot_progress_auc_by_severity


def command_check_data(args: argparse.Namespace) -> None:
    bcfg = BenchmarkConfig.from_yaml(args.config)
    dataset_cfg = bcfg.dataset_config
    dataset = load_dataset_from_folders(
        dataset_cfg["path"],
        name=dataset_cfg.get("name"),
        dt=float(dataset_cfg.get("dt", 1.0)),
        max_length=dataset_cfg.get("max_length"),
        binary_threshold=float(dataset_cfg.get("binary_threshold", 0.5)),
    )
    print(f"Dataset: {dataset.name}")
    print(f"Episodes: {dataset.n_episodes}")
    print(f"Alarm tags: {dataset.n_tags}")
    print(f"Padded time steps: {dataset.n_time_steps}")
    print(describe_dataset(dataset).to_string(index=False))


def command_run(args: argparse.Namespace) -> None:
    cfg = BenchmarkConfig.from_yaml(args.config)
    raw = dict(cfg.raw)
    if args.output_dir is not None:
        raw["output_dir"] = args.output_dir
        cfg = BenchmarkConfig(raw=raw, config_path=cfg.config_path)
    runner = BenchmarkRunner(cfg)
    runner.run()
    print(f"Benchmark outputs written to {cfg.output_dir}")


def command_plot(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir) if args.figures_dir else results_dir / "figures"
    progress_auc_path = results_dir / "progress_auc.csv"
    overall_summary_path = results_dir / "overall_scores_summary.csv"
    if progress_auc_path.exists():
        progress_auc = pd.read_csv(progress_auc_path)
        paths = plot_progress_auc_by_severity(progress_auc, figures_dir)
        print(f"Wrote {len(paths)} degradation-curve figures to {figures_dir}")
    if overall_summary_path.exists():
        overall = pd.read_csv(overall_summary_path)
        path = plot_overall_scores(overall, figures_dir)
        print(f"Wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AFC robustness benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check-data", help="Load and summarize a configured dataset")
    p_check.add_argument("--config", required=True, help="Path to YAML config")
    p_check.set_defaults(func=command_check_data)

    p_run = sub.add_parser("run", help="Run a configured robustness benchmark")
    p_run.add_argument("--config", required=True, help="Path to YAML config")
    p_run.add_argument("--output-dir", default=None, help="Override output directory")
    p_run.set_defaults(func=command_run)

    p_plot = sub.add_parser("plot", help="Plot saved benchmark outputs")
    p_plot.add_argument("--results-dir", required=True, help="Directory containing CSV outputs")
    p_plot.add_argument("--figures-dir", default=None, help="Optional output figure directory")
    p_plot.set_defaults(func=command_plot)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

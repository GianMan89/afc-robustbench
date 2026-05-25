# Perturbation-Based Robustness Benchmarking for Online Alarm Flood Classification

This repository contains a structured, object-oriented Python implementation of the evaluation protocol for the manuscript

> **Perturbation-Based Robustness Benchmarking for Online Alarm Flood Classification under Alarm-Log Degradations and Detection Delays**

The code is organized so that Jupyter notebooks remain thin, reproducible experiment entry points. All reusable logic is implemented in the `afc_robustness` package under `src/`.

## What is implemented

The repository implements the manuscript's prefix-based online AFC robustness benchmark:

- Loading binary alarm-series CSV files and converting them to alarm-event episodes with `ACT` and `RTN` transitions.
- Event-stream perturbation functions for missing events, tag dropout, spurious events, burst spurious events, timing uncertainty, event-count-based delayed detection, duration-based delayed detection, and ordered mixed perturbations.
- Trace repair that enforces tag-wise alternation while allowing a leading `RTN` event.
- Online evaluation on a time-driven update schedule with causal mapping to a common progress grid.
- Robustness aggregation into degradation profiles, progress-area summaries, scenario-level robustness scores, and overall average/product/minimum scores.
- Six AFC model families aligned with the manuscript: `WDI-1NN`, `JAC-1NN`, `EAC-1NN`, `MBW-LR`, `ACM-SVM`, and `CASIM`.
- Cross-validation with clean validation-based hyperparameter selection and test-time perturbation evaluation.
- CSV outputs and plotting utilities for degradation curves and scalar robustness summaries.

## Repository layout

```text
.
├── configs/                 # YAML experiment configurations
├── data/                    # Empty data folders; place TEP/FCC data here manually
│   ├── tep/
│   └── fcc/
├── docs/                    # Notes on manuscript/code alignment
├── notebooks/               # Thin notebook entry points
├── scripts/                 # Utility scripts, e.g. synthetic-data generation
├── src/afc_robustness/      # Installable Python package
│   ├── data.py              # Dataset loading and padding
│   ├── domain.py            # AlarmEvent and AlarmEpisode domain objects
│   ├── perturbations.py     # Perturbation suite and mixed compositions
│   ├── repair.py            # Trace repair
│   ├── representations.py   # Series/event/set/sequence conversion
│   ├── online.py            # Prefix-based online evaluation
│   ├── metrics.py           # Robustness aggregation
│   ├── experiment.py        # Cross-validation benchmark runner
│   ├── plotting.py          # Result visualizations
│   └── models/              # AFC method implementations
└── tests/                   # Unit tests for core semantics
```

## Installation

Create an environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the exact `CASIM`/MultiRocket backend, install the optional dependency set:

```bash
python -m pip install -e ".[dev,casim]"
```

Without `sktime`, the `CASIM` class falls back to a deterministic `CASIM-lite` random-convolution implementation. That fallback is intended for smoke tests and repository development, not for final manuscript-grade reproduction.

## Data layout

Keep `data/tep` and `data/fcc` empty in version control and add the datasets manually. The default loader expects one subfolder per class:

```text
data/tep/
├── class_01/
│   ├── run_0001.csv
│   └── run_0002.csv
├── class_02/
│   └── run_0001.csv
└── ...
```

Each CSV should be a binary alarm activity matrix with one row per time step and one column per alarm tag. A time column such as `Minutes`, `time`, `timestamp`, or `t` is detected and removed automatically. For example:

```text
Minutes,XMEAS1_HI,XMEAS1_LO,XMEAS2_HI,...
1,0,0,0,...
2,0,0,0,...
3,0,1,0,...
```

The loader returns arrays with shape `(n_episodes, n_alarm_tags, n_time_steps)` and zero-pads shorter runs at the end.

## Running a smoke test

Generate a tiny synthetic dataset, then run the benchmark on the smoke configuration:

```bash
python scripts/create_synthetic_dataset.py --output data/smoke --n-classes 3 --n-runs-per-class 12
python -m afc_robustness.cli run --config configs/smoke.yaml
python -m afc_robustness.cli plot --results-dir results/smoke
```

## Running the paper-style benchmark

After placing TEP/FCC data in `data/tep` and `data/fcc`, adapt `configs/tep.yaml` or `configs/fcc.yaml`, then run:

```bash
python -m afc_robustness.cli run --config configs/tep.yaml
python -m afc_robustness.cli run --config configs/fcc.yaml
```

Primary outputs are written to the configured result directory:

```text
results/<experiment>/
├── raw_predictions.csv              # one row per fold/draw/episode/progress point
├── degradation_profiles.csv          # M_{m,p}(s,rho)
├── progress_auc.csv                  # A_{m,p}(s)
├── scenario_scores.csv               # R_{m,p}
├── overall_scores_by_fold.csv         # R_avg, R_prod, R_min per fold
├── overall_scores_summary.csv         # mean/std over repeated folds
└── selected_hyperparameters.csv       # clean validation-selected parameters
```

## Notebook workflow

The notebooks deliberately contain little logic:

1. `notebooks/01_data_check.ipynb` checks dataset loading and event conversion.
2. `notebooks/02_run_benchmark.ipynb` runs a configured benchmark.
3. `notebooks/03_analyze_results.ipynb` reads saved outputs and creates figures.

## Methodological notes

The implementation separates within-episode alarm-log degradations from pipeline-induced episode-start delays. Within-episode perturbations preserve the episode horizon; delayed-detection perturbations shift the episode start and shorten the available horizon. Mixed perturbations are ordered compositions with trace repair between stages, so later stages act on the stream produced by earlier stages.

See `docs/MANUSCRIPT_ALIGNMENT.md` for a short list of consistency checks to address before submission.

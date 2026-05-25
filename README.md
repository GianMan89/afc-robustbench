# AFC-RobustBench

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research artifact](https://img.shields.io/badge/research-artifact-4b8bbe.svg)](#citation)

**AFC-RobustBench** is a reproducible research artifact for perturbation-based statistical robustness benchmarking of online alarm flood classification (AFC) methods. The benchmark evaluates how prefix-based AFC methods behave when the observed alarm-event stream is degraded by missing events, spurious events, timing uncertainty, mixed perturbations, or delayed alarm-flood detection.

This repository accompanies the manuscript:

> **AFC-RobustBench: Perturbation-Based Robustness Benchmarking of Online Alarm Flood Classification under Alarm-Log Degradations and Detection Delays**

The artifact is designed to support transparent reproduction of the paper experiments, extension to additional AFC methods, and reuse of the perturbation suite for robustness testing of alarm-event-stream classifiers.

---

## Overview

Online AFC assigns an unfolding alarm-flood episode to a predefined alarm-flood class while only a prefix of the episode is observable. In deployment, the available prefix can differ from curated historical alarm logs because alarm notifications can be missing, duplicated or spurious, timestamped inconsistently, or truncated by delayed alarm-flood detection.

AFC-RobustBench implements the following workflow:

1. Load labeled alarm-flood episodes and convert alarm-state trajectories into alarm-event streams.
2. Apply controlled perturbation functions over a severity grid and repeated Monte-Carlo draws.
3. Repair perturbed traces to valid alarm-event sequences.
4. Evaluate trained AFC methods online on a common reporting grid.
5. Aggregate prediction trajectories into degradation profiles, scalar robustness scores, and uncertainty estimates.

The benchmark is method-agnostic: any classifier that can consume a prefix representation and return a class prediction can be evaluated under the same perturbation protocol.

---

## Implemented benchmark components

The core package implements the main components of the robustness protocol:

- alarm-event and alarm-episode domain objects;
- conversion between alarm series, alarm activation sequences, and alarm sets;
- perturbation functions for missing events, spurious events, timing uncertainty, pipeline-induced episode-start delay, and ordered mixed perturbations;
- trace repair enforcing valid tag-wise `ACT`/`RTN` alternation;
- prefix-based online evaluation on event-count-driven or time-driven update schedules;
- aggregation into degradation profiles, progress-averaged performance, scenario-level robustness scores, and overall robustness summaries;
- plotting utilities for degradation profiles, robustness heatmaps, scalar robustness tables, and Pareto comparisons.

The paper benchmark evaluates seven AFC methods:

| Abbreviation | Representation | Method family |
|---|---:|---|
| `WDI-1NN` | alarm set | weighted dissimilarity 1-nearest neighbor |
| `JAC-1NN` | alarm set | Jaccard distance 1-nearest neighbor |
| `EAC-1NN` | alarm sequence | exponentially attenuated components 1-nearest neighbor |
| `MBW-LR` | alarm sequence | modified bag-of-words with logistic regression |
| `Hybrid AE+Trans.` | alarm sequence | hybrid autoencoder--Transformer with time-encoded histograms |
| `ACM-SVM` | alarm series | alarm coactivation matrix with support vector machine |
| `CASIM` | alarm series | convolutional-kernel features with ridge classifier ensemble |

---

## Repository layout

```text
.
├── configs/                    # YAML experiment configurations
├── data/                       # Local data directory
│   ├── tep/
│   └── fcc/
├── notebooks/                  # Reproducible notebook entry points
|── figures/                    # Generated publication figures
├── results/                    # Generated benchmark outputs
├── src/afc_robustness/         # Installable Python package
│   ├── data.py                 # Dataset loading and padding
│   ├── domain.py               # AlarmEvent and AlarmEpisode domain objects
│   ├── perturbations.py        # Perturbation suite and mixed compositions
│   ├── repair.py               # Trace repair
│   ├── representations.py      # Alarm-series, set, and sequence representations
│   ├── online.py               # Prefix-based online evaluation
│   ├── metrics.py              # Robustness aggregation
│   ├── experiment.py           # Cross-validation benchmark runner
│   ├── plotting.py             # Result visualizations
│   └── models/                 # AFC method implementations
└── tests/                      # Unit tests for core semantics
```

Large raw data outputs, i.e., raw_predictions.csv and perturbation_diagnostics_raw.csv, are not committed due to file size restrictions on GitHub. All output data can be generated using the provided notebooks and scripts.

---

## Installation

Create a fresh Python environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The package requires Python 3.10 or newer. The optional `casim` dependency set installs the backend used for the full CASIM-style convolutional feature implementation:

```bash
python -m pip install -e ".[dev,casim]"
```

Without the optional backend, the repository can still be used for smoke tests and development, but final manuscript-grade reproduction should use the dependencies specified for the relevant experiment configuration.

---

## Data

Raw datasets are not tracked in the repository. Place the data manually under `data/tep/` and `data/fcc/`.

The paper uses two process-industry alarm datasets:

- **Tennessee-Eastman Process alarm dataset**: `https://dx.doi.org/10.21227/326k-qr90`
- **Fluidized Catalytic Cracking alarm dataset**: `https://doi.org/10.60517/2v23vv393`

The default loader expects one subfolder per class:

```text
data/tep/
├── class_01/
│   ├── run_0001.csv
│   └── run_0002.csv
├── class_02/
│   └── run_0001.csv
└── ...
```

Each CSV file should contain a binary alarm activity matrix with one row per time step and one column per alarm tag. A time column such as `Minutes`, `time`, `timestamp`, or `t` is detected and removed automatically.

Example:

```text
Minutes,XMEAS1_HI,XMEAS1_LO,XMEAS2_HI,...
1,0,0,0,...
2,0,0,0,...
3,0,1,0,...
```

The loader returns arrays with shape

```text
(n_episodes, n_alarm_tags, n_time_steps)
```

and zero-pads shorter runs at the end.

---

## Reproducing the paper-style experiments

After placing the TEP and FCC data in the expected directories, run the configured benchmarks:

```bash
afc-benchmark run --config configs/tep.yaml
afc-benchmark run --config configs/fcc.yaml
```

---

## Main output files

Each experiment writes CSV files to the configured result directory:

```text
results/<experiment>/
├── raw_predictions.csv              # prediction per fold/draw/episode/progress point
├── degradation_profiles.csv          # Mbar_{m,p}(s, pi)
├── progress_auc.csv                  # Abar_{m,p}(s)
├── scenario_scores.csv               # Rbar_{m,p}
├── overall_scores_by_fold.csv         # R_avg, R_min per fold/draw unit
├── overall_scores_summary.csv         # mean/std summaries
└── selected_hyperparameters.csv       # clean validation-selected parameters
```

The main reporting quantities are:

- `degradation_profiles.csv`: online accuracy over perturbation severity and observation progress;
- `progress_auc.csv`: online accuracy averaged over the reporting grid;
- `scenario_scores.csv`: normalized severity-integrated robustness scores for each method and scenario;
- `overall_scores_summary.csv`: average and worst-scenario robustness summaries over the selected perturbation scenarios.

---

## Perturbation scenarios

The paper benchmark uses four base perturbation families:

| Scenario | Interpretation |
|---|---|
| missing events | loss of alarm notifications within an extracted episode |
| spurious events | additional alarm notifications inserted into the event stream |
| timing uncertainty | timestamp shifts that can alter recovered event ordering |
| episode-start delay | delayed flood detection or conservative segmentation that truncates early context |

Mixed perturbations are evaluated as ordered compositions. Trace repair is applied between stages, so later perturbation stages act on the repaired output of earlier stages.

---

## Reproducibility notes

The benchmark uses fixed train-test splits, severity grids, Monte-Carlo perturbation draws, update times, and reporting grids in the paper configurations. For manuscript-grade reproduction, use the configuration files in `configs/` and keep the following items unchanged:

- cross-validation split strategy;
- perturbation scenario list and severity grid;
- number of Monte-Carlo draws;
- native online update schedule;
- reporting grid;
- method hyperparameters.

For development or ablation studies, modify the YAML configuration files rather than changing the package internals.

---

## Testing

Run the unit tests with:

```bash
pytest
```

The tests focus on core semantic behavior such as event conversion, perturbation validity, trace repair, and aggregation consistency.

---

## Citation

After using this repository, please cite the corresponding paper. The final bibliographic entry and DOI will be added after publication.

```bibtex
@article{Manca2026_AFCRobustBench,
  author  = {Manca, Gianluca and Najafi, Amirhossein and Tamascelli, Nicola and Kunze, Franz C. and Dix, Marcel and Hollender, Martin and Fay, Alexander and Chen, Tongwen},
  title   = {{AFC-RobustBench}: Perturbation-Based Robustness Benchmarking of Online Alarm Flood Classification under Alarm-Log Degradations and Detection Delays},
  journal = {TBD},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Please also cite the datasets when using them:

```bibtex
@misc{Manca2020_TEPAlarmDataset,
  author       = {Manca, Gianluca},
  title        = {{Tennessee-Eastman-Process} Alarm Management Dataset},
  howpublished = {IEEE Dataport},
  year         = {2020},
  doi          = {10.21227/326k-qr90}
}
```

```bibtex
@misc{Kunze2025_FCCAlarmDataset,
  author       = {Kunze, Franz C. and Manca, Gianluca and Fay, Alexander},
  title        = {{FCC} Alarm Dataset for Alarm Flood Classification},
  howpublished = {ReSeeD},
  year         = {2025},
  doi          = {10.60517/2v23vv393}
}
```

---

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).

---

## Contact

For questions about the benchmark, the paper experiments, or reproducibility, open an issue in this repository or contact the corresponding author listed in the manuscript.

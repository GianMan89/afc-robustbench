.PHONY: install test lint smoke

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

smoke:
	python scripts/create_synthetic_dataset.py --output data/smoke --n-classes 3 --n-runs-per-class 12
	python -m afc_robustness.cli run --config configs/smoke.yaml
	python -m afc_robustness.cli plot --results-dir results/smoke

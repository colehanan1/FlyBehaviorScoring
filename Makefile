PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTHON_BIN := $(VENV)/bin/python

.PHONY: venv install test demo clean

venv:
$(PYTHON) -m venv $(VENV)

install: venv
$(PIP) install --upgrade pip
$(PIP) install -e .

test:
$(PYTHON_BIN) -m pytest -q

demo: install
mkdir -p outputs/demo
$(PYTHON_BIN) - <<'PY'
from pathlib import Path
import pandas as pd
from flypca.synthetic import generate_synthetic_trials

trials, meta = generate_synthetic_trials(n_flies=4, trials_per_fly=10)
rows = []
for trial in trials:
    df = pd.DataFrame({
        "time": trial.time,
        "distance": trial.distance,
        "trial_id": trial.trial_id,
        "fly_id": trial.fly_id,
        "odor_on_idx": trial.odor_on_idx,
        "odor_off_idx": trial.odor_off_idx,
        "fps": trial.fps,
    })
    rows.append(df)
data = pd.concat(rows, ignore_index=True)
data.to_csv("outputs/demo/data.csv", index=False)
meta.to_csv("outputs/demo/meta.csv", index=False)
PY
$(PYTHON_BIN) -m flypca.cli fit-lag-pca --data outputs/demo/data.csv --config config/example.yaml --out outputs/demo/lagpca.joblib
$(PYTHON_BIN) -m flypca.cli project --model outputs/demo/lagpca.joblib --data outputs/demo/data.csv --out outputs/demo/projections
$(PYTHON_BIN) -m flypca.cli features --data outputs/demo/data.csv --config config/example.yaml --model outputs/demo/lagpca.joblib --projections outputs/demo/projections --out outputs/demo/features.parquet
$(PYTHON_BIN) -m flypca.cli cluster --features outputs/demo/features.parquet --out outputs/demo/clusters.csv
$(PYTHON_BIN) -m flypca.cli report --features outputs/demo/features.parquet --clusters outputs/demo/clusters.csv --model outputs/demo/lagpca.joblib --projections outputs/demo/projections --out-dir outputs/demo

clean:
rm -rf $(VENV) outputs/demo

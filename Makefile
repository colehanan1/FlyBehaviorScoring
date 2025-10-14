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
mkdir -p artifacts/demo
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
data.to_csv("artifacts/demo/data.csv", index=False)
meta.to_csv("artifacts/demo/meta.csv", index=False)
PY
$(PYTHON_BIN) -m flypca.cli fit-lag-pca --data artifacts/demo/data.csv --config configs/default.yaml --out artifacts/demo/lagpca.joblib
$(PYTHON_BIN) -m flypca.cli project --model artifacts/demo/lagpca.joblib --data artifacts/demo/data.csv --out artifacts/demo/projections
$(PYTHON_BIN) -m flypca.cli features --data artifacts/demo/data.csv --config configs/default.yaml --model artifacts/demo/lagpca.joblib --projections artifacts/demo/projections --out artifacts/demo/features.parquet
$(PYTHON_BIN) -m flypca.cli cluster --features artifacts/demo/features.parquet --out artifacts/demo/clusters.csv
$(PYTHON_BIN) -m flypca.cli report --features artifacts/demo/features.parquet --clusters artifacts/demo/clusters.csv --model artifacts/demo/lagpca.joblib --projections artifacts/demo/projections --out-dir artifacts/demo

clean:
rm -rf $(VENV) artifacts/demo

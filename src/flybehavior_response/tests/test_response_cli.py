from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from flybehavior_response.io import LABEL_COLUMN


def _run_cli(args: list[str]) -> None:
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run([sys.executable, "-m", "flybehavior_response.cli", *args], check=True, env=env)


def _write_sample(tmp_path: Path) -> tuple[Path, Path]:
    data = pd.DataFrame(
        {
            "fly": ["a", "b"],
            "fly_number": [1, 2],
            "trial_label": ["t1", "t2"],
            "dir_val_0": [0.1, 0.2],
            "dir_val_1": [0.2, 0.3],
            "AUC-During": [1.0, 1.1],
            "TimeToPeak-During": [5.0, 5.5],
            "Peak-Value": [0.9, 1.0],
        }
    )
    labels = pd.DataFrame(
        {
            "fly": ["a", "b"],
            "fly_number": [1, 2],
            "trial_label": ["t1", "t2"],
            LABEL_COLUMN: [0, 4],
        }
    )
    data_path = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(labels_path, index=False)
    return data_path, labels_path


def test_cli_help_runs() -> None:
    _run_cli(["--help"])


def test_cli_train_eval_dry_run(tmp_path: Path) -> None:
    data_path, labels_path = _write_sample(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    _run_cli(
        [
            "train",
            "--data-csv",
            str(data_path),
            "--labels-csv",
            str(labels_path),
            "--artifacts-dir",
            str(artifacts_dir),
            "--model",
            "logreg",
            "--logreg-max-iter",
            "200",
            "--logreg-solver",
            "liblinear",
        ]
    )
    _run_cli(
        [
            "eval",
            "--data-csv",
            str(data_path),
            "--labels-csv",
            str(labels_path),
            "--artifacts-dir",
            str(artifacts_dir),
            "--dry-run",
        ]
    )

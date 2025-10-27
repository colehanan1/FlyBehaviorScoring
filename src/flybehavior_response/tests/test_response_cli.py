from __future__ import annotations

import os
import subprocess
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
    flies = [f"fly_{idx}" for idx in range(10)]
    trial_ids = [f"t{idx + 1}" for idx in range(10)]
    data = pd.DataFrame(
        {
            "dataset": ["opto_EB"] * 10,
            "fly": flies,
            "fly_number": list(range(1, 11)),
            "trial_type": ["testing"] * 10,
            "trial_label": trial_ids,
            "testing_trial": trial_ids,
            "dir_val_0": [0.1 + 0.05 * idx for idx in range(10)],
            "dir_val_1": [0.2 + 0.05 * idx for idx in range(10)],
            "AUC-During": [1.0 + 0.05 * ((-1) ** idx) for idx in range(10)],
            "TimeToPeak-During": [5.0 + 0.2 * ((-1) ** idx) for idx in range(10)],
            "Peak-Value": [0.9 + 0.03 * ((-1) ** idx) for idx in range(10)],
        }
    )
    labels = pd.DataFrame(
        {
            "dataset": ["opto_EB"] * 10,
            "fly": flies,
            "fly_number": list(range(1, 11)),
            "trial_type": ["testing"] * 10,
            "trial_label": trial_ids,
            LABEL_COLUMN: [0, 4] * 5,
        }
    )
    data_path = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(labels_path, index=False)
    return data_path, labels_path


def _write_geometry(tmp_path: Path) -> tuple[Path, Path]:
    records = []
    for fly_idx in range(2):
        fly = f"fly_geom_{fly_idx}"
        fly_number = fly_idx + 1
        for trial_idx in range(2):
            trial_label = f"tg{fly_idx}_{trial_idx}"
            for frame_idx in range(3):
                records.append(
                    {
                        "dataset": "geom_ds",
                        "fly": fly,
                        "fly_number": fly_number,
                        "trial_type": "testing",
                        "trial_label": trial_label,
                        "frame_idx": frame_idx,
                        "x": float(fly_idx + trial_idx) + 0.1 * frame_idx,
                        "y": float(trial_idx) - 0.05 * frame_idx,
                    }
                )
    frames_df = pd.DataFrame(records)
    labels = []
    for fly_idx in range(2):
        fly = f"fly_geom_{fly_idx}"
        fly_number = fly_idx + 1
        for trial_idx in range(2):
            labels.append(
                {
                    "dataset": "geom_ds",
                    "fly": fly,
                    "fly_number": fly_number,
                    "trial_type": "testing",
                    "trial_label": f"tg{fly_idx}_{trial_idx}",
                    LABEL_COLUMN: 0 if trial_idx == 0 else 3,
                }
            )
    labels_df = pd.DataFrame(labels)
    frames_path = tmp_path / "geometry_frames.csv"
    labels_path = tmp_path / "geometry_labels.csv"
    frames_df.to_csv(frames_path, index=False)
    labels_df.to_csv(labels_path, index=False)
    return frames_path, labels_path


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


def test_cli_predict_filters_single_trial(tmp_path: Path) -> None:
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
        ]
    )

    run_dirs = [path for path in artifacts_dir.iterdir() if path.is_dir()]
    assert run_dirs, "Expected an artifact directory containing trained models"
    run_dir = max(run_dirs, key=lambda item: item.stat().st_mtime)
    model_path = run_dir / "model_logreg.joblib"
    assert model_path.exists(), "Logistic regression model joblib missing"

    predictions_csv = tmp_path / "predictions.csv"
    _run_cli(
        [
            "predict",
            "--data-csv",
            str(data_path),
            "--model-path",
            str(model_path),
            "--output-csv",
            str(predictions_csv),
            "--fly",
            "fly_1",
            "--fly-number",
            "2",
            "--testing-trial",
            "t2",
        ]
    )

    output = pd.read_csv(predictions_csv)
    assert len(output) == 1
    assert output.loc[0, "fly"] == "fly_1"
    assert output.loc[0, "fly_number"] == 2
    assert output.loc[0, "testing_trial"] == "t2"
    assert "prediction" in output.columns


def test_cli_geometry_prepare_train_predict(tmp_path: Path) -> None:
    frames_path, labels_path = _write_geometry(tmp_path)
    prepare_dir = tmp_path / "prepare_artifacts"
    _run_cli(
        [
            "prepare",
            "--data-csv",
            str(frames_path),
            "--labels-csv",
            str(labels_path),
            "--artifacts-dir",
            str(prepare_dir),
            "--aggregate-geometry",
            "--geom-chunk-size",
            "2",
            "--aggregate-format",
            "csv",
        ]
    )
    aggregated = prepare_dir / "geometry_aggregates.csv"
    assert aggregated.exists()

    artifacts_dir = tmp_path / "geom_artifacts"
    _run_cli(
        [
            "train",
            "--geometry-frames",
            str(frames_path),
            "--labels-csv",
            str(labels_path),
            "--artifacts-dir",
            str(artifacts_dir),
            "--model",
            "logreg",
            "--test-size",
            "0.5",
        ]
    )

    run_dirs = [path for path in artifacts_dir.iterdir() if path.is_dir()]
    assert run_dirs, "Expected artifacts from geometry training"
    run_dir = max(run_dirs, key=lambda item: item.stat().st_mtime)
    model_path = run_dir / "model_logreg.joblib"
    predictions_csv = tmp_path / "geom_predictions.csv"
    _run_cli(
        [
            "predict",
            "--geometry-frames",
            str(frames_path),
            "--model-path",
            str(model_path),
            "--output-csv",
            str(predictions_csv),
        ]
    )
    output = pd.read_csv(predictions_csv)
    assert not output.empty
    assert "prediction" in output.columns

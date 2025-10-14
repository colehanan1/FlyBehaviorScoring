"""Tests for flypca.io data loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flypca.io import load_trials


def _write_csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_load_trials_stacked_with_column_mapping(tmp_path: Path) -> None:
    data = pd.DataFrame(
            {
                "id": ["t1", "t1", "t2", "t2"],
                "fly": ["f1", "f1", "f2", "f2"],
                "odor_on": [1, 1, 1, 1],
                "distance_mm": [0.0, 0.2, 0.1, 0.5],
                "time_s": [0.0, 0.025, 0.0, 0.025],
            }
    )
    path = _write_csv(tmp_path, "stacked.csv", data)
    config = {
        "fps": 40.0,
        "io": {
            "format": "stacked",
            "stacked": {
                "trial_id_column": "id",
                "fly_id_column": "fly",
                "distance_column": "distance_mm",
                "time_column": "time_s",
                "odor_on_column": "odor_on",
            },
        },
    }
    trials = load_trials(path, config)
    assert len(trials) == 2
    assert trials[0].trial_id == "t1"
    np.testing.assert_allclose(trials[0].distance, [0.0, 0.2])


def test_load_trials_wide(tmp_path: Path) -> None:
    data = pd.DataFrame(
            {
                "trial_label": ["t1", "t2"],
                "fly_name": ["f1", "f2"],
                "odor_on_frame": [1, 1],
                "fps": [40.0, 40.0],
                "dir_val_0": [0.0, 0.1],
                "dir_val_1": [0.4, 0.2],
                "dir_val_2": [0.6, np.nan],
            }
    )
    path = _write_csv(tmp_path, "wide.csv", data)
    config = {
        "fps": 40.0,
        "io": {
            "format": "wide",
            "wide": {
                "trial_id_column": "trial_label",
                "fly_id_column": "fly_name",
                "odor_on_column": "odor_on_frame",
                "fps_column": "fps",
                "time_columns": {"prefix": "dir_val_"},
            },
        },
    }
    trials = load_trials(path, config)
    assert len(trials) == 2
    assert trials[1].trial_id == "t2"
    # ensure NaNs trimmed
    np.testing.assert_allclose(trials[1].distance, [0.1, 0.2])
    assert trials[1].odor_on_idx == 1


def test_load_trials_wide_with_constants_and_templates(tmp_path: Path) -> None:
    samples = {f"dir_val_{i}": [0.1 + 0.01 * i] for i in range(12)}
    samples.update({"fly": ["flyA"], "trial_label": ["trial_01"]})
    data = pd.DataFrame(samples)
    path = _write_csv(tmp_path, "wide_constants.csv", data)
    config = {
        "fps": 40.0,
        "io": {
            "format": "wide",
            "wide": {
                "trial_id_column": "trial_label",
                "trial_id_template": "{fly}-{trial_label}",
                "fly_id_column": "fly",
                "odor_on_value": 3,
                "odor_off_value": 10,
                "time_columns": {"prefix": "dir_val_"},
            },
        },
    }
    trials = load_trials(path, config)
    assert len(trials) == 1
    trial = trials[0]
    assert trial.trial_id == "flyA-trial_01"
    assert trial.fly_id == "flyA"
    assert trial.odor_on_idx == 3
    assert trial.odor_off_idx == 10
    # fps should fall back to config default when column missing or NaN
    assert trial.fps == 40.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flybehavior_response.io import (
    LABEL_COLUMN,
    LABEL_INTENSITY_COLUMN,
    DataValidationError,
    RAW_TRACE_PREFIXES,
    load_and_merge,
)


@pytest.fixture
def sample_csvs(tmp_path: Path) -> tuple[Path, Path]:
    data = pd.DataFrame(
        {
            "fly": ["a", "b"],
            "fly_number": [1, 2],
            "trial_label": ["t1", "t2"],
            "dir_val_0": [0.1, 0.2],
            "dir_val_10": [0.3, 0.4],
            "dir_val_3600": [0.5, 0.6],
            "dir_val_3601": [0.7, 0.8],
            "AUC-Before": [1.0, 2.0],
            "AUC-During": [1.1, 2.1],
            "AUC-After": [1.2, 2.2],
            "TimeToPeak-During": [5.0, 6.0],
            "Peak-Value": [0.9, 1.1],
        }
    )
    labels = pd.DataFrame(
        {
            "fly": ["a", "b"],
            "fly_number": [1, 2],
            "trial_label": ["t1", "t2"],
            LABEL_COLUMN: [0, 5],
        }
    )
    data_path = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(labels_path, index=False)
    return data_path, labels_path


def test_load_and_merge_filters_traces(sample_csvs: tuple[Path, Path]) -> None:
    data_path, labels_path = sample_csvs
    dataset = load_and_merge(data_path, labels_path)
    assert dataset.frame.shape[0] == 2
    assert "dir_val_3601" not in dataset.frame.columns
    assert dataset.trace_columns[0] == "dir_val_0"
    assert dataset.trace_columns[-1] == "dir_val_3600"
    assert dataset.frame[LABEL_COLUMN].tolist() == [0, 1]
    assert dataset.frame[LABEL_INTENSITY_COLUMN].tolist() == [0, 5]
    assert dataset.sample_weights.tolist() == [1.0, 5.0]


def test_load_and_merge_invalid_labels(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "fly": ["a"],
            "fly_number": [1],
            "trial_label": ["t1"],
            "dir_val_0": [0.1],
            "AUC-Before": [1.0],
            "AUC-During": [1.2],
            "AUC-After": [1.3],
            "TimeToPeak-During": [5.0],
            "Peak-Value": [0.9],
        }
    )
    labels = pd.DataFrame(
        {
            "fly": ["a"],
            "fly_number": [1],
            "trial_label": ["t1"],
            LABEL_COLUMN: [-1],
        }
    )
    data_path = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(labels_path, index=False)
    with pytest.raises(DataValidationError):
        load_and_merge(data_path, labels_path)


def test_load_and_merge_non_integer_labels(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "fly": ["a"],
            "fly_number": [1],
            "trial_label": ["t1"],
            "dir_val_0": [0.1],
            "AUC-Before": [1.0],
            "AUC-During": [1.2],
            "AUC-After": [1.3],
            "TimeToPeak-During": [5.0],
            "Peak-Value": [0.9],
        }
    )
    labels = pd.DataFrame(
        {
            "fly": ["a"],
            "fly_number": [1],
            "trial_label": ["t1"],
            LABEL_COLUMN: [1.5],
        }
    )
    data_path = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(labels_path, index=False)
    with pytest.raises(DataValidationError):
        load_and_merge(data_path, labels_path)


def test_load_and_merge_duplicate_keys(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "fly": ["a", "a"],
            "fly_number": [1, 1],
            "trial_label": ["t1", "t1"],
            "dir_val_0": [0.1, 0.2],
            "AUC-Before": [1.0, 2.0],
            "AUC-During": [1.2, 2.2],
            "AUC-After": [1.3, 2.3],
            "TimeToPeak-During": [5.0, 6.0],
            "Peak-Value": [0.9, 1.0],
        }
    )
    labels = pd.DataFrame(
        {
            "fly": ["a"],
            "fly_number": [1],
            "trial_label": ["t1"],
            LABEL_COLUMN: [1],
        }
    )
    data_path = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(labels_path, index=False)
    with pytest.raises(DataValidationError):
        load_and_merge(data_path, labels_path)


def test_load_and_merge_detects_raw_prefixes(tmp_path: Path) -> None:
    frames = {
        "fly": ["a", "b"],
        "fly_number": [1, 2],
        "trial_label": ["t1", "t2"],
        "AUC-During": [0.5, 0.7],
        "TimeToPeak-During": [5.0, 6.0],
        "Peak-Value": [0.9, 1.2],
    }
    for prefix in RAW_TRACE_PREFIXES:
        frames[f"{prefix}0"] = [0.1, 0.2]
        frames[f"{prefix}1"] = [0.3, 0.4]
    data = pd.DataFrame(frames)
    labels = pd.DataFrame(
        {
            "fly": ["a", "b"],
            "fly_number": [1, 2],
            "trial_label": ["t1", "t2"],
            LABEL_COLUMN: [0, 5],
        }
    )
    data_path = tmp_path / "raw_data.csv"
    labels_path = tmp_path / "raw_labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(labels_path, index=False)

    dataset = load_and_merge(data_path, labels_path)

    assert dataset.trace_prefixes == list(RAW_TRACE_PREFIXES)
    assert dataset.trace_columns[:4] == [f"{RAW_TRACE_PREFIXES[0]}0", f"{RAW_TRACE_PREFIXES[0]}1", f"{RAW_TRACE_PREFIXES[1]}0", f"{RAW_TRACE_PREFIXES[1]}1"]


def test_load_and_merge_allows_trace_only_inputs(tmp_path: Path) -> None:
    frames = {
        "dataset": ["d", "d"],
        "fly": ["f1", "f2"],
        "fly_number": [1, 2],
        "trial_label": ["t1", "t2"],
        "trial_type": ["testing", "testing"],
    }
    for prefix in RAW_TRACE_PREFIXES:
        frames[f"{prefix}0"] = [0.1, 0.2]
        frames[f"{prefix}1"] = [0.3, 0.4]
    data = pd.DataFrame(frames)
    labels = pd.DataFrame(
        {
            "dataset": ["d", "d"],
            "fly": ["f1", "f2"],
            "fly_number": [1, 2],
            "trial_label": ["t1", "t2"],
            LABEL_COLUMN: [0, 1],
        }
    )
    data_path = tmp_path / "trace_only.csv"
    labels_path = tmp_path / "trace_only_labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(labels_path, index=False)

    dataset = load_and_merge(data_path, labels_path, trace_prefixes=RAW_TRACE_PREFIXES)

    assert dataset.trace_prefixes == list(RAW_TRACE_PREFIXES)
    assert dataset.feature_columns == []

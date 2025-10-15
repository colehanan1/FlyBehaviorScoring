from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flybehavior_response.prepare_raw import DEFAULT_PREFIXES, prepare_raw


@pytest.fixture
def sample_tables(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(123)
    data = pd.DataFrame(
        {
            "dataset": ["d"] * 4,
            "fly": ["f"] * 4,
            "fly_number": [1] * 4,
            "trial_type": ["testing"] * 4,
            "testing_trial": [1, 2, 3, 4],
            **{f"eye_x_f{i}": rng.normal(size=4) for i in range(5)},
            **{f"eye_y_f{i}": rng.normal(size=4) for i in range(5)},
            **{f"prob_x_f{i}": rng.normal(size=4) for i in range(5)},
            **{f"prob_y_f{i}": rng.normal(size=4) for i in range(5)},
        }
    )
    labels = pd.DataFrame(
        {
            "dataset": ["d"] * 4,
            "fly": ["f"] * 4,
            "fly_number": [1] * 4,
            "trial_type": ["testing"] * 4,
            "testing_trial": [1, 2, 3, 4],
            "trial_label": [0, 1, 0, 1],
        }
    )
    data_path = tmp_path / "data.csv"
    label_path = tmp_path / "labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(label_path, index=False)
    return data_path, label_path


def test_prepare_raw_roundtrip(sample_tables: tuple[Path, Path], tmp_path: Path) -> None:
    data_path, label_path = sample_tables
    out_path = tmp_path / "prepared.csv"
    prepared = prepare_raw(
        data_csv=data_path,
        labels_csv=label_path,
        out_path=out_path,
        series_prefixes=DEFAULT_PREFIXES,
        verbose=True,
    )
    assert out_path.exists()
    assert list(prepared.columns[:10]) == [
        "dataset",
        "fly",
        "fly_number",
        "trial_type",
        "testing_trial",
        "fps",
        "odor_on_idx",
        "odor_off_idx",
        "total_frames",
        "label",
    ]
    assert {col for col in prepared.columns if col.startswith("eye_x_f")} == {
        "eye_x_f0",
        "eye_x_f1",
        "eye_x_f2",
        "eye_x_f3",
        "eye_x_f4",
    }
    assert prepared["total_frames"].iloc[0] == 5


def test_prepare_raw_truncation_and_dirval(sample_tables: tuple[Path, Path], tmp_path: Path) -> None:
    data_path, label_path = sample_tables
    out_path = tmp_path / "prepared_dirval.csv"
    prepared = prepare_raw(
        data_csv=data_path,
        labels_csv=label_path,
        out_path=out_path,
        truncate_before=1,
        truncate_after=1,
        compute_dir_val=True,
        odor_on_idx=2,
        odor_off_idx=3,
    )
    assert prepared["total_frames"].iloc[0] == 4
    assert any(col.startswith("dir_val_f") for col in prepared.columns)


def test_prepare_raw_row_order_alignment(sample_tables: tuple[Path, Path], tmp_path: Path) -> None:
    data_path, label_path = sample_tables
    labels_no_trial = pd.read_csv(label_path).drop(columns=["testing_trial"])
    fallback_labels_path = tmp_path / "labels_no_trial.csv"
    labels_no_trial.to_csv(fallback_labels_path, index=False)
    prepared = prepare_raw(
        data_csv=data_path,
        labels_csv=fallback_labels_path,
        out_path=tmp_path / "prepared_fallback.csv",
    )
    assert prepared["label"].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_prepare_raw_rejects_per_fly_table(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "dataset": ["d"],
            "fly": ["f"],
            "fly_number": [1],
            "trial_type": ["testing"],
            "testing_trial": [1],
            **{f"eye_x_f{i}": [0.1 * i] for i in range(5)},
            **{f"eye_y_f{i}": [0.2 * i] for i in range(5)},
            **{f"prob_x_f{i}": [0.3 * i] for i in range(5)},
            **{f"prob_y_f{i}": [0.4 * i] for i in range(5)},
        }
    )
    labels = pd.DataFrame(
        {
            "dataset": ["d"],
            "fly": ["f"],
            "fly_number": [1],
            "trial_type": ["testing"],
            "testing_trial": [1],
            "trial_label": [1],
        }
    )
    data_path = tmp_path / "agg.csv"
    label_path = tmp_path / "agg_labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(label_path, index=False)
    with pytest.raises(ValueError, match="per-trial CSV"):
        prepare_raw(
            data_csv=data_path,
            labels_csv=label_path,
            out_path=tmp_path / "should_fail.csv",
        )


def test_prepare_raw_from_matrix(tmp_path: Path) -> None:
    rng = np.random.default_rng(321)
    matrix = rng.normal(size=(3, 6, 4))
    matrix_path = tmp_path / "coords.npy"
    np.save(matrix_path, matrix)

    metadata = [
        {
            "dataset": "d",
            "fly": "f1",
            "fly_number": 1,
            "trial_type": "testing",
            "testing_trial": idx + 1,
        }
        for idx in range(3)
    ]
    meta_payload = {
        "metadata": metadata,
        "layout": "trial_time_channel",
        "channel_prefixes": DEFAULT_PREFIXES,
    }
    meta_path = tmp_path / "coords_meta.json"
    meta_path.write_text(json.dumps(meta_payload))

    labels = pd.DataFrame(
        {
            "dataset": ["d"] * 3,
            "fly": ["f1"] * 3,
            "fly_number": [1] * 3,
            "trial_type": ["testing"] * 3,
            "testing_trial": [1, 2, 3],
            "trial_label": [0, 1, 0],
        }
    )
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)

    prepared = prepare_raw(
        data_npy=matrix_path,
        matrix_meta=meta_path,
        labels_csv=labels_path,
        out_path=tmp_path / "prepared_matrix.csv",
    )

    assert prepared.shape[0] == 3
    assert prepared["total_frames"].iat[0] == 6
    assert prepared.filter(regex=r"^eye_x_f").shape[1] == 6

    # Validate alternate layout handling.
    alt_matrix = np.transpose(matrix, (0, 2, 1))
    alt_matrix_path = tmp_path / "coords_nct.npy"
    np.save(alt_matrix_path, alt_matrix)
    meta_payload["layout"] = "trial_channel_time"
    meta_path.write_text(json.dumps(meta_payload))
    prepared_alt = prepare_raw(
        data_npy=alt_matrix_path,
        matrix_meta=meta_path,
        labels_csv=labels_path,
        out_path=tmp_path / "prepared_matrix_alt.csv",
    )
    assert prepared_alt.equals(prepared)

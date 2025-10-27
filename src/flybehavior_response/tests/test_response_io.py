from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flybehavior_response.io import (
    LABEL_COLUMN,
    LABEL_INTENSITY_COLUMN,
    DataValidationError,
    RAW_TRACE_PREFIXES,
    aggregate_trials,
    load_and_merge,
    load_geom_frames,
    load_geometry_dataset,
)


@pytest.fixture
def sample_csvs(tmp_path: Path) -> tuple[Path, Path]:
    data = pd.DataFrame(
        {
            "dataset": ["opto_EB", "opto_EB"],
            "fly": ["a", "b"],
            "fly_number": [1, 2],
            "trial_type": ["testing", "testing"],
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
            "dataset": ["opto_EB", "opto_EB"],
            "fly": ["a", "b"],
            "fly_number": [1, 2],
            "trial_type": ["testing", "testing"],
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
    assert dataset.frame["dataset"].tolist() == ["opto_EB", "opto_EB"]
    assert dataset.frame["trial_type"].tolist() == ["testing", "testing"]
    assert "dir_val_3601" not in dataset.frame.columns
    assert dataset.trace_columns[0] == "dir_val_0"
    assert dataset.trace_columns[-1] == "dir_val_3600"
    assert dataset.frame[LABEL_COLUMN].tolist() == [0, 1]
    assert dataset.frame[LABEL_INTENSITY_COLUMN].tolist() == [0, 5]
    assert dataset.sample_weights.tolist() == [1.0, 5.0]


def test_load_and_merge_invalid_labels(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "dataset": ["opto_EB"],
            "fly": ["a"],
            "fly_number": [1],
            "trial_type": ["testing"],
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
            "dataset": ["opto_EB"],
            "fly": ["a"],
            "fly_number": [1],
            "trial_type": ["testing"],
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
            "dataset": ["opto_EB"],
            "fly": ["a"],
            "fly_number": [1],
            "trial_type": ["testing"],
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
            "dataset": ["opto_EB"],
            "fly": ["a"],
            "fly_number": [1],
            "trial_type": ["testing"],
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
            "dataset": ["opto_EB", "opto_EB"],
            "fly": ["a", "a"],
            "fly_number": [1, 1],
            "trial_type": ["testing", "testing"],
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
            "dataset": ["opto_EB"],
            "fly": ["a"],
            "fly_number": [1],
            "trial_type": ["testing"],
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
        "dataset": ["opto_EB", "opto_EB"],
        "fly": ["a", "b"],
        "fly_number": [1, 2],
        "trial_type": ["testing", "testing"],
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
            "dataset": ["opto_EB", "opto_EB"],
            "fly": ["a", "b"],
            "fly_number": [1, 2],
            "trial_type": ["testing", "testing"],
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
            "trial_type": ["testing", "testing"],
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


def test_load_and_merge_dir_val_trace_only(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "dataset": ["opto_EB", "opto_EB"],
            "fly": ["a", "b"],
            "fly_number": [1, 2],
            "trial_type": ["testing", "testing"],
            "trial_label": ["t1", "t2"],
            "dir_val_0": [0.1, 0.2],
            "dir_val_1": [0.3, 0.4],
        }
    )
    labels = pd.DataFrame(
        {
            "dataset": ["opto_EB", "opto_EB"],
            "fly": ["a", "b"],
            "fly_number": [1, 2],
            "trial_type": ["testing", "testing"],
            "trial_label": ["t1", "t2"],
            LABEL_COLUMN: [0, 5],
        }
    )
    data_path = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(labels_path, index=False)

    dataset = load_and_merge(data_path, labels_path)

    assert dataset.trace_prefixes == ["dir_val_"]
    assert dataset.trace_columns == ["dir_val_0", "dir_val_1"]
    assert dataset.feature_columns == []


def test_load_and_merge_requires_dataset_column(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "fly": ["a"],
            "fly_number": [1],
            "trial_type": ["testing"],
            "trial_label": ["t1"],
            "dir_val_0": [0.1],
        }
    )
    labels = pd.DataFrame(
        {
            "fly": ["a"],
            "fly_number": [1],
            "trial_type": ["testing"],
            "trial_label": ["t1"],
            LABEL_COLUMN: [0],
        }
    )
    data_path = tmp_path / "missing_dataset.csv"
    labels_path = tmp_path / "missing_dataset_labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(labels_path, index=False)

    with pytest.raises(DataValidationError):
        load_and_merge(data_path, labels_path)


@pytest.fixture
def geometry_csvs(tmp_path: Path) -> tuple[Path, Path]:
    frames = pd.DataFrame(
        {
            "dataset": [
                " opto_EB",
                "opto_EB",
                "opto_EB ",
                " opto_EB",
                "opto_EB",
                "opto_EB ",
            ],
            "fly": [" a", "a", "a ", "b", "b", "b"],
            "fly_number": ["1", "1", "1", "2", "2", "2"],
            "trial_type": [
                " testing",
                "testing",
                "testing ",
                " testing",
                "testing",
                "testing ",
            ],
            "trial_label": ["t1", "t1", "t1", "t2", "t2", "t2"],
            "frame_idx": [0, 1, 2, 0, 1, 2],
            "x": [0.1, 0.2, 0.3, 0.0, 0.1, 0.2],
            "y": [0.0, 0.5, 0.6, 0.2, 0.2, 0.3],
        }
    )
    labels = pd.DataFrame(
        {
            "dataset": ["opto_EB", "opto_EB"],
            "fly": ["a", "b"],
            "fly_number": [1, 2],
            "trial_type": ["testing", "testing"],
            "trial_label": ["t1", "t2"],
            LABEL_COLUMN: [0, 5],
        }
    )
    frames_path = tmp_path / "frames.csv"
    labels_path = tmp_path / "labels.csv"
    frames.to_csv(frames_path, index=False)
    labels.to_csv(labels_path, index=False)
    return frames_path, labels_path


def test_load_geom_frames_streams_and_joins(geometry_csvs: tuple[Path, Path]) -> None:
    frames_path, labels_path = geometry_csvs
    chunks = list(
        load_geom_frames(
            frames_path,
            chunk_size=2,
            columns=[
                "dataset",
                "fly",
                "fly_number",
                "trial_type",
                "trial_label",
                "frame_idx",
                "x",
                "y",
            ],
            labels_csv=labels_path,
        )
    )
    assert len(chunks) == 3
    first = chunks[0]
    assert first.columns.tolist() == [
        "dataset",
        "fly",
        "fly_number",
        "trial_type",
        "trial_label",
        "frame_idx",
        "x",
        "y",
        LABEL_COLUMN,
    ]
    assert first["dataset"].tolist() == ["opto_EB", "opto_EB"]
    assert first["fly"].tolist() == ["a", "a"]
    assert first["trial_type"].tolist() == ["testing", "testing"]
    assert first[LABEL_COLUMN].tolist() == [0, 0]
    assert chunks[1]["frame_idx"].tolist() == [2, 0]


def test_load_geom_frames_drops_unlabeled_rows(tmp_path: Path) -> None:
    frames = pd.DataFrame(
        {
            "dataset": ["opto_EB"] * 6,
            "fly": ["a"] * 3 + ["c"] * 3,
            "fly_number": [1] * 3 + [3] * 3,
            "trial_type": ["testing"] * 6,
            "trial_label": ["t1"] * 3 + ["t3"] * 3,
            "frame_idx": [0, 1, 2, 0, 1, 2],
            "x": [0.1, 0.2, 0.3, 0.0, 0.1, 0.2],
            "y": [0.0, 0.5, 0.6, 0.1, 0.2, 0.3],
        }
    )
    labels = pd.DataFrame(
        {
            "dataset": ["opto_EB"],
            "fly": ["a"],
            "fly_number": [1],
            "trial_type": ["testing"],
            "trial_label": ["t1"],
            LABEL_COLUMN: [5],
        }
    )
    frames_path = tmp_path / "frames.csv"
    labels_path = tmp_path / "labels.csv"
    frames.to_csv(frames_path, index=False)
    labels.to_csv(labels_path, index=False)

    chunks = list(
        load_geom_frames(
            frames_path,
            chunk_size=4,
            columns=[
                "dataset",
                "fly",
                "fly_number",
                "trial_type",
                "trial_label",
                "frame_idx",
                "x",
                "y",
            ],
            labels_csv=labels_path,
        )
    )
    combined = pd.concat(chunks, ignore_index=True)
    assert combined[["dataset", "fly", "trial_label"]].drop_duplicates().to_dict(
        orient="records"
    ) == [{"dataset": "opto_EB", "fly": "a", "trial_label": "t1"}]
    assert combined[LABEL_COLUMN].eq(5).all()


def test_load_geom_frames_strict_missing_labels_raises(tmp_path: Path) -> None:
    frames = pd.DataFrame(
        {
            "dataset": ["opto_EB", "opto_EB"],
            "fly": ["a", "c"],
            "fly_number": [1, 3],
            "trial_type": ["testing", "testing"],
            "trial_label": ["t1", "t3"],
            "frame_idx": [0, 0],
            "x": [0.1, 0.2],
            "y": [0.0, 0.1],
        }
    )
    labels = pd.DataFrame(
        {
            "dataset": ["opto_EB"],
            "fly": ["a"],
            "fly_number": [1],
            "trial_type": ["testing"],
            "trial_label": ["t1"],
            LABEL_COLUMN: [0],
        }
    )
    frames_path = tmp_path / "frames.csv"
    labels_path = tmp_path / "labels.csv"
    frames.to_csv(frames_path, index=False)
    labels.to_csv(labels_path, index=False)

    with pytest.raises(DataValidationError):
        list(
            load_geom_frames(
                frames_path,
                chunk_size=1,
                labels_csv=labels_path,
                drop_missing_labels=False,
            )
        )


def test_load_geom_frames_handles_missing_default_frame_column(tmp_path: Path) -> None:
    frames = pd.DataFrame(
        {
            "dataset": ["opto_EB", "opto_EB", "opto_EB", "opto_EB"],
            "fly": ["f1", "f1", "f2", "f2"],
            "fly_number": [1, 1, 2, 2],
            "trial_type": ["testing", "testing", "testing", "testing"],
            "trial_label": ["t1", "t1", "t2", "t2"],
            "frame": [0, 1, 0, 1],
            "x": [0.1, 0.2, 0.3, 0.4],
        }
    )
    frames_path = tmp_path / "frames.csv"
    frames.to_csv(frames_path, index=False)

    chunks = list(
        load_geom_frames(
            frames_path,
            chunk_size=2,
            columns=[
                "dataset",
                "fly",
                "fly_number",
                "trial_type",
                "trial_label",
                "frame",
                "x",
            ],
        )
    )

    assert len(chunks) == 2
    combined = pd.concat(chunks, ignore_index=True)
    assert "frame" in combined.columns
    assert "frame_idx" not in combined.columns
    assert combined["frame"].tolist() == [0, 1, 0, 1]


def test_load_geom_frames_cache_roundtrip(geometry_csvs: tuple[Path, Path], tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="Parquet cache tests require pyarrow")
    frames_path, labels_path = geometry_csvs
    cache_path = tmp_path / "cache.parquet"

    streamed = list(
        load_geom_frames(
            frames_path,
            chunk_size=3,
            cache_parquet=cache_path,
            labels_csv=labels_path,
        )
    )
    assert cache_path.exists()
    cached = list(
        load_geom_frames(
            frames_path,
            chunk_size=3,
            cache_parquet=cache_path,
            use_cache=True,
            labels_csv=labels_path,
        )
    )
    assert [chunk.equals(streamed[idx]) for idx, chunk in enumerate(cached)]


def test_load_geom_frames_duplicate_labels(geometry_csvs: tuple[Path, Path], tmp_path: Path) -> None:
    frames_path, labels_path = geometry_csvs
    dup_labels = pd.read_csv(labels_path)
    dup_labels = pd.concat([dup_labels, dup_labels.iloc[[0]]], ignore_index=True)
    dup_path = tmp_path / "dup.csv"
    dup_labels.to_csv(dup_path, index=False)
    with pytest.raises(DataValidationError):
        list(
            load_geom_frames(
                frames_path,
                chunk_size=2,
                labels_csv=dup_path,
            )
        )


def test_aggregate_trials_matches_chunked_means(geometry_csvs: tuple[Path, Path]) -> None:
    frames_path, labels_path = geometry_csvs
    stream = load_geom_frames(
        frames_path,
        chunk_size=2,
        columns=[
            "dataset",
            "fly",
            "fly_number",
            "trial_type",
            "trial_label",
            "frame_idx",
            "x",
            "y",
        ],
        labels_csv=labels_path,
    )
    aggregated = aggregate_trials(stream, stats=["mean", "max"])
    assert set(aggregated["trial_label"]) == {"t1", "t2"}
    assert set(aggregated["dataset"]) == {"opto_EB"}
    assert set(aggregated["trial_type"]) == {"testing"}
    t1_row = aggregated.loc[aggregated["trial_label"] == "t1"].iloc[0]
    assert pytest.approx(t1_row["x_mean"], rel=1e-6) == 0.2
    assert pytest.approx(t1_row["y_max"], rel=1e-6) == 0.6


def test_load_geometry_dataset_trial_granularity(geometry_csvs: tuple[Path, Path]) -> None:
    frames_path, labels_path = geometry_csvs
    dataset = load_geometry_dataset(
        frames_path,
        labels_csv=labels_path,
        granularity="trial",
        stats=("mean", "max"),
        normalization="zscore",
    )
    assert dataset.granularity == "trial"
    assert dataset.frame.shape[0] == 2
    assert LABEL_COLUMN in dataset.frame.columns
    assert LABEL_INTENSITY_COLUMN in dataset.frame.columns
    assert dataset.frame[LABEL_COLUMN].tolist() == [0, 1]
    assert "x_mean" in dataset.frame.columns
    assert dataset.frame["x_mean"].dtype == "float32"
    assert dataset.feature_columns and "x_mean" in dataset.feature_columns


def test_load_geometry_dataset_builds_traces(tmp_path: Path) -> None:
    frames = pd.DataFrame(
        {
            "dataset": ["d", "d", "d", "d"],
            "fly": ["f", "f", "f", "f"],
            "fly_number": [1, 1, 1, 1],
            "trial_type": ["testing", "testing", "testing", "testing"],
            "trial_label": ["t1", "t1", "t1", "t1"],
            "frame": [0, 1, 2, 3],
            "eye_x": [0.1, 0.2, 0.3, 0.4],
            "eye_y": [0.0, 0.1, 0.2, 0.3],
            "prob_x": [0.5, 0.4, 0.3, 0.2],
            "prob_y": [0.6, 0.5, 0.4, 0.3],
        }
    )
    labels = pd.DataFrame(
        {
            "dataset": ["d"],
            "fly": ["f"],
            "fly_number": [1],
            "trial_type": ["testing"],
            "trial_label": ["t1"],
            LABEL_COLUMN: [5],
        }
    )
    frames_path = tmp_path / "geom.csv"
    labels_path = tmp_path / "labels.csv"
    frames.to_csv(frames_path, index=False)
    labels.to_csv(labels_path, index=False)

    dataset = load_geometry_dataset(
        frames_path,
        labels_csv=labels_path,
        granularity="trial",
        normalization="none",
    )

    assert dataset.trace_prefixes == RAW_TRACE_PREFIXES
    expected_columns = [
        f"{prefix}{idx}"
        for prefix in RAW_TRACE_PREFIXES
        for idx in range(4)
    ]
    assert dataset.trace_columns == expected_columns
    assert dataset.frame.loc[0, "eye_x_f0"] == pytest.approx(0.1)
    assert dataset.frame.loc[0, "eye_x_f3"] == pytest.approx(0.4)
    assert dataset.frame.loc[0, "prob_y_f3"] == pytest.approx(0.3)
    assert all(dataset.frame[col].dtype == "float32" for col in expected_columns)


def test_load_geometry_dataset_frame_granularity(geometry_csvs: tuple[Path, Path]) -> None:
    frames_path, labels_path = geometry_csvs
    dataset = load_geometry_dataset(
        frames_path,
        labels_csv=labels_path,
        granularity="frame",
        normalization="minmax",
    )
    assert dataset.granularity == "frame"
    assert len(dataset.frame) == 6
    assert dataset.frame[LABEL_COLUMN].isin({0, 1}).all()
    assert dataset.feature_columns
    first_feature = dataset.feature_columns[0]
    assert dataset.frame[first_feature].dtype == "float32"


def test_load_geom_frames_requires_full_key_set(tmp_path: Path) -> None:
    frames = pd.DataFrame(
        {
            "fly": ["a", "a"],
            "fly_number": [1, 1],
            "trial_type": ["testing", "testing"],
            "trial_label": ["t1", "t1"],
            "frame_idx": [0, 1],
        }
    )
    labels = pd.DataFrame(
        {
            "fly": ["a"],
            "fly_number": [1],
            "trial_type": ["testing"],
            "trial_label": ["t1"],
            LABEL_COLUMN: [0],
        }
    )
    frames_path = tmp_path / "missing_keys.csv"
    labels_path = tmp_path / "missing_keys_labels.csv"
    frames.to_csv(frames_path, index=False)
    labels.to_csv(labels_path, index=False)

    with pytest.raises(DataValidationError):
        list(
            load_geom_frames(
                frames_path,
                chunk_size=1,
                labels_csv=labels_path,
            )
        )

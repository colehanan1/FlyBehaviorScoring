"""Utilities for converting raw coordinate CSVs into modeling-ready tables."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .io import RAW_TRACE_PREFIXES
from .io_wide import find_series_columns
from .logging_utils import get_logger

TRIAL_ID_COLUMN = "trial_label"

REQUIRED_METADATA_COLUMNS = [
    "dataset",
    "fly",
    "fly_number",
    "trial_type",
    TRIAL_ID_COLUMN,
]
DEFAULT_OUTPUT_PATH = Path(
    "/home/ramanlab/Documents/cole/Data/Opto/Combined/all_eye_prob_coords_prepared.csv"
)
DEFAULT_PREFIXES = list(RAW_TRACE_PREFIXES)
LABEL_COLUMN_CANDIDATES = [
    "user_score_odor",
    "label",
    "response_label",
    "response",
    "score",
    "trial_response",
]


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(
            "Missing required metadata columns: {missing}. Ensure the input CSV is per-trial with testing metadata.".format(
                missing=missing
            )
        )


def _normalise_trial_identifier(df: pd.DataFrame, logger: Any, source: str) -> pd.DataFrame:
    """Ensure ``TRIAL_ID_COLUMN`` exists, renaming legacy columns when necessary."""

    if TRIAL_ID_COLUMN in df.columns:
        return df
    if "testing_trial" in df.columns:
        logger.warning(
            "Renaming legacy column 'testing_trial' from %s to '%s'.",
            source,
            TRIAL_ID_COLUMN,
        )
        return df.rename(columns={"testing_trial": TRIAL_ID_COLUMN})
    df = df.copy()
    df[TRIAL_ID_COLUMN] = pd.NA
    logger.warning(
        "%s is missing '%s'; downstream alignment will rely on row order within groups.",
        source,
        TRIAL_ID_COLUMN,
    )
    return df


def enumerate_time_columns(df: pd.DataFrame, prefixes: Sequence[str]) -> tuple[List[str], int]:
    """Return flattened time columns and frame count for the given prefixes."""

    mapping = find_series_columns(df, prefixes)
    frame_count = len(next(iter(mapping.values())))
    ordered: List[str] = []
    for prefix in prefixes:
        ordered.extend(mapping[prefix])
    return ordered, frame_count


def slice_time_columns(row: pd.Series, prefixes: Sequence[str], start: int, end: int) -> dict[str, Any]:
    """Extract a slice of time-series values from ``row``.

    Columns are renumbered to begin at zero after slicing.
    """

    result: "OrderedDict[str, Any]" = OrderedDict()
    length = end - start
    for offset in range(length):
        frame_idx = start + offset
        for prefix in prefixes:
            result[f"{prefix}{offset}"] = row[f"{prefix}{frame_idx}"]
    return result


def _group_align_labels(
    data_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    by_keys: Sequence[str],
    logger_name: str,
    label_column: str,
) -> pd.Series:
    """Align labels to trials when the per-trial identifier is missing.

    This alignment respects the original row order of the labels CSV and
    prioritises the per-trial identifier ordering for the data when available.
    """

    logger = get_logger(logger_name)
    labels_df = labels_df.copy()
    labels_df["_original_order"] = np.arange(len(labels_df), dtype=int)

    aligned_labels = []
    warnings: List[str] = []
    label_groups = labels_df.groupby(list(by_keys), dropna=False, sort=False)
    for key, data_idx in data_df.groupby(list(by_keys), dropna=False, sort=False).groups.items():
        if key not in label_groups.groups:
            raise ValueError(
                "Labels missing for group {key}. Ensure both CSVs cover identical per-trial entries.".format(
                    key=key
                )
            )
        data_subset = data_df.loc[data_idx]
        label_subset = labels_df.loc[label_groups.groups[key]]
        if len(data_subset) != len(label_subset):
            raise ValueError(
                "Label count mismatch for group {key}: data has {data_count}, labels have {label_count}.".format(
                    key=key,
                    data_count=len(data_subset),
                    label_count=len(label_subset),
                )
            )
        if TRIAL_ID_COLUMN in data_subset.columns:
            data_subset = data_subset.sort_values(TRIAL_ID_COLUMN, na_position="last")
        label_subset = label_subset.sort_values("_original_order")
        warnings.append(str(key))
        aligned = pd.Series(label_subset[label_column].to_numpy(), index=data_subset.index)
        aligned_labels.append(aligned)
    if warnings:
        logger.warning(
            "Aligned labels by row-order for groups lacking '%s': %s",
            TRIAL_ID_COLUMN,
            ", ".join(warnings),
        )
    if aligned_labels:
        return pd.concat(aligned_labels).reindex(data_df.index)
    return pd.Series([], dtype=float)


def _load_matrix_trials(
    *,
    data_path: Path,
    meta_path: Path,
    prefixes: Sequence[str],
    logger: Any,
) -> tuple[pd.DataFrame, List[str]]:
    """Load per-trial metadata and traces from a matrix + JSON description."""

    meta = json.loads(meta_path.read_text())
    metadata_entries = meta.get("metadata") or meta.get("trials")
    columns_spec = (
        meta.get("columns")
        or meta.get("column_names")
        or meta.get("cols")
        or meta.get("header")
    )

    try:
        matrix = np.load(data_path)
    except ValueError as exc:
        message = str(exc)
        if "allow_pickle=False" in message:
            logger.warning(
                "Reloading %s with allow_pickle=True because the stored array uses object dtype.",
                data_path,
            )
            matrix = np.load(data_path, allow_pickle=True)
        else:
            raise

    if matrix.ndim == 2:
        if columns_spec is None:
            raise ValueError(
                "2D matrix inputs require a 'columns' (or 'column_names') array in the metadata JSON."
            )
        columns = [str(col) for col in columns_spec]
        if len(columns) != matrix.shape[1]:
            raise ValueError(
                "Column count %d in JSON does not match matrix width %d."
                % (len(columns), matrix.shape[1])
            )
        frame_df = pd.DataFrame(matrix, columns=columns)
        frame_df = _normalise_trial_identifier(frame_df, logger, "matrix table")
        if metadata_entries is not None:
            metadata_df = pd.DataFrame(metadata_entries)
            if len(metadata_df) != len(frame_df):
                raise ValueError(
                    "Metadata rows (%d) do not match matrix trials (%d)."
                    % (len(metadata_df), len(frame_df))
                )
            metadata_df = _normalise_trial_identifier(metadata_df, logger, "matrix metadata")
            for column in metadata_df.columns:
                frame_df[column] = metadata_df[column].to_numpy()
        for prefix in prefixes:
            matching = [col for col in frame_df.columns if col.startswith(prefix)]
            for column in matching:
                frame_df[column] = pd.to_numeric(frame_df[column], errors="raise")
        return frame_df, list(prefixes)

    if matrix.dtype == object and matrix.ndim != 2:
        logger.warning(
            "Coercing object-dtype matrix from %s to a numeric array; ensure ragged trials are not present.",
            data_path,
        )
        try:
            matrix = np.array(matrix.tolist(), dtype=float)
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                "Loaded matrix contains non-uniform object entries; ensure every trial shares the same frame/channel shape."
            ) from exc
    if matrix.ndim != 3:
        raise ValueError(
            "Expected data_npy to contain a 3D array (trials x frames x channels or trials x channels x frames)."
        )
    if metadata_entries is None:
        raise ValueError(
            "Matrix metadata JSON must include a 'metadata' or 'trials' array describing each trial."
        )
    metadata_df = pd.DataFrame(metadata_entries)
    if metadata_df.empty:
        raise ValueError("Metadata JSON contained no trial descriptions.")

    metadata_df = _normalise_trial_identifier(metadata_df, logger, "matrix metadata")

    expected_trials = matrix.shape[0]
    if len(metadata_df) != expected_trials:
        raise ValueError(
            "Matrix trials (%d) do not match metadata entries (%d)." % (expected_trials, len(metadata_df))
        )

    layout = meta.get("layout") or meta.get("matrix_layout") or meta.get("order")
    if layout is None:
        layout = "trial_time_channel"
    layout = str(layout).lower()
    if layout in {"trial_channel_time", "nct", "channels_last_false"}:
        matrix = np.transpose(matrix, (0, 2, 1))
    elif layout in {"trial_time_channel", "ntc", "channels_last_true"}:
        pass
    else:
        raise ValueError(
            "Unsupported matrix layout '%s'. Expected 'trial_time_channel' or 'trial_channel_time'." % layout
        )

    n_trials, frame_count, channel_count = matrix.shape
    json_prefixes = meta.get("channel_prefixes") or meta.get("prefixes") or meta.get("channels")
    if json_prefixes is not None:
        json_prefixes = [str(item) for item in json_prefixes]
        if len(json_prefixes) != len(prefixes):
            raise ValueError(
                "channel_prefixes length %d from JSON does not match expected prefixes %d." % (
                    len(json_prefixes),
                    len(prefixes),
                )
            )
        if list(prefixes) != json_prefixes:
            raise ValueError(
                "Provided series_prefixes %s do not match channel_prefixes %s from matrix metadata." % (
                    list(prefixes),
                    json_prefixes,
                )
            )

    if channel_count != len(prefixes):
        raise ValueError(
            "Matrix channel dimension (%d) does not match number of prefixes (%d)." % (
                channel_count,
                len(prefixes),
            )
        )

    flattened: "OrderedDict[str, Any]" = OrderedDict()
    for channel_idx, prefix in enumerate(prefixes):
        for frame in range(frame_count):
            flattened[f"{prefix}{frame}"] = matrix[:, frame, channel_idx]

    traces_df = pd.DataFrame(flattened)
    combined_df = pd.concat([metadata_df.reset_index(drop=True), traces_df], axis=1)
    logger.debug(
        "Loaded matrix with %d trials, %d frames, %d channels into wide table.",
        n_trials,
        frame_count,
        channel_count,
    )
    return combined_df, list(prefixes)


def prepare_raw(
    *,
    data_csv: Path | None = None,
    data_npy: Path | None = None,
    matrix_meta: Path | None = None,
    labels_csv: Path,
    out_path: Path = DEFAULT_OUTPUT_PATH,
    fps: int = 40,
    odor_on_idx: int = 1230,
    odor_off_idx: int = 2430,
    truncate_before: int = 0,
    truncate_after: int = 0,
    series_prefixes: Sequence[str] | None = None,
    compute_dir_val: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Prepare raw coordinate trials for downstream modeling."""

    prefixes = list(series_prefixes or DEFAULT_PREFIXES)
    if len(prefixes) != 4:
        raise ValueError(
            "Expected exactly four prefixes for eye/proboscis coordinates; received {count}: {prefixes}".format(
                count=len(prefixes), prefixes=prefixes
            )
        )
    logger = get_logger("prepare_raw", verbose=verbose)
    if data_npy is not None:
        if data_csv is not None:
            raise ValueError("Provide either data_csv or data_npy, not both.")
        if matrix_meta is None:
            raise ValueError("matrix_meta JSON is required when supplying data_npy.")
        logger.info("Reading data matrix %s", data_npy)
        data_df, prefixes = _load_matrix_trials(
            data_path=data_npy,
            meta_path=matrix_meta,
            prefixes=prefixes,
            logger=logger,
        )
    else:
        if data_csv is None:
            raise ValueError("Either data_csv or data_npy must be provided to prepare_raw.")
        logger.info("Reading data CSV %s", data_csv)
        data_df = pd.read_csv(data_csv)
    data_df = _normalise_trial_identifier(data_df, logger, "data table")
    logger.debug("Data shape: %s", data_df.shape)
    _ensure_columns(data_df, REQUIRED_METADATA_COLUMNS[:-1])
    _ensure_columns(data_df, REQUIRED_METADATA_COLUMNS)

    group_counts = (
        data_df.groupby(["dataset", "fly", "fly_number", "trial_type"], dropna=False).size()
    )
    if group_counts.max() <= 1:
        raise ValueError(
            "Detected only one row per fly group. Provide a per-trial CSV (one row per testing trial) before running prepare-raw."
        )

    if not (data_df["trial_type"].astype(str).str.lower() == "testing").all():
        raise ValueError("All rows must have trial_type='testing' for prepare-raw input.")

    data_df = data_df.sort_values(REQUIRED_METADATA_COLUMNS).reset_index(drop=True)
    logger.debug("Sorted data for deterministic ordering.")

    _, frame_count = enumerate_time_columns(data_df, prefixes)
    logger.debug("Detected %d frame columns per prefix", frame_count)

    if truncate_before == 0 and truncate_after == 0:
        start_frame = 0
        end_frame = frame_count
    else:
        max_index = max(frame_count - 1, 0)
        odor_on_clamped = min(max(odor_on_idx, 0), max_index)
        odor_off_clamped = min(max(odor_off_idx, 0), max_index)
        start_frame = max(0, odor_on_clamped - truncate_before)
        end_frame = min(frame_count, odor_off_clamped + 1 + truncate_after)
    if start_frame >= end_frame:
        raise ValueError("Invalid truncation bounds; no frames remain after slicing.")
    if truncate_before or truncate_after:
        logger.info(
            "Truncating frames to [%d, %d) based on odor window %d-%d and truncate_before=%d truncate_after=%d",
            start_frame,
            end_frame,
            odor_on_idx,
            odor_off_idx,
            truncate_before,
            truncate_after,
        )
    sliced_frame_count = end_frame - start_frame

    truncated = OrderedDict()
    for prefix in prefixes:
        source_columns = [f"{prefix}{idx}" for idx in range(start_frame, end_frame)]
        for offset, column in enumerate(source_columns):
            new_column = f"{prefix}{offset}"
            truncated.setdefault(new_column, data_df[column].to_numpy())
    time_df = pd.DataFrame(truncated, index=data_df.index)

    labels_df = pd.read_csv(labels_csv)
    labels_df = _normalise_trial_identifier(labels_df, logger, "labels table")
    logger.info("Reading labels CSV %s", labels_csv)
    logger.debug("Labels shape: %s", labels_df.shape)

    label_candidates = [col for col in LABEL_COLUMN_CANDIDATES if col in labels_df.columns]
    if label_candidates:
        label_column = label_candidates[0]
    elif TRIAL_ID_COLUMN in labels_df.columns:
        label_column = TRIAL_ID_COLUMN
        logger.warning(
            "No explicit label column found; using '%s' from labels CSV as the response column.",
            TRIAL_ID_COLUMN,
        )
    else:
        raise ValueError(
            "Labels CSV must include one of %s (or a numeric '%s') to provide response values."
            % (LABEL_COLUMN_CANDIDATES, TRIAL_ID_COLUMN)
        )

    try:
        labels_df[label_column] = pd.to_numeric(labels_df[label_column], errors="raise")
    except Exception as exc:  # pragma: no cover - pandas specific
        raise ValueError(
            "Label column '%s' must contain numeric values for classification." % label_column
        ) from exc

    if labels_df[TRIAL_ID_COLUMN].notna().all():
        join_keys = REQUIRED_METADATA_COLUMNS
        logger.debug("Joining labels on keys: %s", join_keys)
        merged_labels = pd.merge(
            data_df[join_keys],
            labels_df[join_keys + [label_column]],
            on=join_keys,
            how="left",
            validate="one_to_one",
        )
        if merged_labels[label_column].isna().any():
            missing = data_df.loc[merged_labels[label_column].isna(), join_keys]
            raise ValueError(
                "Missing labels for %d trials after key join. Offending rows: %s" % (
                    len(missing),
                    missing.to_dict(orient="records"),
                )
            )
        label_series = merged_labels[label_column].astype(float)
    else:
        logger.debug("Falling back to row-order alignment within groups: %s", REQUIRED_METADATA_COLUMNS[:-1])
        label_series = _group_align_labels(
            data_df,
            labels_df,
            by_keys=REQUIRED_METADATA_COLUMNS[:-1],
            logger_name="prepare_raw",
            label_column=label_column,
        )
        if label_series.isna().any():
            raise ValueError("Row-order alignment produced NaN labels; check CSV consistency.")

    metadata = data_df[["dataset", "fly", "fly_number", "trial_type", TRIAL_ID_COLUMN]].copy()
    metadata["fps"] = int(fps)
    metadata["odor_on_idx"] = int(odor_on_idx)
    metadata["odor_off_idx"] = int(odor_off_idx)
    metadata["total_frames"] = int(sliced_frame_count)
    metadata["label"] = label_series.astype(float)

    prepared = pd.concat([metadata, time_df], axis=1)

    if compute_dir_val:
        logger.info("Computing dir_val derived channel from eye/proboscis deltas.")
        for frame in range(sliced_frame_count):
            dx = prepared[f"prob_x_f{frame}"] - prepared[f"eye_x_f{frame}"]
            dy = prepared[f"prob_y_f{frame}"] - prepared[f"eye_y_f{frame}"]
            prepared[f"dir_val_f{frame}"] = np.hypot(dx, dy)

    ordered_columns: List[str] = [
        "dataset",
        "fly",
        "fly_number",
        "trial_type",
        TRIAL_ID_COLUMN,
        "fps",
        "odor_on_idx",
        "odor_off_idx",
        "total_frames",
        "label",
    ]
    for frame in range(sliced_frame_count):
        for prefix in prefixes:
            ordered_columns.append(f"{prefix}{frame}")
    if compute_dir_val:
        for frame in range(sliced_frame_count):
            ordered_columns.append(f"dir_val_f{frame}")
    prepared = prepared.loc[:, ordered_columns]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    logger.info("Wrote prepared dataset with %d trials and %d frames to %s", len(prepared), sliced_frame_count, out_path)

    return prepared


if __name__ == "__main__":  # pragma: no cover - self-check harness
    rng = np.random.default_rng(0)
    base_data = pd.DataFrame(
        {
            "dataset": ["d"] * 2,
            "fly": ["f"] * 2,
            "fly_number": [1, 1],
            "trial_type": ["testing", "testing"],
            TRIAL_ID_COLUMN: ["trial_1", "trial_2"],
            **{f"eye_x_f{i}": rng.normal(size=2) for i in range(3)},
            **{f"eye_y_f{i}": rng.normal(size=2) for i in range(3)},
            **{f"prob_x_f{i}": rng.normal(size=2) for i in range(3)},
            **{f"prob_y_f{i}": rng.normal(size=2) for i in range(3)},
        }
    )
    keyed_labels = pd.DataFrame(
        {
            "dataset": ["d", "d"],
            "fly": ["f", "f"],
            "fly_number": [1, 1],
            "trial_type": ["testing", "testing"],
            TRIAL_ID_COLUMN: ["trial_1", "trial_2"],
            "user_score_odor": [0, 1],
        }
    )
    fallback_labels = keyed_labels.drop(columns=[TRIAL_ID_COLUMN])
    fallback_labels.loc[:, "user_score_odor"] = [1, 0]

    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "data.csv"
        label_path = tmp_path / "labels.csv"
        base_data.to_csv(data_path, index=False)
        keyed_labels.to_csv(label_path, index=False)

        out_path = tmp_path / "prepared.csv"
        prepared_df = prepare_raw(
            data_csv=data_path,
            labels_csv=label_path,
            out_path=out_path,
            verbose=True,
        )
        assert list(prepared_df.columns[:10]) == [
            "dataset",
            "fly",
            "fly_number",
            "trial_type",
            TRIAL_ID_COLUMN,
            "fps",
            "odor_on_idx",
            "odor_off_idx",
            "total_frames",
            "label",
        ]
        assert prepared_df.shape[1] == 10 + 4 * 3
        assert prepared_df["total_frames"].iloc[0] == 3

        truncated_path = tmp_path / "prepared_trunc.csv"
        prepare_raw(
            data_csv=data_path,
            labels_csv=label_path,
            out_path=truncated_path,
            truncate_before=1,
            truncate_after=1,
            odor_on_idx=1,
            odor_off_idx=1,
        )
        truncated_df = pd.read_csv(truncated_path)
        assert truncated_df.shape[1] == 10 + 4 * 2

        fallback_path = tmp_path / "prepared_fallback.csv"
        fallback_labels_path = tmp_path / "labels_fallback.csv"
        fallback_labels.to_csv(fallback_labels_path, index=False)
        prepare_raw(
            data_csv=data_path,
            labels_csv=fallback_labels_path,
            out_path=fallback_path,
        )
        fallback_df = pd.read_csv(fallback_path)
        assert fallback_df["label"].tolist() == [1.0, 0.0]

        dirval_path = tmp_path / "prepared_dirval.csv"
        prepare_raw(
            data_csv=data_path,
            labels_csv=label_path,
            out_path=dirval_path,
            compute_dir_val=True,
        )
        dirval_df = pd.read_csv(dirval_path)
        assert any(col.startswith("dir_val_f") for col in dirval_df.columns)

        matrix = rng.normal(size=(2, 3, 4))
        matrix_path = tmp_path / "matrix.npy"
        np.save(matrix_path, matrix)
        meta_payload = {
            "metadata": [
                {
                    "dataset": "d",
                    "fly": "f",
                    "fly_number": 1,
                    "trial_type": "testing",
                    TRIAL_ID_COLUMN: f"trial_{idx + 1}",
                }
                for idx in range(matrix.shape[0])
            ],
            "layout": "trial_time_channel",
            "channel_prefixes": DEFAULT_PREFIXES,
        }
        meta_path = tmp_path / "matrix.json"
        meta_path.write_text(json.dumps(meta_payload))
        prepare_raw(
            data_npy=matrix_path,
            matrix_meta=meta_path,
            labels_csv=label_path,
            out_path=tmp_path / "prepared_matrix.csv",
        )

    print("prepare_raw self-checks completed successfully.")

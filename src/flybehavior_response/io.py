"""I/O utilities for flybehavior_response."""

from __future__ import annotations

import re
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .logging_utils import get_logger

MERGE_KEYS = ["fly", "fly_number", "trial_label"]
OPTIONAL_KEYS = ["dataset", "trial_type"]
LABEL_COLUMN = "user_score_odor"
LABEL_INTENSITY_COLUMN = "user_score_odor_intensity"
TRACE_PATTERN = re.compile(r"^dir_val_(\d+)$")
TRACE_RANGE = (0, 3600)
FEATURE_COLUMNS = {
    "AUC-Before",
    "AUC-During",
    "AUC-After",
    "AUC-During-Before-Ratio",
    "AUC-After-Before-Ratio",
    "TimeToPeak-During",
    "Peak-Value",
}


class DataValidationError(RuntimeError):
    """Raised when data schema validation fails."""


@dataclass(slots=True)
class MergedDataset:
    """Container for merged dataset and metadata."""

    frame: pd.DataFrame
    trace_columns: List[str]
    feature_columns: List[str]
    label_intensity: pd.Series
    sample_weights: pd.Series


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas specific
        raise DataValidationError(f"Failed to read CSV {path}: {exc}") from exc


def _validate_keys(frame: pd.DataFrame, path: Path) -> None:
    missing = [col for col in MERGE_KEYS if col not in frame.columns]
    if missing:
        raise DataValidationError(
            f"File {path} is missing required key columns: {missing}. "
            "Ensure the CSV includes fly identifiers and trial labels."
        )
    dup_mask = frame.duplicated(subset=MERGE_KEYS, keep=False)
    if dup_mask.any():
        dup_rows = frame.loc[dup_mask, MERGE_KEYS].drop_duplicates()
        raise DataValidationError(
            "Duplicate keys detected in file "
            f"{path}. Resolve duplicates for keys: {dup_rows.to_dict(orient='records')}"
        )


def _extract_trace_columns(columns: Iterable[str]) -> List[str]:
    trace_cols: List[str] = []
    for col in columns:
        match = TRACE_PATTERN.match(col)
        if not match:
            continue
        idx = int(match.group(1))
        if TRACE_RANGE[0] <= idx <= TRACE_RANGE[1]:
            trace_cols.append(col)
    return sorted(trace_cols, key=lambda x: int(x.split("_")[-1]))


def validate_feature_columns(frame: pd.DataFrame) -> List[str]:
    available = [col for col in frame.columns if col in FEATURE_COLUMNS]
    if not available:
        raise DataValidationError(
            "No engineered feature columns detected. Expected columns include: "
            f"{sorted(FEATURE_COLUMNS)}"
        )
    return available


def _coerce_labels(labels: pd.Series, labels_csv: Path) -> pd.Series:
    try:
        numeric = pd.to_numeric(labels, errors="raise")
    except Exception as exc:  # pragma: no cover - pandas specific
        raise DataValidationError(
            f"Label column '{LABEL_COLUMN}' in {labels_csv} must be numeric with 0 indicating no response and positive integers for responses."
        ) from exc
    if (numeric < 0).any():
        raise DataValidationError(
            f"Negative label values detected in {labels_csv}. Expected 0 for no response and positive integers for response strength."
        )
    if not (numeric.dropna() == numeric.dropna().astype(int)).all():
        raise DataValidationError(
            f"Non-integer label values detected in {labels_csv}. Use integers 0-5 to encode response strength."
        )
    return numeric.astype(int)


def _compute_sample_weights(intensity: pd.Series) -> pd.Series:
    weights = pd.Series(1.0, index=intensity.index, dtype=float)
    positive_mask = intensity > 0
    if positive_mask.any():
        weights.loc[positive_mask] = intensity.loc[positive_mask].astype(float)
    return weights


def load_and_merge(data_csv: Path, labels_csv: Path, logger_name: str = __name__) -> MergedDataset:
    """Load and merge data and labels CSVs."""
    logger = get_logger(logger_name)
    logger.info("Loading data CSV: %s", data_csv)
    data_df = _load_csv(data_csv)
    logger.debug("Data shape: %s", data_df.shape)

    logger.info("Loading labels CSV: %s", labels_csv)
    labels_df = _load_csv(labels_csv)
    logger.debug("Labels shape: %s", labels_df.shape)

    logger.debug(
        "Key column dtypes | data: %s | labels: %s",
        {col: dtype.name for col, dtype in data_df[MERGE_KEYS].dtypes.items()},
        {col: dtype.name for col, dtype in labels_df[MERGE_KEYS].dtypes.items()},
    )

    _validate_keys(data_df, data_csv)
    _validate_keys(labels_df, labels_csv)

    if LABEL_COLUMN not in labels_df.columns:
        raise DataValidationError(
            f"Labels file {labels_csv} missing required label column '{LABEL_COLUMN}'."
        )

    label_values = labels_df[LABEL_COLUMN]
    if label_values.isna().any():
        na_count = int(label_values.isna().sum())
        labels_df = labels_df.loc[~label_values.isna()].copy()
        logger.warning(
            "Dropped %d rows with NaN labels in %s.", na_count, labels_csv
        )

    coerced_labels = _coerce_labels(labels_df[LABEL_COLUMN], labels_csv)
    labels_df[LABEL_COLUMN] = coerced_labels

    trace_cols = _extract_trace_columns(data_df.columns)
    if not trace_cols:
        raise DataValidationError(
            "No trace columns found. Expected columns matching 'dir_val_0'..'dir_val_3600'."
        )

    to_drop = [col for col in data_df.columns if TRACE_PATTERN.match(col) and col not in trace_cols]
    if to_drop:
        data_df = data_df.drop(columns=to_drop)
        logger.info("Dropped %d trace columns outside %s", len(to_drop), TRACE_RANGE)

    feature_cols = validate_feature_columns(data_df)
    merged = pd.merge(
        data_df,
        labels_df[[*MERGE_KEYS, LABEL_COLUMN]],
        on=MERGE_KEYS,
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        _diagnose_merge_failure(data_df, labels_df, logger)
        raise DataValidationError(
            "Merge produced no rows. Verify matching keys across CSVs and column types."
        )

    merged.sort_values(MERGE_KEYS, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    if merged[LABEL_COLUMN].isna().any():
        raise DataValidationError("Merged data contains NaN labels after merge.")

    intensity = merged[LABEL_COLUMN].astype(int)
    weights = _compute_sample_weights(intensity)
    merged[LABEL_INTENSITY_COLUMN] = intensity
    merged[LABEL_COLUMN] = (intensity > 0).astype(int)

    distribution = intensity.value_counts().sort_index().to_dict()
    logger.info("Label intensity distribution: %s", distribution)
    logger.info(
        "Sample weight summary | min=%.2f mean=%.2f max=%.2f",
        float(weights.min()),
        float(weights.mean()),
        float(weights.max()),
    )

    logger.info("Merged dataset shape: %s", merged.shape)

    return MergedDataset(
        frame=merged,
        trace_columns=trace_cols,
        feature_columns=feature_cols,
        label_intensity=intensity,
        sample_weights=weights,
    )


def write_parquet(dataset: MergedDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_parquet(path, index=False)


def _diagnose_merge_failure(
    data_df: pd.DataFrame, labels_df: pd.DataFrame, logger: Logger
) -> None:
    """Emit detailed diagnostics to aid debugging of merge mismatches."""

    data_keys = data_df[MERGE_KEYS].drop_duplicates()
    label_keys = labels_df[MERGE_KEYS].drop_duplicates()

    logger.error(
        "Merge diagnostics | data rows: %d (unique keys: %d) | labels rows: %d (unique keys: %d)",
        len(data_df),
        len(data_keys),
        len(labels_df),
        len(label_keys),
    )

    data_key_set = {
        tuple(row)
        for row in data_keys.itertuples(index=False, name=None)
    }
    label_key_set = {
        tuple(row)
        for row in label_keys.itertuples(index=False, name=None)
    }

    only_in_data = list(data_key_set - label_key_set)[:5]
    only_in_labels = list(label_key_set - data_key_set)[:5]

    if only_in_data:
        logger.error("Example keys present in data but missing in labels: %s", only_in_data)
    if only_in_labels:
        logger.error("Example keys present in labels but missing in data: %s", only_in_labels)

    for key in MERGE_KEYS:
        data_values = set(data_df[key].dropna().unique())
        label_values = set(labels_df[key].dropna().unique())
        missing_from_labels = list(data_values - label_values)[:5]
        missing_from_data = list(label_values - data_values)[:5]
        if missing_from_labels:
            logger.error(
                "Values for '%s' only in data: %s", key, missing_from_labels
            )
        if missing_from_data:
            logger.error(
                "Values for '%s' only in labels: %s", key, missing_from_data
            )

    for key in MERGE_KEYS:
        logger.debug(
            "Value sample for '%s' | data: %s | labels: %s",
            key,
            data_df[key].dropna().astype(str).unique()[:5].tolist(),
            labels_df[key].dropna().astype(str).unique()[:5].tolist(),
        )

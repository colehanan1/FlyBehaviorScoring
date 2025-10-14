"""I/O utilities for flybehavior_response."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .logging_utils import get_logger

MERGE_KEYS = ["fly", "fly_number", "trial_label"]
OPTIONAL_KEYS = ["dataset", "trial_type"]
LABEL_COLUMN = "user_score_odor"
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


def load_and_merge(data_csv: Path, labels_csv: Path, logger_name: str = __name__) -> MergedDataset:
    """Load and merge data and labels CSVs."""
    logger = get_logger(logger_name)
    logger.info("Loading data CSV: %s", data_csv)
    data_df = _load_csv(data_csv)
    logger.debug("Data shape: %s", data_df.shape)

    logger.info("Loading labels CSV: %s", labels_csv)
    labels_df = _load_csv(labels_csv)
    logger.debug("Labels shape: %s", labels_df.shape)

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

    invalid = set(label_values.dropna().unique()) - {0, 1}
    if invalid:
        raise DataValidationError(
            f"Invalid labels detected {invalid}. Labels must be 0 or 1."
        )

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
    merged = pd.merge(data_df, labels_df[[*MERGE_KEYS, LABEL_COLUMN]], on=MERGE_KEYS, how="inner", validate="one_to_one")

    if merged.empty:
        raise DataValidationError("Merge produced no rows. Verify matching keys across CSVs.")

    merged.sort_values(MERGE_KEYS, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    if merged[LABEL_COLUMN].isna().any():
        raise DataValidationError("Merged data contains NaN labels after merge.")

    logger.info("Merged dataset shape: %s", merged.shape)

    return MergedDataset(frame=merged, trace_columns=trace_cols, feature_columns=feature_cols)


def write_parquet(dataset: MergedDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_parquet(path, index=False)

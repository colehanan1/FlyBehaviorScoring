"""Data loading and preprocessing utilities for unsupervised clustering."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


TIME_COLUMN_PATTERN = re.compile(r"dir[\W_]*val[\W_]*(\d+)", re.IGNORECASE)


@dataclass
class PreparedData:
    """Container for standardized traces and associated metadata."""

    traces: np.ndarray
    metadata: pd.DataFrame
    time_columns: List[str]

    @property
    def n_trials(self) -> int:
        return self.traces.shape[0]

    @property
    def n_timepoints(self) -> int:
        return self.traces.shape[1]


def _load_inputs(npy_path: Path, meta_path: Path) -> Tuple[np.ndarray, dict]:
    if not npy_path.exists():
        raise FileNotFoundError(f"Missing npy file: {npy_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    data = np.load(npy_path)
    with meta_path.open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return data, meta


def _to_python_scalar(value: object) -> object:
    """Convert NumPy scalar types to native Python scalars."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_mapping(mapping: object) -> dict:
    """Return a dictionary with Python-native keys for mapping operations."""

    if isinstance(mapping, dict):
        items: Iterable[tuple] = mapping.items()
    elif isinstance(mapping, Iterable) and not isinstance(mapping, (str, bytes)):
        items = mapping  # type: ignore[assignment]
    else:
        raise TypeError("code_maps entries must be dicts or iterable pairs")

    normalized: dict = {}
    for key, value in items:  # type: ignore[misc]
        normalized[_to_python_scalar(key)] = value
    return normalized


def _decode_categorical_columns(df: pd.DataFrame, code_maps: dict) -> pd.DataFrame:
    for column, raw_mapping in code_maps.items():
        if column not in df.columns:
            continue

        mapping = _normalize_mapping(raw_mapping)
        df[column] = df[column].apply(
            lambda raw: mapping.get(_to_python_scalar(raw), raw)
        )
    return df


def _identify_time_columns(columns: Sequence[str]) -> List[str]:
    extracted: List[Tuple[str, int]] = []
    for column in columns:
        match = TIME_COLUMN_PATTERN.search(column)
        if match:
            try:
                index = int(match.group(1))
            except ValueError:
                # Skip columns where the captured group is not an integer.
                continue
            extracted.append((column, index))

    extracted.sort(key=lambda pair: pair[1])
    return [name for name, _ in extracted]


def _canonicalize_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "-")
    if "3" in normalized and "oct" in normalized:
        return "3-octonol"
    if normalized in {"eb", "ethyl-butyrate", "ethylbutyrate"}:
        return "EB"
    return name


def prepare_data(
    npy_path: Path,
    meta_path: Path,
    *,
    target_datasets: Sequence[str] | None = None,
) -> PreparedData:
    """Load, filter, and z-score trials for clustering.

    Parameters
    ----------
    npy_path: Path
        Path to the trial matrix stored as a NumPy array.
    meta_path: Path
        Path to JSON metadata describing column order and categorical maps.
    target_datasets: Sequence[str] | None, optional keyword-only
        Dataset names to retain. Defaults to {"EB", "3-octonol"}.

    Returns
    -------
    PreparedData
        The standardized traces alongside metadata and ordered time columns.
    """

    matrix, meta = _load_inputs(npy_path, meta_path)

    column_order: Sequence[str] = meta["column_order"]
    code_maps: dict = meta.get("code_maps", {})

    if matrix.shape[1] != len(column_order):
        raise ValueError(
            "Mismatch between matrix feature count and metadata column order"
        )

    df = pd.DataFrame(matrix, columns=column_order)
    df = _decode_categorical_columns(df, code_maps)

    time_columns = _identify_time_columns(df.columns)
    if not time_columns:
        sample_columns = ", ".join(list(df.columns[:10]))
        raise ValueError(
            "No time-series columns found matching pattern similar to 'dir_val_#'. "
            f"First columns: [{sample_columns}]"
        )

    # Canonicalize dataset names to expected values.
    if "dataset_name" in df.columns:
        df["dataset_name"] = df["dataset_name"].astype(str).map(_canonicalize_dataset_name)

    # Filter for testing trials and targeted datasets.
    trial_type_series = df.get("trial_type_name", pd.Series("", index=df.index)).astype(str)
    dataset_series = df.get("dataset_name", pd.Series("", index=df.index)).astype(str)
    filters = trial_type_series.str.lower() == "testing"
    if target_datasets is None:
        dataset_candidates = {"EB", "3-octonol"}
    else:
        dataset_candidates = {
            _canonicalize_dataset_name(str(candidate)) for candidate in target_datasets
        }
    dataset_mask = dataset_series.isin(dataset_candidates)
    combined_mask = filters & dataset_mask
    filtered = df.loc[combined_mask].reset_index(drop=True)

    if filtered.empty:
        raise ValueError(
            "No trials remaining after filtering for testing trials in datasets: "
            f"{sorted(dataset_candidates)}."
        )

    traces = filtered[time_columns].to_numpy(dtype=float)

    # Z-score per trial (row-wise standardization).
    means = traces.mean(axis=1, keepdims=True)
    stds = traces.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    zscored = (traces - means) / stds

    metadata_columns = [col for col in filtered.columns if col not in time_columns]
    metadata = filtered[metadata_columns].copy()

    return PreparedData(traces=zscored, metadata=metadata, time_columns=list(time_columns))


__all__ = ["PreparedData", "prepare_data"]

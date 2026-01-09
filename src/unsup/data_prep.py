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
MAX_TIME_COLUMNS = 3600


@dataclass
class PreparedData:
    """Container for standardized traces and associated metadata."""

    traces: np.ndarray
    metadata: pd.DataFrame
    time_columns: List[str]
    measurement_columns: List[str]

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

        # Some metadata files store mappings in label -> code form instead of
        # code -> label. Detect this by checking whether any observed column
        # values appear as mapping keys; if not, but they do appear among the
        # mapping values, invert the dictionary so numeric codes map onto their
        # human-readable labels.
        column_values = {
            _to_python_scalar(value) for value in df[column].dropna().unique()
        }
        mapping_keys = set(mapping.keys())
        if column_values and not (column_values & mapping_keys):
            mapping_values = {
                _to_python_scalar(value) for value in mapping.values()
            }
            if column_values & mapping_values:
                inverted: dict = {}
                for key, value in mapping.items():
                    inverted[_to_python_scalar(value)] = key
                mapping = inverted

        df[column] = df[column].apply(
            lambda raw: mapping.get(_to_python_scalar(raw), raw)
        )
    return df


def _normalize_column_key(name: str) -> str:
    """Return a simplified key for fuzzy column matching."""

    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _stringify(value: object) -> str:
    if pd.isna(value):  # type: ignore[arg-type]
        return ""
    return str(_to_python_scalar(value))


def _extract_string_series(
    df: pd.DataFrame,
    candidate_names: Sequence[str],
    *,
    default_value: str = "",
) -> Tuple[pd.Series, str | None]:
    """Return the first matching column converted to strings."""

    matched_name: str | None = None
    for candidate in candidate_names:
        candidate_key = _normalize_column_key(candidate)
        for column in df.columns:
            if _normalize_column_key(column) == candidate_key:
                matched_name = column
                break
        if matched_name is not None:
            break

    if matched_name is None:
        return pd.Series(default_value, index=df.index, dtype=str), None

    series = df[matched_name].apply(_stringify)
    return series.astype(str), matched_name


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
    """Map dataset aliases onto canonical EB/3-octonol labels."""

    # Collapse punctuation and underscores so variants like ``opto_EB`` or
    # ``Manual 3 Octonol`` normalize to predictable tokens.
    normalized = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    if not normalized:
        return str(name)

    tokens = normalized.split()
    joined = "".join(tokens)

    if any(token.startswith("3") for token in tokens) and "oct" in normalized:
        return "3-octonol"
    if "3oct" in joined or "threeoct" in joined:
        return "3-octonol"

    if "eb" in tokens or joined in {"eb", "ethylbutyrate"}:
        return "EB"
    if "ethyl" in tokens and "butyrate" in tokens:
        return "EB"

    return str(name)


def prepare_data(
    npy_path: Path,
    meta_path: Path,
    *,
    target_datasets: Sequence[str] | None = None,
    debug: bool = False,
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

    if debug:
        print(
            "[prepare_data] Loaded inputs:",
            f"matrix_shape={matrix.shape}",
            f"n_columns_meta={len(meta.get('column_order', []))}",
        )

    column_order: Sequence[str] = meta["column_order"]
    code_maps: dict = meta.get("code_maps", {})

    if matrix.shape[1] != len(column_order):
        raise ValueError(
            "Mismatch between matrix feature count and metadata column order"
        )

    df = pd.DataFrame(matrix, columns=column_order)
    if debug:
        preview_cols = ", ".join(column_order[:8])
        print(f"[prepare_data] Constructed DataFrame with columns: {preview_cols}...")

    df = _decode_categorical_columns(df, code_maps)

    time_columns = _identify_time_columns(df.columns)
    if not time_columns:
        sample_columns = ", ".join(list(df.columns[:10]))
        raise ValueError(
            "No time-series columns found matching pattern similar to 'dir_val_#'. "
            f"First columns: [{sample_columns}]"
        )

    if len(time_columns) > MAX_TIME_COLUMNS:
        extra_time_columns = time_columns[MAX_TIME_COLUMNS:]
        df = df.drop(columns=extra_time_columns)
        time_columns = time_columns[:MAX_TIME_COLUMNS]
        if debug:
            print(
                "[prepare_data] Truncating time-series columns to first",
                MAX_TIME_COLUMNS,
                "entries",
            )

    dataset_series_raw, dataset_column = _extract_string_series(
        df,
        (
            "dataset_name",
            "dataset",
            "dataset_label",
            "dataset_full",
            "dataset_id",
            "datasetname",
        ),
    )
    dataset_series = dataset_series_raw.map(_canonicalize_dataset_name)
    df["dataset_name"] = dataset_series

    if debug:
        raw_counts = dataset_series_raw.value_counts(dropna=False).to_dict()
        canonical_counts = dataset_series.value_counts(dropna=False).to_dict()
        print(
            "[prepare_data] Dataset column detected:",
            dataset_column or "<missing>",
        )
        print(
            "[prepare_data] Dataset counts (raw -> canonical):",
            raw_counts,
            canonical_counts,
        )

    trial_type_series, trial_type_column = _extract_string_series(
        df,
        (
            "trial_type_name",
            "trial_type",
            "trialtype",
            "trial_type_label",
            "trial_label",
        ),
    )
    df["trial_type_name"] = trial_type_series

    if debug:
        trial_counts = trial_type_series.value_counts(dropna=False).to_dict()
        print(
            "[prepare_data] Trial type column detected:",
            trial_type_column or "<missing>",
        )
        print("[prepare_data] Trial type counts:", trial_counts)

    # Filter for testing trials and targeted datasets.
    trial_type_lower = trial_type_series.str.strip().str.lower()
    filters = trial_type_lower == "testing"
    if target_datasets is None:
        dataset_candidates = {"EB", "3-octonol"}
    else:
        dataset_candidates = {
            _canonicalize_dataset_name(str(candidate)) for candidate in target_datasets
        }
    dataset_mask = dataset_series.isin(dataset_candidates)
    combined_mask = filters & dataset_mask
    filtered = df.loc[combined_mask].reset_index(drop=True)

    if debug:
        print(
            "[prepare_data] Target datasets after canonicalization:",
            sorted(dataset_candidates),
        )
        print(
            "[prepare_data] Trials passing dataset filter:",
            int(dataset_mask.sum()),
            "/",
            len(df),
        )
        print(
            "[prepare_data] Trials passing combined filter:",
            len(filtered),
            "/",
            len(df),
        )
        if not filtered.empty:
            filtered_dataset_counts = (
                filtered["dataset_name"].value_counts(dropna=False).to_dict()
            )
            filtered_trial_counts = (
                filtered["trial_type_name"].value_counts(dropna=False).to_dict()
            )
            print(
                "[prepare_data] Filtered dataset counts:",
                filtered_dataset_counts,
            )
            print(
                "[prepare_data] Filtered trial type counts:",
                filtered_trial_counts,
            )

    if filtered.empty:
        available_datasets = sorted({
            _canonicalize_dataset_name(value) for value in dataset_series.unique()
        })
        available_trial_types = sorted({value.lower() for value in trial_type_series.unique()})
        raise ValueError(
            "No trials remaining after filtering for testing trials in datasets: "
            f"{sorted(dataset_candidates)}. "
            f"Available dataset_name values after canonicalization: {available_datasets}. "
            f"Available trial_type_name values: {available_trial_types}. "
            f"Dataset column used: {dataset_column or 'not found'}. "
            f"Trial type column used: {trial_type_column or 'not found'}"
        )

    traces = filtered[time_columns].to_numpy(dtype=float)

    # Z-score per trial (row-wise standardization).
    means = traces.mean(axis=1, keepdims=True)
    stds = traces.std(axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    zscored = (traces - means) / stds

    metadata_columns = [col for col in filtered.columns if col not in time_columns]
    metadata = filtered[metadata_columns].copy()

    measurement_columns = [
        column
        for column in metadata.columns
        if column not in {"dataset_name", "trial_type_name"}
        and pd.api.types.is_numeric_dtype(metadata[column])
    ]

    return PreparedData(
        traces=zscored,
        metadata=metadata,
        time_columns=list(time_columns),
        measurement_columns=measurement_columns,
    )


__all__ = ["PreparedData", "prepare_data"]

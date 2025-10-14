"""Utility helpers shared across the stats package."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import trim_mean
from statsmodels.stats.multitest import multipletests

LOG = logging.getLogger("stats.utils")

GROUP_A_LABEL = "Trained odor testing trials (2, 4, 5)"
GROUP_B_LABEL = "Other odor testing trials"

TIME_COLUMN_PATTERN = re.compile(r"(dir|frame|time)[^0-9]*([0-9]+)", re.IGNORECASE)
TRIAL_NUMBER_PATTERN = re.compile(r"(\d+)")
LIKELY_METADATA_NAMES = {
    "fps",
    "frame_rate",
    "trial_type",
    "trial_label",
    "trialtype",
    "trial_name",
    "trialcategory",
    "odor",
    "odor_name",
    "odor_label",
    "odor_conc",
    "odor_concentration",
    "odorconc",
    "stimulus",
    "stimulus_name",
}


@dataclass
class FlyGroup:
    """Container storing all trials for a single fly within a condition."""

    fly_id: str
    trials: np.ndarray

    def mean_trace(self) -> np.ndarray:
        """Return the mean trace for the stored trials, guarding against empties."""

        if self.trials.size == 0:
            raise ValueError(f"Fly {self.fly_id} has no trials in requested subset.")
        return np.nanmean(self.trials, axis=0)


@dataclass
class FlyGroups:
    """Encapsulate aligned Group A and Group B data for a fly."""

    fly_id: str
    group_a: FlyGroup
    group_b: FlyGroup

    @property
    def has_both(self) -> bool:
        return self.group_a.trials.size > 0 and self.group_b.trials.size > 0


@dataclass
class LoadedMetadata:
    """Structured view of metadata JSON contents."""

    rows: Optional[List[dict]]
    column_order: Optional[Sequence[str]]
    code_maps: Dict[str, Any]
    raw: Any


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def parse_trial_list(raw: str) -> List[int]:
    """Parse a comma-separated list of trial identifiers into integers."""

    trials: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            trials.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid trial identifier '{item}' (must be int).") from exc
    if not trials:
        raise ValueError("Target trials list cannot be empty.")
    return trials


def parse_datasets(raw: str) -> List[str]:
    """Return a list of dataset labels from a comma-separated string."""

    datasets = [item.strip() for item in raw.split(",") if item.strip()]
    if not datasets:
        raise ValueError("At least one dataset label must be provided.")
    return datasets


def ensure_out_dir(path: str) -> None:
    """Create the output directory if it does not exist."""

    os.makedirs(path, exist_ok=True)


def load_matrix(path: str) -> np.ndarray:
    """Load a trial matrix from disk and ensure floating dtype."""

    LOG.debug("Loading matrix from %s", path)
    mat = np.load(path)
    if mat.ndim != 2:
        raise ValueError(f"Expected a 2D matrix [rows, time]; got shape={mat.shape}")
    if not np.issubdtype(mat.dtype, np.floating):
        LOG.warning("Matrix dtype %s is not float; casting to float32 for safety.", mat.dtype)
        mat = mat.astype(np.float32, copy=False)
    return np.asarray(mat)


# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def load_metadata(path: str) -> LoadedMetadata:
    """Load metadata JSON handling a variety of layouts."""

    with open(path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if isinstance(meta, dict):
        rows = _extract_rows(meta)
        column_order: Optional[Sequence[str]] = None
        if rows is None:
            column_order = meta.get("column_order") or meta.get("columns")
        return LoadedMetadata(
            rows=rows,
            column_order=column_order,
            code_maps=meta.get("code_maps", {}),
            raw=meta,
        )
    if isinstance(meta, list) and all(isinstance(item, dict) for item in meta):
        return LoadedMetadata(rows=list(meta), column_order=None, code_maps={}, raw=meta)
    raise ValueError("Unsupported metadata JSON structure; expected dict or list of dicts.")


def rows_to_dataframe(
    rows: Sequence[dict],
    fly_field: str,
    dataset_field: str,
    trial_field: str,
) -> pd.DataFrame:
    """Convert a list of row dictionaries into a canonical DataFrame."""

    df = pd.DataFrame(list(rows))
    missing = [field for field in (fly_field, dataset_field, trial_field) if field not in df.columns]
    if missing:
        raise KeyError(
            "Metadata missing required keys: " + ", ".join(missing) + f". Present keys: {sorted(df.columns)}"
        )
    df.insert(0, "row", np.arange(len(df), dtype=int))
    return df


def select_datasets(df: pd.DataFrame, datasets: Sequence[str], dataset_field: str) -> pd.DataFrame:
    """Filter metadata rows to a requested set of dataset labels."""

    datasets_set = {str(item) for item in datasets}
    filtered = df[df[dataset_field].astype(str).isin(datasets_set)].copy()
    if filtered.empty:
        raise ValueError(
            "No metadata rows matched the requested datasets. "
            f"Requested={sorted(datasets_set)}, available={sorted(df[dataset_field].unique())}"
        )
    return filtered


def dataframe_from_columnar_matrix(
    matrix: np.ndarray,
    column_order: Sequence[str],
    code_maps: Optional[Dict[str, Any]],
    fly_field: str,
    dataset_field: str,
    trial_field: str,
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Decode column-oriented exports into traces and metadata."""

    if matrix.shape[1] != len(column_order):
        raise ValueError(
            "Mismatch between matrix column count and metadata column_order length: "
            f"matrix has {matrix.shape[1]} columns, metadata lists {len(column_order)} entries."
        )
    df = pd.DataFrame(matrix, columns=column_order)
    df = _apply_code_maps_to_dataframe(df, code_maps or {})

    time_columns = _identify_time_columns(df, (fly_field, dataset_field, trial_field))
    if not time_columns:
        sample = ", ".join(list(df.columns[:10]))
        raise ValueError(
            "Unable to identify time-series columns from metadata column_order. "
            "Expect names resembling 'dir_val_#' or numeric indices. "
            f"Sample columns: [{sample}]"
        )

    time_df = df[time_columns].apply(pd.to_numeric, errors="coerce")
    if time_df.isnull().values.any():
        LOG.warning("Non-numeric entries detected in time-series columns; coerced to NaN.")
    traces = time_df.to_numpy(dtype=float)

    meta_columns = [column for column in df.columns if column not in time_columns]
    meta_df = df[meta_columns].copy()
    meta_df.insert(0, "row", np.arange(len(meta_df), dtype=int))

    missing_core = [field for field in (fly_field, dataset_field) if field not in meta_df.columns]
    if missing_core:
        raise KeyError(
            "Metadata columns missing after decoding column_order/np matrix: "
            + ", ".join(missing_core)
            + ". Available columns: "
            + ", ".join(sorted(map(str, meta_df.columns.tolist())))
        )

    if trial_field not in meta_df.columns:
        LOG.warning(
            "Metadata missing trial field '%s'; attempting to infer from other columns.",
            trial_field,
        )
        inferred = False
        candidate_cols = [
            column
            for column in meta_df.columns
            if column not in {trial_field, "row"} and "trial" in str(column).lower()
        ]
        for column in candidate_cols:
            numeric_series = pd.to_numeric(meta_df[column], errors="coerce")
            if numeric_series.notna().all():
                meta_df[trial_field] = numeric_series.astype(int)
                LOG.info(
                    "Inferred trial numbers from column '%s' after numeric coercion.",
                    column,
                )
                inferred = True
                break
            extracted = meta_df[column].astype(str).str.extract(TRIAL_NUMBER_PATTERN, expand=False)
            if extracted.notna().all():
                meta_df[trial_field] = extracted.astype(int)
                LOG.info(
                    "Inferred trial numbers from digit pattern in column '%s'.",
                    column,
                )
                inferred = True
                break
        if not inferred:
            LOG.warning("Falling back to sequential trial numbering within each fly/dataset pair.")
            meta_df[trial_field] = meta_df.groupby([dataset_field, fly_field]).cumcount() + 1
            meta_df[trial_field] = meta_df[trial_field].astype(int)

    missing_after = [field for field in (fly_field, dataset_field, trial_field) if field not in meta_df.columns]
    if missing_after:
        raise KeyError(
            "Metadata columns missing after column_order decoding even after inference attempts: "
            + ", ".join(missing_after)
            + ". Available columns: "
            + ", ".join(sorted(map(str, meta_df.columns.tolist())))
        )

    try:
        meta_df[trial_field] = pd.to_numeric(meta_df[trial_field], errors="raise").astype(int)
    except Exception as exc:  # pragma: no cover - defensive conversion
        raise ValueError(f"Unable to coerce inferred trial field '{trial_field}' to integers.") from exc

    LOG.info(
        "Decoded columnar metadata: %d trials, %d timepoints inferred from %d time columns.",
        traces.shape[0],
        traces.shape[1],
        len(time_columns),
    )
    return traces, meta_df, list(time_columns)


# ---------------------------------------------------------------------------
# Metadata helpers (internal)
# ---------------------------------------------------------------------------

def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _flatten_code_map(mapping: Any) -> Dict[Any, Any]:
    if mapping is None:
        return {}
    if isinstance(mapping, dict):
        direct: Dict[Any, Any] = {}
        if mapping and all(isinstance(k, str) and k.isdigit() for k in mapping.keys()):
            direct = {int(k): v for k, v in mapping.items()}
        elif mapping and all(isinstance(v, (str, int, float, bool)) for v in mapping.values()):
            direct = dict(mapping)
        if direct:
            return direct
        for key in (
            "index_to_value",
            "index_to_label",
            "codes_to_values",
            "map",
            "mapping",
            "lookup",
        ):
            nested = mapping.get(key) if isinstance(mapping, dict) else None
            if isinstance(nested, dict):
                nested_map = _flatten_code_map(nested)
                if nested_map:
                    return nested_map
        for value_key in ("values", "labels"):
            values = mapping.get(value_key)
            if isinstance(values, list):
                keys = mapping.get("keys") or mapping.get("codes") or mapping.get("indices")
                if isinstance(keys, list) and len(keys) == len(values):
                    return {k: v for k, v in zip(keys, values)}
                return {idx: val for idx, val in enumerate(values)}
        if mapping and all(isinstance(v, (int, float)) for v in mapping.values()):
            inverted = {v: k for k, v in mapping.items()}
            if len(inverted) == len(mapping):
                return inverted
    if isinstance(mapping, list):
        return {idx: value for idx, value in enumerate(mapping)}
    return {}


def _decode_value_with_flat_map(value: Any, flat: Dict[Any, Any]) -> Any:
    scalar = _to_python_scalar(value)
    if not flat:
        return scalar
    candidates: List[Any] = [scalar]
    if isinstance(scalar, str) and scalar.isdigit():
        candidates.append(int(scalar))
    else:
        try:
            candidates.append(int(scalar))
        except (TypeError, ValueError):
            pass
    candidates.append(str(scalar))
    for cand in candidates:
        if cand in flat:
            return flat[cand]
    inverse = {v: k for k, v in flat.items() if isinstance(v, (str, int, float, bool))}
    for cand in candidates:
        if cand in inverse:
            return inverse[cand]
    return scalar


def _decode_with_map(column: str, value: Any, code_maps: Optional[Dict[str, Any]]) -> Any:
    if code_maps is None:
        return _to_python_scalar(value)
    mapping = code_maps.get(column)
    flat = _flatten_code_map(mapping)
    return _decode_value_with_flat_map(value, flat)


def _apply_code_maps_to_dataframe(df: pd.DataFrame, code_maps: Dict[str, Any]) -> pd.DataFrame:
    if not code_maps:
        return df
    for column, mapping in code_maps.items():
        if column not in df.columns:
            continue
        flat = _flatten_code_map(mapping)
        if not flat:
            continue
        df[column] = df[column].map(lambda value: _decode_value_with_flat_map(value, flat))
    return df


def _identify_time_columns(df: pd.DataFrame, required_fields: Sequence[str]) -> List[str]:
    required = {str(field) for field in required_fields}
    matches: List[Tuple[str, int]] = []
    for column in df.columns:
        if str(column) in required:
            continue
        match = TIME_COLUMN_PATTERN.search(str(column))
        if match:
            try:
                idx = int(match.group(2))
            except ValueError:
                continue
            matches.append((column, idx))
    if matches:
        matches.sort(key=lambda item: (item[1], str(item[0])))
        return [name for name, _ in matches]

    excluded = {str(field) for field in required_fields}
    excluded.update(name for name in df.columns if str(name).strip().lower() in LIKELY_METADATA_NAMES)
    fallback = [
        column
        for column in df.columns
        if column not in excluded and is_numeric_dtype(df[column])
    ]
    return fallback


def _rows_from_table(
    table: Any,
    columns: Sequence[str],
    code_maps: Optional[Dict[str, Any]],
) -> Optional[List[dict]]:
    if not isinstance(table, list):
        return None
    if not table:
        return []
    first = table[0]
    if isinstance(first, dict):
        rows: List[dict] = []
        for row in table:
            if not isinstance(row, dict):
                return None
            decoded = {key: _decode_with_map(key, row.get(key), code_maps) for key in row.keys()}
            rows.append(decoded)
        return rows
    if isinstance(first, (list, tuple)):
        rows_list: List[dict] = []
        for raw_row in table:
            if not isinstance(raw_row, (list, tuple)):
                return None
            decoded_row: Dict[str, Any] = {}
            limit = min(len(columns), len(raw_row))
            for idx in range(limit):
                col = columns[idx]
                decoded_row[col] = _decode_with_map(col, raw_row[idx], code_maps)
            rows_list.append(decoded_row)
        return rows_list
    return None


def _extract_rows(meta: Any) -> Optional[List[dict]]:
    if isinstance(meta, list):
        if all(isinstance(item, dict) for item in meta):
            return list(meta)
        return None
    if not isinstance(meta, dict):
        return None

    if "rows" in meta and isinstance(meta["rows"], list):
        rows = meta["rows"]
        if all(isinstance(item, dict) for item in rows):
            return rows
        columns = meta.get("columns") or meta.get("column_order")
        if isinstance(columns, list):
            decoded = _rows_from_table(rows, columns, meta.get("code_maps"))
            if decoded is not None:
                return decoded

    if "row_index_to_meta" in meta and isinstance(meta["row_index_to_meta"], dict):
        ordered = sorted(meta["row_index_to_meta"].items(), key=lambda kv: int(kv[0]))
        return [entry for _, entry in ordered]

    if "columns" in meta and "data" in meta and isinstance(meta["data"], list):
        decoded = _rows_from_table(meta["data"], meta["columns"], meta.get("code_maps"))
        if decoded is not None:
            return decoded

    if "column_order" in meta:
        table_key = next(
            (key for key in ("data", "values", "table") if key in meta and isinstance(meta[key], list)),
            None,
        )
        if table_key:
            decoded = _rows_from_table(meta[table_key], meta["column_order"], meta.get("code_maps"))
            if decoded is not None:
                return decoded
    return None


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def build_groups(
    matrix: np.ndarray,
    meta_df: pd.DataFrame,
    fly_field: str,
    trial_field: str,
    target_trials: Sequence[int],
) -> List[FlyGroups]:
    """Construct paired FlyGroups for each fly and filter to flies with both conditions."""

    by_fly: Dict[str, List[pd.Series]] = {}
    for _, row in meta_df.iterrows():
        fly = str(row[fly_field])
        by_fly.setdefault(fly, []).append(row)
    groups: List[FlyGroups] = []
    for fly, rows in sorted(by_fly.items()):
        fly_df = pd.DataFrame(rows)
        rows_a = fly_df[fly_df[trial_field].astype(int).isin(target_trials)]["row"].to_numpy(dtype=int)
        rows_b = fly_df[~fly_df[trial_field].astype(int).isin(target_trials)]["row"].to_numpy(dtype=int)
        trials_a = matrix[rows_a, :] if rows_a.size else np.empty((0, matrix.shape[1]), dtype=matrix.dtype)
        trials_b = matrix[rows_b, :] if rows_b.size else np.empty((0, matrix.shape[1]), dtype=matrix.dtype)
        LOG.debug(
            "Fly %s: Group A trials=%s, Group B trials=%s",
            fly,
            rows_a.tolist(),
            rows_b.tolist(),
        )
        groups.append(
            FlyGroups(
                fly_id=fly,
                group_a=FlyGroup(fly_id=fly, trials=trials_a),
                group_b=FlyGroup(fly_id=fly, trials=trials_b),
            )
        )
    usable = [g for g in groups if g.has_both]
    dropped = [g.fly_id for g in groups if not g.has_both]
    if dropped:
        LOG.warning("Dropping flies without at least one trial in both groups: %s", ", ".join(dropped))
    if not usable:
        raise ValueError("No flies retained after filtering for both Group A and Group B trials.")
    return usable


def stack_mean_traces(groups: Sequence[FlyGroups]) -> Tuple[np.ndarray, np.ndarray]:
    """Return stacked mean traces (A, B) for the provided groups."""

    mean_a = np.vstack([g.group_a.mean_trace() for g in groups])
    mean_b = np.vstack([g.group_b.mean_trace() for g in groups])
    return mean_a, mean_b


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini–Hochberg FDR correction preserving NaNs."""

    qvals = np.full_like(pvals, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    if mask.sum() == 0:
        return qvals
    _, q, _, _ = multipletests(pvals[mask], alpha=alpha, method="fdr_bh")
    qvals[mask] = q.astype(float)
    return qvals


def find_contiguous_windows(time_s: np.ndarray, mask: np.ndarray) -> List[Dict[str, float]]:
    """Identify contiguous True segments in a boolean mask."""

    windows: List[Dict[str, float]] = []
    if mask.size == 0:
        return windows
    start_idx: Optional[int] = None
    for idx, value in enumerate(mask):
        if value and start_idx is None:
            start_idx = idx
        elif not value and start_idx is not None:
            windows.append(
                {
                    "start_idx": start_idx,
                    "end_idx": idx - 1,
                    "start_s": float(time_s[start_idx]),
                    "end_s": float(time_s[idx - 1]),
                    "duration_s": float(time_s[idx - 1] - time_s[start_idx]) if idx - 1 >= start_idx else 0.0,
                }
            )
            start_idx = None
    if start_idx is not None:
        windows.append(
            {
                "start_idx": start_idx,
                "end_idx": mask.size - 1,
                "start_s": float(time_s[start_idx]),
                "end_s": float(time_s[-1]),
                "duration_s": float(time_s[-1] - time_s[start_idx]),
            }
        )
    return windows


def save_windows_csv(windows: List[Dict[str, float]], out_path: str) -> None:
    """Save contiguous windows metadata to CSV."""

    if not windows:
        pd.DataFrame(columns=["start_s", "end_s", "duration_s"]).to_csv(out_path, index=False)
        return
    df = pd.DataFrame(windows)
    keep = ["start_s", "end_s", "duration_s"]
    existing = [col for col in keep if col in df.columns]
    df[existing].to_csv(out_path, index=False)


def latency_to_threshold(
    traces: np.ndarray,
    threshold: float,
    window: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute first-crossing latencies (in samples) and censor indicators."""

    if window is None:
        start, end = 0, traces.shape[1]
    else:
        start, end = window
        if start < 0 or end <= start or end > traces.shape[1]:
            raise ValueError(
                f"Invalid window {window}; must satisfy 0 <= start < end <= trace length {traces.shape[1]}"
            )
    latencies = np.empty(traces.shape[0], dtype=float)
    events = np.zeros(traces.shape[0], dtype=bool)
    search_len = end - start
    for idx, trace in enumerate(traces):
        segment = trace[start:end]
        crossings = np.where(segment >= threshold)[0]
        if crossings.size:
            first = crossings[0]
            if first >= search_len - 1:
                latencies[idx] = start + search_len - 1
                events[idx] = False
            else:
                latencies[idx] = start + first
                events[idx] = True
        else:
            latencies[idx] = start + search_len - 1
            events[idx] = False
    return latencies, events


def trimmed_mean_diff(
    diffs: np.ndarray,
    trim_pct: float,
) -> Tuple[float, float, int]:
    """Compute Yuen trimmed-mean difference, winsorized variance, and effective df."""

    if diffs.ndim != 1:
        raise ValueError("Diff array must be 1D for trimmed mean computation.")
    if not 0.0 <= trim_pct < 0.5:
        raise ValueError("trim_pct must be within [0, 0.5).")
    n = diffs.size
    g = int(np.floor(trim_pct * n))
    if n - 2 * g <= 2:
        raise ValueError("Not enough samples after trimming to compute Yuen statistic.")
    sorted_diff = np.sort(diffs)
    trimmed_mean = float(trim_mean(diffs, proportiontocut=trim_pct))
    winsorized = sorted_diff.copy()
    winsorized[:g] = sorted_diff[g]
    winsorized[n - g :] = sorted_diff[n - g - 1]
    winsor_mean = float(np.mean(winsorized))
    winsor_var = float(np.sum((winsorized - winsor_mean) ** 2) / (n - 1))
    se = np.sqrt(winsor_var / ((n - 2 * g) * (n - 2 * g - 1)))
    t_stat = trimmed_mean / se if se > 0 else np.nan
    df = n - 2 * g - 1
    return t_stat, winsor_var, df


def yuen_p_value(diffs: np.ndarray, trim_pct: float) -> Tuple[float, float]:
    """Compute Yuen trimmed-mean t-statistic and two-sided p-value."""

    from scipy.stats import t as student_t

    mask = np.isfinite(diffs)
    diffs = diffs[mask]
    if diffs.size < 3:
        return np.nan, np.nan
    t_stat, _, df = trimmed_mean_diff(diffs, trim_pct)
    if not np.isfinite(t_stat):
        return t_stat, np.nan
    p = 2 * student_t.sf(np.abs(t_stat), df)
    return t_stat, p


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def update_manifest_entry(
    manifest: Dict[str, Any],
    key: str,
    entry: Dict[str, Any],
) -> None:
    """Append or create a manifest entry for a particular analysis type."""

    manifest.setdefault(key, [])
    manifest[key].append(entry)


def compute_time_axis(length: int, hz: float) -> np.ndarray:
    """Return a uniformly sampled time axis in seconds."""

    return np.arange(length, dtype=float) / float(hz)


def diff_matrix(groups: Sequence[FlyGroups]) -> np.ndarray:
    """Return the fly-by-time difference matrix (A minus B)."""

    mean_a, mean_b = stack_mean_traces(groups)
    return mean_a - mean_b


__all__ = [
    "FlyGroup",
    "FlyGroups",
    "LoadedMetadata",
    "GROUP_A_LABEL",
    "GROUP_B_LABEL",
    "parse_trial_list",
    "parse_datasets",
    "ensure_out_dir",
    "load_matrix",
    "load_metadata",
    "rows_to_dataframe",
    "select_datasets",
    "dataframe_from_columnar_matrix",
    "build_groups",
    "stack_mean_traces",
    "bh_fdr",
    "find_contiguous_windows",
    "save_windows_csv",
    "latency_to_threshold",
    "trimmed_mean_diff",
    "yuen_p_value",
    "update_manifest_entry",
    "compute_time_axis",
    "diff_matrix",
]

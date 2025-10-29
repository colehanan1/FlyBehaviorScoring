"""I/O utilities for flybehavior_response."""

from __future__ import annotations

import re
import math
from collections import defaultdict
from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, MutableMapping, Sequence

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np

from .io_wide import find_series_columns
from .logging_utils import get_logger

MERGE_KEYS = ["dataset", "fly", "fly_number", "trial_type", "trial_label"]
LABEL_COLUMN = "user_score_odor"
LABEL_INTENSITY_COLUMN = "user_score_odor_intensity"
TRACE_PATTERN = re.compile(r"^dir_val_(\d+)$")
TRACE_RANGE = (0, 3600)
DEFAULT_TRACE_PREFIXES = ["dir_val_"]
RAW_TRACE_PREFIXES = ["eye_x_f", "eye_y_f", "prob_x_f", "prob_y_f"]
RAW_GEOM_TRACE_CANDIDATES = {
    "eye_x": "eye_x_f",
    "eye_y": "eye_y_f",
    "prob_x": "prob_x_f",
    "prob_y": "prob_y_f",
}

DEFAULT_FPS = 40
ODOR_ON_COLUMN = "odor_on_idx"
ODOR_OFF_COLUMN = "odor_off_idx"
R_PCT_COLUMN = "r_pct_robust_fly"
DX_COLUMN = "dx"
DY_COLUMN = "dy"
EYE_X_COLUMN = "eye_x"
EYE_Y_COLUMN = "eye_y"
PROB_X_COLUMN = "prob_x"
PROB_Y_COLUMN = "prob_y"
IS_BEFORE_COLUMN = "is_before"
IS_DURING_COLUMN = "is_during"
IS_AFTER_COLUMN = "is_after"
HIGH_EXTENSION_THRESHOLD = 75.0
FEATURE_COLUMNS = {
    "AUC-Before",
    "AUC-During",
    "AUC-After",
    "AUC-During-Before-Ratio",
    "AUC-After-Before-Ratio",
    "TimeToPeak-During",
    "Peak-Value",
    "global_min",
    "global_max",
    "local_min",
    "local_max",
    "local_min_during",
    "local_max_during",
    "local_max_over_global_min",
    "local_max_during_over_global_min",
    "local_max_during_odor",
    "local_max_during_odor_over_global_min",
}

TRIAL_SUMMARY_REQUIRED_COLUMNS = [
    "dataset",
    "fly",
    "fly_number",
    "trial_type",
    "trial_label",
    "W_est_fly",
    "H_est_fly",
    "diag_est_fly",
    "r_min_fly",
    "r_max_fly",
    "r_p01_fly",
    "r_p99_fly",
    "r_mean_fly",
    "r_std_fly",
    "n_frames",
    "r_mean_trial",
    "r_std_trial",
    "r_max_trial",
    "r95_trial",
    "dx_mean_abs",
    "dy_mean_abs",
    "r_pct_robust_fly_max",
    "r_pct_robust_fly_mean",
    "r_before_mean",
    "r_before_std",
    "r_during_mean",
    "r_during_std",
    "r_during_minus_before_mean",
    "cos_theta_during_mean",
    "sin_theta_during_mean",
    "direction_consistency",
    "frac_high_ext_during",
    "rise_speed",
]

TRIAL_SUMMARY_DUPLICATE_COLUMNS = {
    "r_before_mean",
    "r_before_std",
    "r_during_mean",
    "r_during_std",
    "r_during_minus_before_mean",
    "cos_theta_during_mean",
    "sin_theta_during_mean",
    "direction_consistency",
    "frac_high_ext_during",
    "rise_speed",
}


class DataValidationError(RuntimeError):
    """Raised when data schema validation fails."""


SUPPORTED_AGG_STATS = {"mean", "std", "min", "max", "sum", "first", "last"}


STRING_KEY_COLUMNS = ["dataset", "fly", "trial_type", "trial_label"]

FRAME_COLUMN_ALIASES = {
    "frame": "frame",
}


@dataclass(slots=True)
class GeometryFrameStream:
    """Iterator wrapper that exposes geometry stream metadata."""

    _iterator: Iterator[pd.DataFrame]
    frame_column: str | None
    available_columns: List[str]

    def __iter__(self) -> "GeometryFrameStream":
        return self

    def __next__(self) -> pd.DataFrame:
        return next(self._iterator)


@dataclass(slots=True)
class BehavioralFeatureConfig:
    """Configuration describing responder feature enrichment requirements."""

    r_pct_column: str = R_PCT_COLUMN
    dx_column: str | None = DX_COLUMN
    dy_column: str | None = DY_COLUMN
    eye_x_column: str = EYE_X_COLUMN
    eye_y_column: str = EYE_Y_COLUMN
    prob_x_column: str = PROB_X_COLUMN
    prob_y_column: str = PROB_Y_COLUMN
    odor_on_column: str = ODOR_ON_COLUMN
    odor_off_column: str = ODOR_OFF_COLUMN
    is_before_column: str = IS_BEFORE_COLUMN
    is_during_column: str = IS_DURING_COLUMN
    is_after_column: str = IS_AFTER_COLUMN
    fps: int = DEFAULT_FPS
    high_extension_threshold: float = HIGH_EXTENSION_THRESHOLD


@dataclass(slots=True)
class TrialBehaviorAccumulator:
    """Running statistics for responder-oriented aggregates."""

    baseline_count: int = 0
    baseline_sum: float = 0.0
    baseline_sum_sq: float = 0.0
    during_count: int = 0
    during_sum: float = 0.0
    during_sum_sq: float = 0.0
    high_extension_count: int = 0
    direction_cos_sum: float = 0.0
    direction_sin_sum: float = 0.0
    direction_count: int = 0
    first_second_sum: float = 0.0
    first_second_count: int = 0

    def update(
        self,
        group: pd.DataFrame,
        *,
        config: BehavioralFeatureConfig,
        frame_column: str | None,
    ) -> None:
        if config.r_pct_column not in group.columns:
            return
        if config.is_before_column not in group.columns or config.is_during_column not in group.columns:
            return

        r_values = pd.to_numeric(group[config.r_pct_column], errors="coerce")
        before_mask = group[config.is_before_column].fillna(0).astype(bool)
        during_mask = group[config.is_during_column].fillna(0).astype(bool)

        valid_before = before_mask & r_values.notna()
        valid_during = during_mask & r_values.notna()

        if valid_before.any():
            before_vals = r_values.loc[valid_before].astype(float)
            self.baseline_count += len(before_vals)
            self.baseline_sum += float(before_vals.sum())
            self.baseline_sum_sq += float((before_vals**2).sum())

        if valid_during.any():
            during_vals = r_values.loc[valid_during].astype(float)
            self.during_count += len(during_vals)
            self.during_sum += float(during_vals.sum())
            self.during_sum_sq += float((during_vals**2).sum())
            threshold = config.high_extension_threshold
            if threshold is not None:
                self.high_extension_count += int((during_vals >= threshold).sum())

        # Direction metrics
        dx_series: pd.Series | None = None
        dy_series: pd.Series | None = None
        if config.dx_column and config.dx_column in group.columns and config.dy_column and config.dy_column in group.columns:
            dx_series = pd.to_numeric(group[config.dx_column], errors="coerce")
            dy_series = pd.to_numeric(group[config.dy_column], errors="coerce")
        elif (
            config.eye_x_column in group.columns
            and config.eye_y_column in group.columns
            and config.prob_x_column in group.columns
            and config.prob_y_column in group.columns
        ):
            dx_series = pd.to_numeric(group[config.prob_x_column], errors="coerce") - pd.to_numeric(
                group[config.eye_x_column], errors="coerce"
            )
            dy_series = pd.to_numeric(group[config.prob_y_column], errors="coerce") - pd.to_numeric(
                group[config.eye_y_column], errors="coerce"
            )

        if dx_series is not None and dy_series is not None:
            vector_valid = dx_series.notna() & dy_series.notna() & valid_during
            if vector_valid.any():
                dx_vals = dx_series.loc[vector_valid].astype(float)
                dy_vals = dy_series.loc[vector_valid].astype(float)
                magnitude = np.sqrt(dx_vals**2 + dy_vals**2)
                nonzero = magnitude > 0
                if nonzero.any():
                    dx_vals = dx_vals.loc[nonzero]
                    dy_vals = dy_vals.loc[nonzero]
                    magnitude = magnitude.loc[nonzero]
                    cos_vals = dx_vals / magnitude
                    sin_vals = dy_vals / magnitude
                    self.direction_cos_sum += float(cos_vals.sum())
                    self.direction_sin_sum += float(sin_vals.sum())
                    self.direction_count += len(cos_vals)

        if frame_column and frame_column in group.columns and valid_during.any():
            frames = pd.to_numeric(group[frame_column], errors="coerce")
            odor_on_series = group.get(config.odor_on_column)
            odor_off_series = group.get(config.odor_off_column)
            if not isinstance(odor_on_series, pd.Series) or not isinstance(odor_off_series, pd.Series):
                return
            odor_on_values = pd.to_numeric(odor_on_series, errors="coerce")
            odor_off_values = pd.to_numeric(odor_off_series, errors="coerce")
            first_valid = (
                frames.notna()
                & odor_on_values.notna()
                & odor_off_values.notna()
                & valid_during
            )
            if first_valid.any():
                odor_on_unique = odor_on_values.loc[first_valid].dropna().unique()
                odor_off_unique = odor_off_values.loc[first_valid].dropna().unique()
                if len(odor_on_unique) == 1 and len(odor_off_unique) == 1:
                    odor_on_idx = int(odor_on_unique[0])
                    odor_off_idx = int(odor_off_unique[0])
                    window_end = min(odor_on_idx + config.fps, odor_off_idx + 1)
                    window_mask = (
                        frames.loc[first_valid].astype(float) >= odor_on_idx
                    ) & (frames.loc[first_valid].astype(float) < window_end)
                    if window_mask.any():
                        window_values = r_values.loc[first_valid].astype(float).loc[window_mask]
                        if not window_values.empty:
                            self.first_second_sum += float(window_values.sum())
                            self.first_second_count += len(window_values)

    def _mean(self, total: float, count: int) -> float:
        return total / count if count else math.nan

    def baseline_mean(self) -> float:
        return self._mean(self.baseline_sum, self.baseline_count)

    def baseline_std(self) -> float:
        if self.baseline_count > 1:
            mean = self.baseline_sum / self.baseline_count
            variance = max(self.baseline_sum_sq / self.baseline_count - mean**2, 0.0)
            return math.sqrt(variance * self.baseline_count / (self.baseline_count - 1))
        return math.nan

    def during_mean(self) -> float:
        return self._mean(self.during_sum, self.during_count)

    def during_std(self) -> float:
        if self.during_count > 1:
            mean = self.during_sum / self.during_count
            variance = max(self.during_sum_sq / self.during_count - mean**2, 0.0)
            return math.sqrt(variance * self.during_count / (self.during_count - 1))
        return math.nan

    def direction_means(self) -> tuple[float, float]:
        if self.direction_count == 0:
            return math.nan, math.nan
        cos_mean = self.direction_cos_sum / self.direction_count
        sin_mean = self.direction_sin_sum / self.direction_count
        return cos_mean, sin_mean

    def direction_consistency(self) -> float:
        cos_mean, sin_mean = self.direction_means()
        if math.isnan(cos_mean) or math.isnan(sin_mean):
            return math.nan
        return math.sqrt(cos_mean**2 + sin_mean**2)

    def frac_high_extension(self) -> float:
        if self.during_count == 0:
            return math.nan
        return self.high_extension_count / self.during_count

    def first_second_mean(self) -> float:
        return self._mean(self.first_second_sum, self.first_second_count)

    def rise_speed(self) -> float:
        baseline_mean = self.baseline_mean()
        first_second = self.first_second_mean()
        if math.isnan(baseline_mean) or math.isnan(first_second):
            return math.nan
        return first_second - baseline_mean

def _downcast_numeric(frame: pd.DataFrame, *, float_dtype: str = "float32") -> pd.DataFrame:
    """Return a copy with floating-point columns safely downcast."""

    if frame.empty:
        return frame.copy()

    result = frame.copy()
    float_columns = result.select_dtypes(include=["float", "float16", "float32", "float64"]).columns
    for column in float_columns:
        coerced = pd.to_numeric(result[column], errors="coerce")
        if float_dtype:
            result[column] = coerced.astype(float_dtype)
        else:  # pragma: no cover - fallback guard
            result[column] = pd.to_numeric(coerced, downcast="float")
    return result


def _normalise_numeric(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    mode: str,
) -> pd.DataFrame:
    """Apply normalization to numeric columns based on the requested mode."""

    if not columns or mode == "none":
        return frame

    supported = {"none", "zscore", "minmax"}
    if mode not in supported:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    result = frame.copy()
    numeric_cols = [col for col in columns if col in result.columns]
    if not numeric_cols:
        return result

    numeric_df = result[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if mode == "zscore":
        means = numeric_df.mean()
        stds = numeric_df.std(ddof=0).replace(0, 1.0)
        transformed = (numeric_df - means) / stds
    elif mode == "minmax":
        minima = numeric_df.min()
        maxima = numeric_df.max()
        denom = (maxima - minima).replace(0, 1.0)
        transformed = (numeric_df - minima) / denom
    else:  # pragma: no cover - defensive default
        transformed = numeric_df

    transformed = transformed.astype("float32")
    for column in numeric_cols:
        # Assign as float32 to avoid pandas upcasting warnings when the
        # original column used an integer-backed dtype.
        result[column] = transformed[column].astype("float32")
    return result


def normalize_key_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonicalised merge keys.

    Ensures string-based identifiers (``dataset``, ``fly``, ``trial_type``,
    ``trial_label``) are trimmed strings and ``fly_number`` is stored as a
    nullable integer. Columns missing from ``frame`` are ignored so callers can
    reuse the helper across heterogeneous inputs.
    """

    result = frame.copy()
    for column in STRING_KEY_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype(str).str.strip()
            result.loc[result[column].isin({"", "nan", "None"}), column] = pd.NA
    if "fly_number" in result.columns:
        def _coerce(value: object) -> object:
            if pd.isna(value) or value == "":
                return pd.NA
            try:
                return int(value)
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise DataValidationError(
                    "fly_number must contain integer-compatible values."
                ) from exc

        result["fly_number"] = result["fly_number"].map(_coerce).astype("Int64")
    return result


def _ensure_columns(frame: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise DataValidationError(
            f"Missing required columns {missing} while processing {context}."
        )


def _load_trial_summary_csv(path: Path, *, logger: Logger) -> pd.DataFrame:
    """Load and validate a geometry trial summary CSV."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Geometry trial summary not found: {csv_path}")
    logger.info("Merging geometry trial summary from %s", csv_path)
    summary = pd.read_csv(csv_path)
    _ensure_columns(summary, TRIAL_SUMMARY_REQUIRED_COLUMNS, context=f"geometry trial summary {csv_path}")
    summary = normalize_key_columns(summary)
    dup_mask = summary.duplicated(subset=MERGE_KEYS, keep=False)
    if dup_mask.any():
        dup_rows = summary.loc[dup_mask, MERGE_KEYS].drop_duplicates()
        raise DataValidationError(
            "Geometry trial summary contains duplicate merge keys; ensure one row per trial."
            f" Offending keys: {dup_rows.to_dict(orient='records')}"
        )
    for column in summary.columns:
        if column in MERGE_KEYS:
            continue
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    return summary


def _merge_trial_summary(
    aggregates: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    logger: Logger,
) -> pd.DataFrame:
    """Merge external per-trial geometry summaries into streamed aggregates."""

    base_keys = {
        tuple(row)
        for row in aggregates.loc[:, list(MERGE_KEYS)].itertuples(index=False, name=None)
    }
    summary_keys = {
        tuple(row)
        for row in summary.loc[:, list(MERGE_KEYS)].itertuples(index=False, name=None)
    }
    missing_keys = base_keys - summary_keys
    extra_keys = summary_keys - base_keys
    if missing_keys:
        logger.warning(
            "Geometry trial summary missing %d of %d aggregated trial(s); new columns will contain NaN for those rows.",
            len(missing_keys),
            len(base_keys),
        )
    if extra_keys:
        logger.info(
            "Geometry trial summary includes %d additional trial(s) not present in streamed aggregates; they were ignored.",
            len(extra_keys),
        )
    overlapping = [
        column
        for column in summary.columns
        if column in aggregates.columns and column not in MERGE_KEYS
    ]
    if overlapping:
        for column in overlapping:
            if column not in TRIAL_SUMMARY_DUPLICATE_COLUMNS:
                logger.warning(
                    "Geometry trial summary column '%s' already exists; streamed values will be retained.",
                    column,
                )
            joined = aggregates.loc[:, [*MERGE_KEYS, column]].merge(
                summary.loc[:, [*MERGE_KEYS, column]],
                on=MERGE_KEYS,
                how="left",
                suffixes=("_stream", "_summary"),
                validate="one_to_one",
            )
            stream_vals = pd.to_numeric(joined[f"{column}_stream"], errors="coerce")
            summary_vals = pd.to_numeric(joined[f"{column}_summary"], errors="coerce")
            fill_mask = stream_vals.isna() & summary_vals.notna()
            if fill_mask.any():
                stream_vals = stream_vals.where(~fill_mask, summary_vals)
                aggregates[column] = stream_vals.to_numpy(dtype=float)
            stream_arr = stream_vals.to_numpy(dtype=float)
            summary_arr = summary_vals.to_numpy(dtype=float)
            valid = (~np.isnan(stream_arr)) & (~np.isnan(summary_arr))
            mismatch_count = 0
            if valid.any():
                diff_mask = ~np.isclose(stream_arr[valid], summary_arr[valid], rtol=1e-3, atol=1e-3)
                mismatch_count += int(diff_mask.sum())
            nan_mask = np.isnan(stream_arr) ^ np.isnan(summary_arr)
            mismatch_count += int(nan_mask.sum())
            if mismatch_count:
                logger.warning(
                    "Geometry trial summary column '%s' differs from streamed aggregates for %d trial(s); keeping streamed values.",
                    column,
                    mismatch_count,
                )
        summary = summary.drop(columns=overlapping)
    additional_columns = [col for col in summary.columns if col not in MERGE_KEYS]
    if not additional_columns:
        logger.info("Geometry trial summary did not contribute additional columns after alignment.")
        return aggregates
    merged = aggregates.merge(
        summary,
        on=MERGE_KEYS,
        how="left",
        validate="one_to_one",
    )
    return merged



def _as_key_tuple(key: object) -> tuple[object, ...]:
    if isinstance(key, tuple):
        return key
    return (key,)


def _validate_block_contiguity(
    chunk: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    frame_column: str,
    last_seen: MutableMapping[tuple[object, ...], int],
) -> None:
    if frame_column not in chunk.columns:
        return
    if chunk[frame_column].isna().any():
        raise DataValidationError(
            f"Frame column '{frame_column}' contains NaN values during streaming."
        )
    ordered = chunk.sort_values(list(key_columns) + [frame_column])
    for key, group in ordered.groupby(list(key_columns), sort=False):
        values = group[frame_column].astype(int).to_numpy()
        if len(values) == 0:
            continue
        diffs = pd.Series(values).diff().dropna()
        if not diffs.eq(1).all():
            raise DataValidationError(
                f"Frame column '{frame_column}' must increase by 1 within each block."
            )
        tuple_key = _as_key_tuple(key)
        previous = last_seen.get(tuple_key)
        if previous is not None and values[0] != previous + 1:
            raise DataValidationError(
                f"Detected non-contiguous frames for key {tuple_key}."
            )
        last_seen[tuple_key] = int(values[-1])


def _prepare_labels_table(labels_csv: Path, join_keys: Sequence[str]) -> pd.DataFrame:
    labels = _load_csv(labels_csv)
    _ensure_columns(labels, join_keys, context=str(labels_csv))
    labels = normalize_key_columns(labels)
    duplicated = labels.duplicated(subset=list(join_keys), keep=False)
    if duplicated.any():
        raise DataValidationError(
            "Labels CSV contains duplicate keys: "
            f"{labels.loc[duplicated, list(join_keys)].drop_duplicates().to_dict(orient='records')}"
        )
    return labels


def _merge_labels(
    chunk: pd.DataFrame,
    labels: pd.DataFrame,
    join_keys: Sequence[str],
    *,
    drop_missing: bool,
) -> pd.DataFrame:
    _ensure_columns(chunk, join_keys, context="geometry frame chunk")
    extras = [col for col in labels.columns if col not in join_keys and col not in chunk.columns]
    selected = labels[list(dict.fromkeys([*join_keys, *extras]))]
    merged = chunk.merge(
        selected,
        on=list(join_keys),
        how="left",
        validate="many_to_one",
        sort=False,
        indicator=True,
    )
    missing_mask = merged["_merge"] == "left_only"
    if missing_mask.any():
        missing_rows = merged.loc[missing_mask, list(join_keys)].drop_duplicates()
        if drop_missing:
            merged = merged.loc[~missing_mask].copy()
        else:
            raise DataValidationError(
                "Missing labels for keys "
                f"{missing_rows.to_dict(orient='records')}"
            )
    merged = merged.drop(columns="_merge")
    return merged


def _iter_parquet_chunks(
    path: Path,
    *,
    chunk_size: int,
    columns: Sequence[str] | None,
) -> Iterator[pd.DataFrame]:
    frame = pd.read_parquet(path, columns=list(columns) if columns else None)
    total = len(frame)
    if total == 0:
        yield frame
        return
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        yield frame.iloc[start:end].copy()
        start = end


def _compute_chunk_stats(
    chunk: pd.DataFrame,
    key_columns: Sequence[str],
    frame_column: str,
) -> dict[str, object]:
    stats: dict[str, object] = {
        "rows": int(len(chunk)),
        "unique_blocks": int(chunk[key_columns].drop_duplicates().shape[0])
        if key_columns and not chunk.empty
        else 0,
    }
    if frame_column in chunk.columns and not chunk.empty:
        stats["frame_min"] = int(chunk[frame_column].min())
        stats["frame_max"] = int(chunk[frame_column].max())
    return stats


def _resolve_column_order(
    *,
    chunk: pd.DataFrame,
    requested: Sequence[str] | None,
    labels: pd.DataFrame | None,
) -> List[str]:
    base_order = list(chunk.columns)
    if requested:
        requested_set = {name for name in requested if name in chunk.columns}
        base_order = [name for name in requested if name in requested_set]
        base_order.extend([col for col in chunk.columns if col not in requested_set])
    if labels is not None:
        for column in labels.columns:
            if column in base_order:
                continue
            base_order.append(column)
    return base_order


def _open_parquet_writer(
    path: Path,
    chunk: pd.DataFrame,
    *,
    compression: str,
):
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise DataValidationError(
            "pyarrow is required for parquet caching. Install flybehavior-response with the optional parquet extras."
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    writer = pq.ParquetWriter(str(path), table.schema, compression=compression)
    return writer, pa


def _enrich_epoch_columns(
    chunk: pd.DataFrame,
    *,
    frame_column: str | None,
    config: BehavioralFeatureConfig,
) -> pd.DataFrame:
    """Add baseline/during/post epoch flags when odor indices are available."""

    if frame_column is None or frame_column not in chunk.columns:
        return chunk
    if config.odor_on_column not in chunk.columns or config.odor_off_column not in chunk.columns:
        return chunk

    frames = pd.to_numeric(chunk[frame_column], errors="coerce")
    odor_on = pd.to_numeric(chunk[config.odor_on_column], errors="coerce")
    odor_off = pd.to_numeric(chunk[config.odor_off_column], errors="coerce")
    valid = frames.notna() & odor_on.notna() & odor_off.notna()
    if not valid.any():
        chunk[config.is_before_column] = 0
        chunk[config.is_during_column] = 0
        chunk[config.is_after_column] = 0
        return chunk

    is_before = pd.Series(0, index=chunk.index, dtype="uint8")
    is_during = pd.Series(0, index=chunk.index, dtype="uint8")
    is_after = pd.Series(0, index=chunk.index, dtype="uint8")

    before_mask = valid & (frames < odor_on)
    during_mask = valid & (frames >= odor_on) & (frames <= odor_off)
    after_mask = valid & (frames > odor_off)

    is_before.loc[before_mask] = 1
    is_during.loc[during_mask] = 1
    is_after.loc[after_mask] = 1

    chunk[config.is_before_column] = is_before
    chunk[config.is_during_column] = is_during
    chunk[config.is_after_column] = is_after
    return chunk


def load_geom_frames(
    source: Path,
    *,
    chunk_size: int = 100_000,
    columns: Sequence[str] | None = None,
    schema: Mapping[str, str] | None = None,
    logger: Logger | None = None,
    logger_name: str | None = None,
    cache_parquet: Path | None = None,
    use_cache: bool = False,
    compression: str = "zstd",
    labels_csv: Path | None = None,
    join_keys: Sequence[str] = MERGE_KEYS,
    frame_column: str = "frame_idx",
    drop_missing_labels: bool = True,
    behavioral_config: BehavioralFeatureConfig | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream geometry frames with optional parquet caching and label joins.

    Parameters
    ----------
    source:
        CSV or parquet file containing per-frame coordinates.
    chunk_size:
        Number of rows per streamed chunk when reading CSVs or cached parquet.
    columns:
        Optional ordered subset of columns to retain. Merge keys and ``frame_column``
        are automatically appended if omitted.
    schema:
        Optional dtype mapping forwarded to :func:`pandas.read_csv`.
    logger:
        Optional logger for status updates. When ``None`` a module-level logger is
        created using ``logger_name``.
    cache_parquet:
        Destination for cached parquet output. When ``use_cache`` is set and the
        path exists, the cached data is loaded instead of the source file.
    use_cache:
        If ``True`` and ``cache_parquet`` exists, skip parsing the source file and
        stream directly from the cached parquet.
    compression:
        Compression codec applied when writing parquet caches. ``"zstd"`` provides
        a good balance between size and speed.
    labels_csv:
        Optional labels CSV providing trial-level metadata. The merge enforces
        uniqueness via ``join_keys`` and raises :class:`DataValidationError` when
        duplicates or missing keys are encountered.
    join_keys:
        Key columns that identify each trial or recording block.
    frame_column:
        Column storing the per-frame index. When present the function validates
        contiguity across streamed chunks.
    drop_missing_labels:
        When ``True`` (the default) rows without matching labels are removed prior
        to streaming so only labelled trials are processed. When ``False`` the
        loader raises :class:`DataValidationError` if any geometry rows are
        missing labels.

    Yields
    ------
    pandas.DataFrame
        Processed chunk preserving column order.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    source = Path(source)
    log = logger or get_logger(logger_name or __name__)
    config = behavioral_config or BehavioralFeatureConfig()

    if use_cache and not cache_parquet:
        raise ValueError("use_cache=True requires --cache-parquet to be specified")

    label_table: pd.DataFrame | None = None
    label_projection: pd.DataFrame | None = None
    label_key_set: set[tuple[object, ...]] | None = None
    if labels_csv is not None:
        label_table = _prepare_labels_table(labels_csv, join_keys)
        extra_columns = [col for col in label_table.columns if col not in join_keys]
        label_projection = pd.DataFrame(columns=extra_columns)
        if drop_missing_labels:
            label_key_set = {
                tuple(row)
                for row in label_table.loc[:, list(join_keys)].itertuples(index=False, name=None)
            }

    requested_columns: List[str] | None = None
    if columns is not None:
        requested_columns = list(columns)
    if requested_columns is not None:
        for key in join_keys:
            if key not in requested_columns:
                requested_columns.append(key)
        if frame_column and frame_column not in requested_columns:
            requested_columns.append(frame_column)

    resolved_frame_column = frame_column if frame_column else None
    frame_column_present = bool(resolved_frame_column)
    available_columns: List[str] | None = None
    geometry_stream: GeometryFrameStream | None = None

    def _iter_source() -> Iterator[pd.DataFrame]:
        nonlocal frame_column_present, resolved_frame_column, available_columns, requested_columns, geometry_stream
        if use_cache and cache_parquet and cache_parquet.exists():
            log.info("Loading geometry frames from cache: %s", cache_parquet)
            yield from _iter_parquet_chunks(
                cache_parquet,
                chunk_size=chunk_size,
                columns=requested_columns,
            )
            return

        if not source.exists():
            raise FileNotFoundError(f"Geometry input not found: {source}")

        if use_cache and cache_parquet:
            log.info(
                "Cache parquet %s not found; streaming source %s instead.",
                cache_parquet,
                source,
            )

        suffix = source.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            yield from _iter_parquet_chunks(
                source,
                chunk_size=chunk_size,
                columns=requested_columns,
            )
            return

        header = pd.read_csv(source, nrows=0)
        available_columns = list(header.columns)
        missing_keys = [col for col in join_keys if col not in available_columns]
        if missing_keys:
            raise DataValidationError(
                "Geometry source is missing required key columns "
                f"{missing_keys}. Provide a CSV containing the full identifier set."
            )

        if resolved_frame_column and resolved_frame_column not in available_columns:
            alias = next(
                (candidate for candidate in FRAME_COLUMN_ALIASES if candidate in available_columns),
                None,
            )
            if alias:
                log.info(
                    "Frame column '%s' not found in %s; using '%s' for contiguity validation.",
                    resolved_frame_column,
                    source,
                    alias,
                )
                resolved_frame_column = alias
                frame_column_present = True
                if geometry_stream is not None:
                    geometry_stream.frame_column = resolved_frame_column
                if requested_columns is not None and frame_column in requested_columns:
                    requested_columns = [
                        col for col in requested_columns if col != frame_column
                    ]
                    if alias not in requested_columns:
                        requested_columns.append(alias)
            else:
                frame_column_present = False
                log.info(
                    "Frame column '%s' not detected in %s; skipping contiguity validation.",
                    resolved_frame_column,
                    source,
                )
                if geometry_stream is not None:
                    geometry_stream.frame_column = None

        if requested_columns:
            deduped_requested = list(dict.fromkeys(requested_columns))
            missing_requested = [
                name for name in deduped_requested if name not in available_columns
            ]
            missing_required = [
                name for name in missing_requested if name in join_keys
            ]
            missing_optional = [
                name
                for name in missing_requested
                if name not in join_keys and name != frame_column
            ]
            if missing_required:
                raise DataValidationError(
                    "Geometry source is missing required columns "
                    f"{missing_required}."
                )
            if missing_optional:
                log.info(
                    "Skipping %s absent columns from geometry stream: %s",
                    len(missing_optional),
                    ", ".join(sorted(missing_optional)),
                )
            usecols = [name for name in deduped_requested if name in available_columns]
        else:
            usecols = None

        csv_iter = pd.read_csv(
            source,
            chunksize=chunk_size,
            dtype=schema,
            usecols=usecols,
            low_memory=False,
        )
        for chunk in csv_iter:
            yield chunk

    def _stream() -> Iterator[pd.DataFrame]:
        nonlocal available_columns, frame_column_present, resolved_frame_column, geometry_stream
        last_seen: MutableMapping[tuple[object, ...], int] = {}
        final_order: List[str] | None = None
        writer = None
        pa_module = None
        total_rows = 0
        chunk_counter = 0

        try:
            for chunk_counter, raw_chunk in enumerate(_iter_source(), start=1):
                if raw_chunk.empty:
                    continue
                if available_columns is None:
                    available_columns = list(raw_chunk.columns)
                    if geometry_stream is not None:
                        geometry_stream.available_columns = available_columns
                    if (
                        resolved_frame_column
                        and resolved_frame_column not in available_columns
                        and frame_column_present
                    ):
                        alias = next(
                            (
                                candidate
                                for candidate in FRAME_COLUMN_ALIASES
                                if candidate in available_columns
                            ),
                            None,
                        )
                        if alias:
                            log.info(
                                "Frame column '%s' not found in streamed chunk; using '%s' instead.",
                                resolved_frame_column,
                                alias,
                            )
                            resolved_frame_column = alias
                            if geometry_stream is not None:
                                geometry_stream.frame_column = resolved_frame_column
                        else:
                            log.info(
                                "Frame column '%s' absent from streamed chunk; skipping contiguity validation.",
                                resolved_frame_column,
                            )
                            frame_column_present = False
                            if geometry_stream is not None:
                                geometry_stream.frame_column = None
                chunk = normalize_key_columns(raw_chunk)
                _ensure_columns(chunk, join_keys, context="geometry frame chunk")
                if label_key_set is not None:
                    key_mask = [
                        tuple(row) in label_key_set
                        for row in chunk.loc[:, list(join_keys)].itertuples(index=False, name=None)
                    ]
                    key_series = pd.Series(key_mask, index=chunk.index, dtype=bool)
                    kept = int(key_series.sum())
                    dropped = len(chunk) - kept
                    if dropped:
                        log.info(
                            "Dropping %d geometry rows without matching labels during streaming.",
                            dropped,
                        )
                    if kept == 0:
                        continue
                    chunk = chunk.loc[key_series].copy()
                if frame_column_present and resolved_frame_column:
                    _validate_block_contiguity(
                        chunk,
                        key_columns=join_keys,
                        frame_column=resolved_frame_column,
                        last_seen=last_seen,
                    )
                if label_table is not None:
                    chunk = _merge_labels(
                        chunk,
                        label_table,
                        join_keys,
                        drop_missing=drop_missing_labels,
                    )
                    if chunk.empty:
                        continue
                if final_order is None:
                    final_order = _resolve_column_order(
                        chunk=chunk,
                        requested=requested_columns,
                        labels=label_projection,
                    )
                missing_columns = [col for col in final_order if col not in chunk.columns]
                for name in missing_columns:
                    chunk[name] = pd.NA
                chunk = chunk.loc[:, final_order]

                chunk = _enrich_epoch_columns(
                    chunk,
                    frame_column=resolved_frame_column if frame_column_present else None,
                    config=config,
                )

                stats = _compute_chunk_stats(
                    chunk,
                    join_keys,
                    resolved_frame_column if resolved_frame_column else frame_column,
                )
                total_rows += int(stats.get("rows", 0))
                log.info(
                    "Chunk %03d | rows=%d | unique_blocks=%d | frame_range=%s",
                    chunk_counter,
                    stats.get("rows", 0),
                    stats.get("unique_blocks", 0),
                    (
                        f"{stats['frame_min']}->{stats['frame_max']}"
                        if "frame_min" in stats and "frame_max" in stats
                        else "n/a"
                    ),
                )

                if cache_parquet and not use_cache:
                    if writer is None:
                        if cache_parquet.exists():
                            cache_parquet.unlink()
                        writer, pa_module = _open_parquet_writer(
                            cache_parquet,
                            chunk,
                            compression=compression,
                        )
                    table = pa_module.Table.from_pandas(chunk, preserve_index=False)
                    writer.write_table(table)
                yield chunk
        finally:
            if writer is not None:
                writer.close()
            if chunk_counter == 0:
                log.info("Streamed 0 rows from %s", source)
            else:
                log.info(
                    "Completed geometry stream from %s | chunks=%d | rows=%d",
                    source,
                    chunk_counter,
                    total_rows,
                )

    geometry_stream = GeometryFrameStream(
        _stream(),
        frame_column=resolved_frame_column,
        available_columns=available_columns or [],
    )
    return geometry_stream


class TrialAggregateBuilder:
    """Incrementally aggregate streamed frames and optionally assemble traces."""

    def __init__(
        self,
        *,
        key_columns: Sequence[str],
        frame_column: str | None,
        value_columns: Sequence[str] | None,
        stats: Sequence[str],
        exclude_columns: Sequence[str] | None,
        trace_candidates: Mapping[str, str] | None = None,
        behavioral_config: "BehavioralFeatureConfig | None" = None,
    ) -> None:
        self.key_columns = list(key_columns)
        self.frame_column = frame_column
        self.value_columns_param = list(value_columns) if value_columns is not None else None
        self.exclude_set = {name for name in (exclude_columns or [])}
        self.stats_order = list(dict.fromkeys(stat.lower() for stat in stats))
        stats_set = set(self.stats_order)
        unsupported = stats_set - SUPPORTED_AGG_STATS
        if unsupported:
            raise ValueError(
                f"Unsupported aggregation stats requested: {sorted(unsupported)}. "
                f"Supported values are {sorted(SUPPORTED_AGG_STATS)}."
            )
        self.aggregation: dict[tuple[object, ...], dict[str, object]] = {}
        self.resolved_values: List[str] | None = None
        self.trace_candidates = dict(trace_candidates or {})
        self.trace_mapping: dict[str, str] | None = None
        self.trace_prefixes: List[str] = []
        self.trace_values: dict[tuple[object, ...], dict[str, List[float]]] = {}
        self.trace_next_frame: dict[tuple[object, ...], int] = {}
        self.trace_initialised = not bool(self.trace_candidates)
        self.sorted_keys: List[tuple[object, ...]] | None = None
        self.behavioral_config = behavioral_config
        self.behavioral_data: dict[tuple[object, ...], TrialBehaviorAccumulator] = {}

    @property
    def traces_enabled(self) -> bool:
        return bool(self.trace_mapping)

    def _initialise_traces(self, chunk: pd.DataFrame) -> None:
        if self.trace_initialised:
            return
        missing = [col for col in self.trace_candidates if col not in chunk.columns]
        if missing:
            self.trace_initialised = True
            return
        if not self.frame_column or self.frame_column not in chunk.columns:
            raise DataValidationError(
                "Trace generation requires a frame column; provide --geom-frame-column or include the column in the stream."
            )
        self.trace_mapping = dict(self.trace_candidates)
        self.trace_prefixes = list(dict.fromkeys(self.trace_mapping.values()))
        self.trace_initialised = True

    def _resolve_value_columns(self, chunk: pd.DataFrame) -> None:
        if self.resolved_values is not None:
            return
        if self.value_columns_param is None:
            self.resolved_values = [
                col
                for col in chunk.columns
                if col not in self.key_columns
                and col != self.frame_column
                and is_numeric_dtype(chunk[col])
                and col not in self.exclude_set
            ]
        else:
            missing = [col for col in self.value_columns_param if col not in chunk.columns]
            if missing:
                raise DataValidationError(
                    f"Aggregation requested missing columns: {missing}"
                )
            non_numeric = [
                col
                for col in self.value_columns_param
                if col not in self.exclude_set and not is_numeric_dtype(chunk[col])
            ]
            if non_numeric:
                raise DataValidationError(
                    f"Aggregation columns must be numeric; offending columns: {non_numeric}"
                )
            self.resolved_values = [
                col for col in self.value_columns_param if col not in self.exclude_set
            ]

    def _accumulate_trace(
        self,
        key: tuple[object, ...],
        group: pd.DataFrame,
    ) -> None:
        if not self.trace_mapping or not self.frame_column:
            return
        ordered = group.sort_values(self.frame_column)
        frames = ordered[self.frame_column].dropna().astype(int)
        if frames.empty:
            return
        expected = self.trace_next_frame.get(key, 0)
        first_frame = int(frames.iloc[0])
        if first_frame != expected:
            raise DataValidationError(
                f"Trace assembly for key {key} expected frame {expected} but received {first_frame}."
            )
        diffs = frames.diff().dropna()
        if not diffs.eq(1).all():
            raise DataValidationError(
                f"Frame column '{self.frame_column}' must increase by 1 within each trace block."
            )
        entry = self.trace_values.setdefault(
            key, {prefix: [] for prefix in self.trace_prefixes}
        )
        for prefix in self.trace_prefixes:
            entry.setdefault(prefix, [])
        for column, prefix in self.trace_mapping.items():
            values = ordered[column].astype(float).tolist()
            entry[prefix].extend(values)
        self.trace_next_frame[key] = expected + len(frames)

    def process_chunk(self, chunk: pd.DataFrame) -> None:
        if chunk.empty:
            return
        _ensure_columns(chunk, self.key_columns, context="aggregation chunk")
        chunk = normalize_key_columns(chunk)
        self._initialise_traces(chunk)
        self._resolve_value_columns(chunk)

        if self.resolved_values is None:
            self.resolved_values = []

        missing = [col for col in self.resolved_values if col not in chunk.columns]
        if missing:
            raise DataValidationError(
                f"Stream chunk missing expected columns: {missing}"
            )

        grouped = chunk.groupby(list(self.key_columns), sort=False, dropna=False)
        for key, group in grouped:
            tuple_key = _as_key_tuple(key)
            entry = self.aggregation.setdefault(
                tuple_key,
                {
                    "count": 0,
                    "frame_min": None,
                    "frame_max": None,
                    "values": defaultdict(dict),
                },
            )
            entry["count"] = int(entry.get("count", 0)) + len(group)

            if self.frame_column and self.frame_column in group.columns:
                frame_vals = group[self.frame_column].dropna().astype(int)
                if not frame_vals.empty:
                    current_min = entry.get("frame_min")
                    new_min = int(frame_vals.min())
                    entry["frame_min"] = (
                        new_min if current_min is None else min(int(current_min), new_min)
                    )
                    current_max = entry.get("frame_max")
                    new_max = int(frame_vals.max())
                    entry["frame_max"] = (
                        new_max if current_max is None else max(int(current_max), new_max)
                    )

            for column in self.resolved_values:
                series = group[column].dropna()
                if series.empty:
                    continue
                numeric = series.astype(float)
                value_entry = entry["values"].setdefault(
                    column,
                    {
                        "count": 0,
                        "sum": 0.0,
                        "sum_sq": 0.0,
                        "min": None,
                        "max": None,
                        "first": None,
                        "last": None,
                    },
                )
                count = int(value_entry["count"]) + len(numeric)
                value_entry["count"] = count
                value_entry["sum"] = float(value_entry["sum"]) + float(numeric.sum())
                value_entry["sum_sq"] = float(value_entry["sum_sq"]) + float(
                    (numeric ** 2).sum()
                )
                min_val = float(numeric.min())
                max_val = float(numeric.max())
                value_entry["min"] = (
                    min_val
                    if value_entry["min"] is None
                    else min(float(value_entry["min"]), min_val)
                )
                value_entry["max"] = (
                    max_val
                    if value_entry["max"] is None
                    else max(float(value_entry["max"]), max_val)
                )
                if value_entry["first"] is None:
                    value_entry["first"] = float(numeric.iloc[0])
                value_entry["last"] = float(numeric.iloc[-1])

            if self.trace_mapping:
                self._accumulate_trace(tuple_key, group)
            if self.behavioral_config is not None:
                accumulator = self.behavioral_data.setdefault(
                    tuple_key, TrialBehaviorAccumulator()
                )
                accumulator.update(
                    group,
                    config=self.behavioral_config,
                    frame_column=self.frame_column,
                )

    def build_dataframe(self) -> pd.DataFrame:
        if self.resolved_values is None:
            self.resolved_values = list(self.value_columns_param or [])

        rows: List[dict[str, object]] = []
        for key_tuple, entry in self.aggregation.items():
            row = {col: key_tuple[idx] for idx, col in enumerate(self.key_columns)}
            row["frame_count"] = entry.get("count", 0)
            if entry.get("frame_min") is not None:
                row["frame_start"] = entry["frame_min"]
            if entry.get("frame_max") is not None:
                row["frame_end"] = entry["frame_max"]

            values_map = entry.get("values", {})
            for column in self.resolved_values:
                stats_entry = values_map.get(column)
                if not stats_entry:
                    continue
                count = int(stats_entry.get("count", 0))
                for stat in self.stats_order:
                    if stat == "mean":
                        row[f"{column}_mean"] = (
                            stats_entry["sum"] / count if count else math.nan
                        )
                    elif stat == "std":
                        if count > 1:
                            mean = stats_entry["sum"] / count
                            variance = max(stats_entry["sum_sq"] / count - mean**2, 0.0)
                            row[f"{column}_std"] = math.sqrt(
                                variance * count / (count - 1)
                            )
                        else:
                            row[f"{column}_std"] = math.nan
                    elif stat == "min":
                        row[f"{column}_min"] = stats_entry.get("min")
                    elif stat == "max":
                        row[f"{column}_max"] = stats_entry.get("max")
                    elif stat == "sum":
                        row[f"{column}_sum"] = stats_entry.get("sum")
                    elif stat == "first":
                        row[f"{column}_first"] = stats_entry.get("first")
                    elif stat == "last":
                        row[f"{column}_last"] = stats_entry.get("last")
            if self.behavioral_config is not None:
                metrics = self.behavioral_data.get(key_tuple)
                if metrics is not None:
                    before_mean = metrics.baseline_mean()
                    during_mean = metrics.during_mean()
                    cos_mean, sin_mean = metrics.direction_means()
                    row["r_before_mean"] = before_mean
                    row["r_before_std"] = metrics.baseline_std()
                    row["r_during_mean"] = during_mean
                    row["r_during_std"] = metrics.during_std()
                    row["r_during_minus_before_mean"] = (
                        during_mean - before_mean
                        if not math.isnan(during_mean) and not math.isnan(before_mean)
                        else math.nan
                    )
                    row["cos_theta_during_mean"] = cos_mean
                    row["sin_theta_during_mean"] = sin_mean
                    row["direction_consistency"] = metrics.direction_consistency()
                    row["frac_high_ext_during"] = metrics.frac_high_extension()
                    row["rise_speed"] = metrics.rise_speed()
            rows.append(row)

        if not rows:
            base_columns = list(self.key_columns)
            if self.frame_column:
                base_columns.extend(["frame_count", "frame_start", "frame_end"])
            for column in self.resolved_values or []:
                for stat in self.stats_order:
                    base_columns.append(f"{column}_{stat}")
            if self.behavioral_config is not None:
                base_columns.extend(
                    [
                        "r_before_mean",
                        "r_before_std",
                        "r_during_mean",
                        "r_during_std",
                        "r_during_minus_before_mean",
                        "cos_theta_during_mean",
                        "sin_theta_during_mean",
                        "direction_consistency",
                        "frac_high_ext_during",
                        "rise_speed",
                    ]
                )
            self.sorted_keys = []
            return pd.DataFrame(columns=base_columns)

        result = pd.DataFrame(rows)
        result = result.sort_values(list(self.key_columns)).reset_index(drop=True)
        self.sorted_keys = [
            _as_key_tuple(key)
            for key in result.loc[:, list(self.key_columns)].itertuples(index=False, name=None)
        ]
        return result

    def build_trace_frame(self) -> tuple[pd.DataFrame | None, List[str]]:
        if not self.trace_mapping:
            return None, []
        if not self.trace_values:
            columns = [*self.key_columns]
            for prefix in self.trace_prefixes:
                columns.append(f"{prefix}0")
            return pd.DataFrame(columns=columns), []

        lengths = {
            key: self.trace_next_frame.get(key, 0)
            for key in self.trace_values
        }
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise DataValidationError(
                "Inconsistent trace lengths detected across trials; ensure each recording has the same frame count."
            )
        frame_count = unique_lengths.pop() if unique_lengths else 0
        trace_columns: List[str] = []
        for prefix in self.trace_prefixes:
            for idx in range(frame_count):
                trace_columns.append(f"{prefix}{idx}")

        if self.sorted_keys is None:
            self.sorted_keys = sorted(self.trace_values.keys())

        rows: List[dict[str, object]] = []
        for key_tuple in self.sorted_keys:
            entry = self.trace_values.get(key_tuple)
            if entry is None:
                raise DataValidationError(
                    f"Missing trace series for key {key_tuple}; verify geometry input completeness."
                )
            row = {col: key_tuple[idx] for idx, col in enumerate(self.key_columns)}
            for prefix in self.trace_prefixes:
                values = entry.get(prefix, [])
                if len(values) != frame_count:
                    raise DataValidationError(
                        f"Trace series for key {key_tuple} prefix '{prefix}' has {len(values)} frames; expected {frame_count}."
                    )
                for idx, value in enumerate(values):
                    row[f"{prefix}{idx}"] = float(value)
            rows.append(row)

        trace_df = pd.DataFrame(rows, columns=[*self.key_columns, *trace_columns])
        return trace_df, trace_columns


def aggregate_trials(
    frames: Iterable[pd.DataFrame],
    *,
    key_columns: Sequence[str] = MERGE_KEYS,
    frame_column: str = "frame_idx",
    value_columns: Sequence[str] | None = None,
    stats: Sequence[str] = ("mean", "min", "max"),
    exclude_columns: Sequence[str] | None = None,
    behavioral_config: BehavioralFeatureConfig | None = None,
) -> pd.DataFrame:
    """Aggregate streamed geometry frames into trial-level summaries."""

    builder = TrialAggregateBuilder(
        key_columns=key_columns,
        frame_column=frame_column,
        value_columns=value_columns,
        stats=stats,
        exclude_columns=exclude_columns,
        behavioral_config=behavioral_config,
    )
    for chunk in frames:
        builder.process_chunk(chunk)
    return builder.build_dataframe()
@dataclass(slots=True)
class MergedDataset:
    """Container for merged dataset and metadata."""

    frame: pd.DataFrame
    trace_columns: List[str]
    feature_columns: List[str]
    label_intensity: pd.Series | None
    sample_weights: pd.Series | None
    trace_prefixes: List[str]
    granularity: str = "trial"
    normalization: str = "none"


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


def _filter_trace_columns(
    df: pd.DataFrame, prefixes: Sequence[str]
) -> tuple[List[str], pd.DataFrame, List[str]]:
    """Validate and return trace columns for the requested prefixes."""

    requested = list(prefixes)
    resolved_prefixes = list(requested)
    mapping: dict[str, List[str]] | None = None

    try:
        mapping = find_series_columns(df, requested)
    except ValueError:
        if requested == DEFAULT_TRACE_PREFIXES:
            try:
                mapping = find_series_columns(df, RAW_TRACE_PREFIXES)
            except ValueError:
                mapping = None
            else:
                resolved_prefixes = list(RAW_TRACE_PREFIXES)
        if mapping is None:
            if requested != DEFAULT_TRACE_PREFIXES:
                raise
            mapping = {}
            prefix = requested[0]
            matches: List[tuple[int, str]] = []
            for column in df.columns:
                match = TRACE_PATTERN.match(column)
                if match:
                    idx = int(match.group(1))
                    if TRACE_RANGE[0] <= idx <= TRACE_RANGE[1]:
                        matches.append((idx, column))
            if not matches:
                raise
            matches.sort(key=lambda pair: pair[0])
            mapping[prefix] = [name for _, name in matches]
            resolved_prefixes = list(requested)

    adjusted_df = df
    if resolved_prefixes == DEFAULT_TRACE_PREFIXES:
        allowed = TRACE_RANGE[1] - TRACE_RANGE[0] + 1
        extra_columns: List[str] = []
        for prefix, columns in mapping.items():
            if len(columns) > allowed:
                extra_columns.extend(columns[allowed:])
                mapping[prefix] = columns[:allowed]
        if extra_columns:
            adjusted_df = df.drop(columns=extra_columns)

    allowed_columns: List[str] = []
    for prefix in resolved_prefixes:
        allowed_columns.extend(mapping[prefix])
    allowed_set = set(allowed_columns)
    ordered = [col for col in adjusted_df.columns if col in allowed_set]
    return ordered, adjusted_df, resolved_prefixes


def validate_feature_columns(
    frame: pd.DataFrame, *, allow_empty: bool = False
) -> List[str]:
    available = [col for col in frame.columns if col in FEATURE_COLUMNS]
    if not available:
        if allow_empty:
            return []
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


def load_and_merge(
    data_csv: Path,
    labels_csv: Path,
    *,
    logger_name: str = __name__,
    trace_prefixes: Sequence[str] | None = None,
    include_trace_columns: bool = True,
) -> MergedDataset:
    """Load and merge data and labels CSVs."""
    logger = get_logger(logger_name)
    logger.info("Loading data CSV: %s", data_csv)
    data_df = _load_csv(data_csv)
    logger.debug("Data shape: %s", data_df.shape)

    logger.info("Loading labels CSV: %s", labels_csv)
    labels_df = _load_csv(labels_csv)
    logger.debug("Labels shape: %s", labels_df.shape)

    try:
        data_key_dtypes = {col: data_df[col].dtype.name for col in MERGE_KEYS}
    except KeyError:
        data_key_dtypes = {
            col: "missing"
            for col in MERGE_KEYS
            if col not in data_df.columns
        }
    try:
        label_key_dtypes = {col: labels_df[col].dtype.name for col in MERGE_KEYS}
    except KeyError:
        label_key_dtypes = {
            col: "missing"
            for col in MERGE_KEYS
            if col not in labels_df.columns
        }
    logger.debug(
        "Key column dtypes | data: %s | labels: %s",
        data_key_dtypes,
        label_key_dtypes,
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

    trace_cols: List[str]
    resolved_prefixes: List[str]
    if include_trace_columns:
        requested_prefixes = list(trace_prefixes or DEFAULT_TRACE_PREFIXES)
        try:
            trace_cols, data_df, resolved_prefixes = _filter_trace_columns(
                data_df, requested_prefixes
            )
        except ValueError as exc:
            raise DataValidationError(str(exc)) from exc
        if not trace_cols:
            raise DataValidationError(
                "No trace columns found. Expected columns matching prefixes: %s" % requested_prefixes
            )
        if resolved_prefixes == DEFAULT_TRACE_PREFIXES:
            dropped = [col for col in data_df.columns if TRACE_PATTERN.match(col) and col not in trace_cols]
            if dropped:
                data_df = data_df.drop(columns=dropped)
                logger.info("Dropped %d trace columns outside %s", len(dropped), TRACE_RANGE)
    else:
        trace_cols = []
        resolved_prefixes = []
        configured_prefixes = list(trace_prefixes or [])
        drop_prefixes = list(dict.fromkeys([
            *configured_prefixes,
            *DEFAULT_TRACE_PREFIXES,
            *RAW_TRACE_PREFIXES,
        ]))
        drop_columns = [
            col
            for col in data_df.columns
            if TRACE_PATTERN.match(col)
            or any(col.startswith(prefix) for prefix in drop_prefixes)
        ]
        if drop_columns:
            data_df = data_df.drop(columns=drop_columns)
            logger.info(
                "Excluded %d trace columns from dataset because raw traces were disabled.",
                len(drop_columns),
            )

    allow_empty_features = resolved_prefixes != DEFAULT_TRACE_PREFIXES
    if not allow_empty_features:
        # Legacy dir_val_ exports may omit engineered summaries entirely.
        if not any(col in data_df.columns for col in FEATURE_COLUMNS):
            allow_empty_features = True
            logger.info(
                "Detected dir_val_ traces without engineered features; proceeding with trace-only dataset."
            )
    feature_cols = validate_feature_columns(
        data_df, allow_empty=allow_empty_features
    )
    if not feature_cols and allow_empty_features:
        logger.info(
            "No engineered feature columns detected; continuing with trace-only dataset."
        )
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
        trace_prefixes=resolved_prefixes,
        granularity="trial",
        normalization="none",
    )


def write_parquet(dataset: MergedDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.frame.to_parquet(path, index=False)


def load_geometry_dataset(
    source: Path,
    *,
    labels_csv: Path | None,
    logger_name: str = __name__,
    chunk_size: int = 100_000,
    columns: Sequence[str] | None = None,
    schema: Mapping[str, str] | None = None,
    cache_parquet: Path | None = None,
    use_cache: bool = False,
    compression: str = "zstd",
    frame_column: str = "frame_idx",
    stats: Sequence[str] | None = None,
    granularity: str = "trial",
    normalization: str = "none",
    drop_missing_labels: bool = True,
    downcast: bool = True,
    trial_summary: Path | None = None,
    feature_columns: Sequence[str] | None = None,
    include_traces: bool = True,
) -> MergedDataset:
    """Load geometry frames and optionally aggregate into trial-level rows."""

    granularity_norm = granularity.lower()
    if granularity_norm not in {"trial", "frame"}:
        raise ValueError("granularity must be 'trial' or 'frame'")
    normalization_mode = normalization.lower()
    if normalization_mode not in {"none", "zscore", "minmax"}:
        raise ValueError("Unsupported normalization mode for geometry dataset")
    if trial_summary is not None and granularity_norm != "trial":
        raise ValueError("Geometry trial summaries require granularity='trial'")

    stat_sequence = list(stats or ("mean", "min", "max"))
    feature_filter = [col for col in feature_columns] if feature_columns is not None else None
    feature_filter_set = set(feature_filter) if feature_filter else None
    logger = get_logger(logger_name)
    behavioral_config = BehavioralFeatureConfig()

    label_table: pd.DataFrame | None = None
    intensity: pd.Series | None = None
    weights: pd.Series | None = None

    if labels_csv is not None:
        label_table = _prepare_labels_table(labels_csv, MERGE_KEYS)
        if LABEL_COLUMN not in label_table.columns:
            raise DataValidationError(
                f"Labels file {labels_csv} missing required label column '{LABEL_COLUMN}'."
            )
        coerced = _coerce_labels(label_table[LABEL_COLUMN], labels_csv)
        label_table[LABEL_INTENSITY_COLUMN] = coerced.astype(int)
        label_table[LABEL_COLUMN] = (coerced > 0).astype(int)

    stream = load_geom_frames(
        source,
        chunk_size=chunk_size,
        columns=columns,
        schema=schema,
        logger=logger,
        cache_parquet=cache_parquet,
        use_cache=use_cache,
        compression=compression,
        labels_csv=labels_csv if labels_csv is not None else None,
        join_keys=MERGE_KEYS,
        frame_column=frame_column,
        drop_missing_labels=drop_missing_labels if labels_csv is not None else False,
        behavioral_config=behavioral_config,
    )

    resolved_stream_frame_column = getattr(stream, "frame_column", frame_column)

    if not include_traces:
        logger.info(
            "Raw trace assembly disabled for geometry dataset; proceeding without trace columns."
        )

    if granularity_norm == "trial":
        builder = TrialAggregateBuilder(
            key_columns=MERGE_KEYS,
            frame_column=resolved_stream_frame_column,
            value_columns=None,
            stats=stat_sequence,
            exclude_columns=[
                LABEL_COLUMN,
                LABEL_INTENSITY_COLUMN,
                behavioral_config.is_before_column,
                behavioral_config.is_during_column,
                behavioral_config.is_after_column,
                behavioral_config.odor_on_column,
                behavioral_config.odor_off_column,
            ],
            trace_candidates=RAW_GEOM_TRACE_CANDIDATES if include_traces else {},
            behavioral_config=behavioral_config,
        )
        for chunk in stream:
            current_frame_column = getattr(stream, "frame_column", resolved_stream_frame_column)
            if current_frame_column != resolved_stream_frame_column:
                resolved_stream_frame_column = current_frame_column
            if current_frame_column != builder.frame_column:
                builder.frame_column = current_frame_column
            builder.process_chunk(chunk)
        aggregates = builder.build_dataframe()
        aggregates = normalize_key_columns(aggregates)
        if trial_summary is not None:
            summary_df = _load_trial_summary_csv(trial_summary, logger=logger)
            aggregates = _merge_trial_summary(aggregates, summary_df, logger=logger)
        trace_frame, trace_columns = builder.build_trace_frame()
        trace_prefixes = list(builder.trace_prefixes)
        if builder.traces_enabled:
            logger.info(
                "Assembled %d raw trace columns per trial from geometry stream.",
                len(trace_columns),
            )
        else:
            trace_prefixes = []
            trace_columns = []
        feature_columns = [col for col in aggregates.columns if col not in MERGE_KEYS]
        aggregates = _normalise_numeric(
            aggregates,
            feature_columns,
            mode=normalization_mode,
        )
        if downcast:
            aggregates = _downcast_numeric(aggregates)
        if label_table is not None:
            joined = aggregates.merge(
                label_table[[*MERGE_KEYS, LABEL_COLUMN, LABEL_INTENSITY_COLUMN]],
                on=MERGE_KEYS,
                how="inner" if drop_missing_labels else "left",
                validate="one_to_one",
            )
            if LABEL_INTENSITY_COLUMN in joined.columns and joined[LABEL_INTENSITY_COLUMN].isna().any():
                raise DataValidationError(
                    "Aggregated geometry rows missing labels after merge; ensure labels cover all keys."
                )
            intensity = joined[LABEL_INTENSITY_COLUMN].astype(int)
            joined[LABEL_INTENSITY_COLUMN] = intensity
            joined[LABEL_COLUMN] = (intensity > 0).astype(int)
            weights = _compute_sample_weights(intensity)
        else:
            joined = aggregates
            intensity = None
            weights = None

        ordered = joined.sort_values(list(MERGE_KEYS)).reset_index(drop=True)
        if trace_columns:
            ordered = ordered.merge(
                trace_frame,
                on=MERGE_KEYS,
                how="inner",
                validate="one_to_one",
            )
        if downcast and trace_columns:
            ordered = _downcast_numeric(ordered)
        feature_set = set(MERGE_KEYS)
        exclusion = feature_set | {LABEL_COLUMN, LABEL_INTENSITY_COLUMN} | set(trace_columns)
        available_features = [
            col
            for col in ordered.columns
            if col not in exclusion
        ]
        if feature_filter_set is not None:
            missing = sorted(feature_filter_set - set(available_features))
            if missing:
                raise DataValidationError(
                    "Requested geometry feature columns not present in aggregated dataset: "
                    f"{missing}"
                )
            feature_list = [col for col in available_features if col in feature_filter_set]
            if not feature_list:
                raise DataValidationError(
                    "No geometry feature columns remain after applying the requested filter."
                )
            logger.info(
                "Restricted geometry feature set to %d columns via --geom-feature-columns.",
                len(feature_list),
            )
        else:
            feature_list = available_features
        return MergedDataset(
            frame=ordered,
            trace_columns=list(trace_columns),
            feature_columns=feature_list,
            label_intensity=intensity,
            sample_weights=weights,
            trace_prefixes=trace_prefixes,
            granularity="trial",
            normalization=normalization_mode,
        )

    # Frame-level granularity
    collected: List[pd.DataFrame] = []
    for chunk in stream:
        if chunk.empty:
            continue
        chunk = normalize_key_columns(chunk)
        if downcast:
            chunk = _downcast_numeric(chunk)
        collected.append(chunk)

    if collected:
        combined = pd.concat(collected, ignore_index=True)
    else:
        frame_key = resolved_stream_frame_column if resolved_stream_frame_column else frame_column
        base_columns: List[str] = list(dict.fromkeys([*MERGE_KEYS, frame_key]))
        if columns:
            base_columns.extend(col for col in columns if col not in base_columns)
        if labels_csv is not None:
            base_columns.extend(
                col for col in [LABEL_COLUMN, LABEL_INTENSITY_COLUMN] if col not in base_columns
            )
        combined = pd.DataFrame(columns=base_columns)

    combined = normalize_key_columns(combined)
    geometry_columns = [
        col
        for col in combined.columns
        if col not in MERGE_KEYS
        and col not in {LABEL_COLUMN, LABEL_INTENSITY_COLUMN}
        and col != (resolved_stream_frame_column or frame_column)
    ]

    for column in geometry_columns:
        combined[column] = pd.to_numeric(combined[column], errors="coerce").astype(
            "float32"
        )

    combined = _normalise_numeric(combined, geometry_columns, mode=normalization_mode)
    if downcast:
        combined = _downcast_numeric(combined)

    if labels_csv is not None:
        if LABEL_COLUMN not in combined.columns:
            raise DataValidationError(
                "Geometry frames missing label column after merge; verify input files."
            )
        coerced_frame_labels = _coerce_labels(combined[LABEL_COLUMN], labels_csv)
        combined[LABEL_INTENSITY_COLUMN] = coerced_frame_labels.astype(int)
        combined[LABEL_COLUMN] = (coerced_frame_labels > 0).astype(int)
        intensity = combined[LABEL_INTENSITY_COLUMN]
        weights = _compute_sample_weights(intensity)
    else:
        intensity = None
        weights = None

    sort_keys = [
        col
        for col in [*MERGE_KEYS, resolved_stream_frame_column or frame_column]
        if col in combined.columns
    ]
    if sort_keys:
        combined = combined.sort_values(sort_keys).reset_index(drop=True)
    else:
        combined = combined.reset_index(drop=True)

    feature_columns = [
        col
        for col in geometry_columns
        if col != LABEL_COLUMN and col != LABEL_INTENSITY_COLUMN
    ]
    if feature_filter_set is not None:
        missing = sorted(feature_filter_set - set(feature_columns))
        if missing:
            raise DataValidationError(
                "Requested geometry feature columns not present in frame-level dataset: "
                f"{missing}"
            )
        feature_columns = [col for col in feature_columns if col in feature_filter_set]
        if not feature_columns:
            raise DataValidationError(
                "No geometry feature columns remain after applying the requested filter."
            )
        logger.info(
            "Restricted frame-level geometry feature set to %d columns via --geom-feature-columns.",
            len(feature_columns),
        )

    return MergedDataset(
        frame=combined,
        trace_columns=[],
        feature_columns=feature_columns,
        label_intensity=intensity,
        sample_weights=weights,
        trace_prefixes=[],
        granularity="frame",
        normalization=normalization_mode,
    )


def load_dataset(
    *,
    data_csv: Path | None,
    labels_csv: Path | None,
    logger_name: str = __name__,
    trace_prefixes: Sequence[str] | None = None,
    include_traces: bool = True,
    geometry_source: Path | None = None,
    geom_chunk_size: int = 100_000,
    geom_columns: Sequence[str] | None = None,
    geom_cache_parquet: Path | None = None,
    geom_use_cache: bool = False,
    geom_frame_column: str = "frame_idx",
    geom_stats: Sequence[str] | None = None,
    geom_granularity: str = "trial",
    geom_normalization: str = "none",
    geom_drop_missing_labels: bool = True,
    geom_downcast: bool = True,
    geom_trial_summary: Path | None = None,
    geom_feature_columns: Sequence[str] | None = None,
) -> MergedDataset:
    """Load a dataset from either merged CSV inputs or geometry frames."""

    if geometry_source is not None:
        return load_geometry_dataset(
            geometry_source,
            labels_csv=labels_csv,
            logger_name=logger_name,
            chunk_size=geom_chunk_size,
            columns=geom_columns,
            cache_parquet=geom_cache_parquet,
            use_cache=geom_use_cache,
            frame_column=geom_frame_column,
            stats=geom_stats,
            granularity=geom_granularity,
            normalization=geom_normalization,
            drop_missing_labels=geom_drop_missing_labels,
            downcast=geom_downcast,
            trial_summary=geom_trial_summary,
            feature_columns=geom_feature_columns,
            include_traces=include_traces,
        )

    if data_csv is None:
        raise ValueError("data_csv must be provided when geometry_source is not specified")
    if labels_csv is None:
        raise ValueError("labels_csv must be provided when using tabular CSV datasets")

    return load_and_merge(
        data_csv,
        labels_csv,
        logger_name=logger_name,
        trace_prefixes=trace_prefixes,
        include_trace_columns=include_traces,
    )


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

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

from .io_wide import find_series_columns
from .logging_utils import get_logger

MERGE_KEYS = ["dataset", "fly", "fly_number", "trial_type", "trial_label"]
LABEL_COLUMN = "user_score_odor"
LABEL_INTENSITY_COLUMN = "user_score_odor_intensity"
TRACE_PATTERN = re.compile(r"^dir_val_(\d+)$")
TRACE_RANGE = (0, 3600)
DEFAULT_TRACE_PREFIXES = ["dir_val_"]
RAW_TRACE_PREFIXES = ["eye_x_f", "eye_y_f", "prob_x_f", "prob_y_f"]
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


SUPPORTED_AGG_STATS = {"mean", "std", "min", "max", "sum", "first", "last"}


STRING_KEY_COLUMNS = ["dataset", "fly", "trial_type", "trial_label"]


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

    frame_column_present = True

    def _iter_source() -> Iterator[pd.DataFrame]:
        nonlocal frame_column_present
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

        if frame_column and frame_column not in available_columns:
            frame_column_present = False
            log.info(
                "Frame column '%s' not detected in %s; skipping contiguity validation.",
                frame_column,
                source,
            )

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
                if frame_column_present:
                    _validate_block_contiguity(
                        chunk,
                        key_columns=join_keys,
                        frame_column=frame_column,
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

                stats = _compute_chunk_stats(chunk, join_keys, frame_column)
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

    return _stream()


def aggregate_trials(
    frames: Iterable[pd.DataFrame],
    *,
    key_columns: Sequence[str] = MERGE_KEYS,
    frame_column: str = "frame_idx",
    value_columns: Sequence[str] | None = None,
    stats: Sequence[str] = ("mean", "min", "max"),
) -> pd.DataFrame:
    """Aggregate streamed geometry frames into trial-level summaries."""

    stats_order = list(dict.fromkeys(stat.lower() for stat in stats))
    stats_set = set(stats_order)
    unsupported = stats_set - SUPPORTED_AGG_STATS
    if unsupported:
        raise ValueError(
            f"Unsupported aggregation stats requested: {sorted(unsupported)}. "
            f"Supported values are {sorted(SUPPORTED_AGG_STATS)}."
        )

    aggregation: dict[tuple[object, ...], dict[str, object]] = {}
    resolved_values: List[str] | None = None

    for chunk in frames:
        if chunk.empty:
            continue
        _ensure_columns(chunk, key_columns, context="aggregation chunk")
        chunk = normalize_key_columns(chunk)

        if resolved_values is None:
            if value_columns is None:
                resolved_values = [
                    col
                    for col in chunk.columns
                    if col not in key_columns
                    and col != frame_column
                    and is_numeric_dtype(chunk[col])
                ]
            else:
                missing = [col for col in value_columns if col not in chunk.columns]
                if missing:
                    raise DataValidationError(
                        f"Aggregation requested missing columns: {missing}"
                    )
                non_numeric = [col for col in value_columns if not is_numeric_dtype(chunk[col])]
                if non_numeric:
                    raise DataValidationError(
                        f"Aggregation columns must be numeric; offending columns: {non_numeric}"
                    )
                resolved_values = list(value_columns)
        else:
            missing = [col for col in resolved_values if col not in chunk.columns]
            if missing:
                raise DataValidationError(
                    f"Stream chunk missing expected columns: {missing}"
                )

        grouped = chunk.groupby(list(key_columns), sort=False, dropna=False)
        for key, group in grouped:
            tuple_key = _as_key_tuple(key)
            entry = aggregation.setdefault(tuple_key, {
                "count": 0,
                "frame_min": None,
                "frame_max": None,
                "values": defaultdict(dict),
            })
            entry["count"] = int(entry.get("count", 0)) + len(group)

            if frame_column in group.columns and not group[frame_column].isna().all():
                frame_vals = group[frame_column].dropna().astype(int)
                if not frame_vals.empty:
                    current_min = entry.get("frame_min")
                    new_min = int(frame_vals.min())
                    entry["frame_min"] = (
                        new_min
                        if current_min is None
                        else min(int(current_min), new_min)
                    )
                    current_max = entry.get("frame_max")
                    new_max = int(frame_vals.max())
                    entry["frame_max"] = (
                        new_max
                        if current_max is None
                        else max(int(current_max), new_max)
                    )

            for column in resolved_values or []:
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

    if resolved_values is None:
        resolved_values = list(value_columns or [])

    rows: List[dict[str, object]] = []
    for key_tuple, entry in aggregation.items():
        row = {col: key_tuple[idx] for idx, col in enumerate(key_columns)}
        row["frame_count"] = entry.get("count", 0)
        if entry.get("frame_min") is not None:
            row["frame_start"] = entry["frame_min"]
        if entry.get("frame_max") is not None:
            row["frame_end"] = entry["frame_max"]

        values_map = entry.get("values", {})
        for column in resolved_values:
            stats_entry = values_map.get(column)
            if not stats_entry:
                continue
            count = int(stats_entry.get("count", 0))
            for stat in stats_order:
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
        rows.append(row)

    if not rows:
        base_columns = list(key_columns)
        if frame_column:
            base_columns.extend(["frame_count", "frame_start", "frame_end"])
        for column in resolved_values or []:
            for stat in stats_order:
                base_columns.append(f"{column}_{stat}")
        return pd.DataFrame(columns=base_columns)

    result = pd.DataFrame(rows)
    result = result.sort_values(list(key_columns)).reset_index(drop=True)
    return result
@dataclass(slots=True)
class MergedDataset:
    """Container for merged dataset and metadata."""

    frame: pd.DataFrame
    trace_columns: List[str]
    feature_columns: List[str]
    label_intensity: pd.Series
    sample_weights: pd.Series
    trace_prefixes: List[str]


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
) -> MergedDataset:
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

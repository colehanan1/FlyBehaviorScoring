"""Metric computation utilities for synthetic fly generation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .io import LABEL_COLUMN

WINDOW_BEFORE = slice(0, 1260)
WINDOW_DURING = slice(1260, 2460)
WINDOW_AFTER_START = 2460
FPS = 40

MetricDict = Dict[str, float]


@dataclass(frozen=True)
class MetricResult:
    """Container storing per-trial metric outputs."""

    auc_before: float
    auc_during: float
    auc_after: float
    ttpd: float
    peak_value: float
    threshold: float
    std_before: float


def detect_fly_column(frame: pd.DataFrame) -> str:
    """Return the column name to use for grouping by fly.

    Priority order follows the public datasets: ``dataset`` → ``fly`` →
    ``fly_number``. If none of those columns exist, fall back to any column whose
    name includes the word "fly". Raises ``ValueError`` when no suitable
    identifier is located.
    """

    for candidate in ("dataset", "fly", "fly_number"):
        if candidate in frame.columns:
            if frame[candidate].isna().all():
                continue
            return candidate
    for column in frame.columns:
        if "fly" in column.lower() and not frame[column].isna().all():
            return column
    raise ValueError("Unable to locate a fly identifier column in the dataset.")


def discover_trace_columns(frame: pd.DataFrame) -> List[str]:
    """Discover per-frame trace columns using the ``dir_val_`` convention."""

    matches: List[Tuple[int, str]] = []
    for column in frame.columns:
        if not column.startswith("dir_val_"):
            continue
        suffix = column[len("dir_val_") :]
        if suffix.isdigit():
            matches.append((int(suffix), column))
    if not matches:
        raise ValueError("No columns matching 'dir_val_<index>' were detected.")
    matches.sort(key=lambda item: item[0])
    return [name for _, name in matches]


def _window(values: np.ndarray, window: slice) -> np.ndarray:
    return values[window]


def _after_window(values: np.ndarray, start: int) -> np.ndarray:
    return values[start:]


def _compute_auc(values: np.ndarray, threshold: float) -> float:
    adjusted = np.clip(values - threshold, a_min=0.0, a_max=None)
    return float(np.nansum(adjusted))


def _global_peak(values: np.ndarray) -> Tuple[int, float]:
    if np.isnan(values).all():
        return -1, float("nan")
    idx = int(np.nanargmax(values))
    return idx, float(values[idx])


def compute_metrics_for_trace(
    trace: np.ndarray,
    *,
    threshold: float,
    missing_ttpd_policy: str = "zero",
) -> MetricResult:
    """Compute the metric suite for a single time-series."""

    if missing_ttpd_policy not in {"zero", "nan"}:
        raise ValueError("missing_ttpd_policy must be 'zero' or 'nan'")

    before = _window(trace, WINDOW_BEFORE)
    during = _window(trace, WINDOW_DURING)
    after = _after_window(trace, WINDOW_AFTER_START)

    auc_before = _compute_auc(before, threshold)
    auc_during = _compute_auc(during, threshold)
    auc_after = _compute_auc(after, threshold)

    peak_idx, peak_value = _global_peak(trace)
    if peak_idx >= WINDOW_DURING.start and peak_idx < WINDOW_DURING.stop:
        ttpd = (peak_idx - WINDOW_DURING.start) / FPS
        peak = peak_value
    else:
        peak = 0.0 if missing_ttpd_policy == "zero" else float("nan")
        ttpd = 0.0 if missing_ttpd_policy == "zero" else float("nan")

    std_before = float(np.nanstd(before, ddof=0))

    return MetricResult(
        auc_before=float(auc_before),
        auc_during=float(auc_during),
        auc_after=float(auc_after),
        ttpd=float(ttpd),
        peak_value=float(peak),
        threshold=float(threshold),
        std_before=std_before,
    )


def compute_thresholds(
    frame: pd.DataFrame,
    *,
    trace_columns: Sequence[str],
    fly_column: str,
) -> Tuple[pd.Series, pd.Series]:
    """Return per-fly mean-before and per-trial std-before statistics."""

    traces = frame[trace_columns].to_numpy(dtype=float)
    before = traces[:, WINDOW_BEFORE]
    trial_std = np.nanstd(before, axis=1, ddof=0)
    trial_std_series = pd.Series(trial_std, index=frame.index, name="std_before_trial")

    trial_mean = np.nanmean(before, axis=1)
    per_fly_mean = (
        pd.Series(trial_mean, index=frame.index)
        .groupby(frame[fly_column])
        .mean()
        .rename("mean_before_fly")
    )

    mean_before_aligned = frame[fly_column].map(per_fly_mean)
    mean_before_series = pd.Series(mean_before_aligned.to_numpy(), index=frame.index, name="mean_before_fly")
    return mean_before_series, trial_std_series


def compute_metrics(
    frame: pd.DataFrame,
    *,
    trace_columns: Sequence[str],
    fly_column: str,
    missing_ttpd_policy: str = "zero",
) -> pd.DataFrame:
    """Compute metrics and return a dataframe aligned with ``frame``."""

    mean_before, std_before = compute_thresholds(
        frame, trace_columns=trace_columns, fly_column=fly_column
    )
    thresholds = mean_before + 4.0 * std_before

    traces = frame[trace_columns].to_numpy(dtype=float)
    results: Dict[str, List[float]] = {
        "AUC-Before": [],
        "AUC-During": [],
        "AUC-After": [],
        "TimeToPeak-During": [],
        "Peak-Value": [],
        "threshold_trial": [],
        "std_before_trial": [],
        "mean_before_fly": [],
    }

    for idx, trace in enumerate(traces):
        metric = compute_metrics_for_trace(
            trace,
            threshold=float(thresholds.iat[idx]),
            missing_ttpd_policy=missing_ttpd_policy,
        )
        results["AUC-Before"].append(metric.auc_before)
        results["AUC-During"].append(metric.auc_during)
        results["AUC-After"].append(metric.auc_after)
        results["TimeToPeak-During"].append(metric.ttpd)
        results["Peak-Value"].append(metric.peak_value)
        results["threshold_trial"].append(metric.threshold)
        results["std_before_trial"].append(metric.std_before)
        results["mean_before_fly"].append(float(mean_before.iat[idx]))

    return pd.DataFrame(results, index=frame.index)


def validate_metric_parity(
    frame: pd.DataFrame,
    *,
    trace_columns: Sequence[str],
    fly_column: str,
    logger: logging.Logger | None = None,
    missing_ttpd_policy: str = "zero",
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Recompute metrics and log discrepancies above ``tolerance``."""

    recomputed = compute_metrics(
        frame,
        trace_columns=trace_columns,
        fly_column=fly_column,
        missing_ttpd_policy=missing_ttpd_policy,
    )
    columns_to_check = [
        "AUC-Before",
        "AUC-During",
        "AUC-After",
        "TimeToPeak-During",
        "Peak-Value",
    ]
    discrepancies: Dict[str, float] = {}
    for column in columns_to_check:
        if column not in frame.columns:
            continue
        delta = np.abs(frame[column].to_numpy(dtype=float) - recomputed[column].to_numpy(dtype=float))
        max_delta = float(np.nanmax(delta)) if delta.size else 0.0
        if max_delta > tolerance:
            discrepancies[column] = max_delta
    if discrepancies and logger is not None:
        logger.warning("Metric parity check exceeded tolerance: %s", discrepancies)
    return recomputed


def attach_metrics(
    frame: pd.DataFrame,
    *,
    trace_columns: Sequence[str],
    fly_column: str,
    missing_ttpd_policy: str = "zero",
) -> pd.DataFrame:
    """Return a copy of ``frame`` with metric columns recomputed and attached."""

    metrics = compute_metrics(
        frame,
        trace_columns=trace_columns,
        fly_column=fly_column,
        missing_ttpd_policy=missing_ttpd_policy,
    )
    merged = frame.copy()
    for column in metrics.columns:
        merged[column] = metrics[column]
    return merged


def derive_labels_from_intensity(intensity: pd.Series) -> pd.Series:
    """Convert intensity scores to binary labels following repository policy."""

    labels = (intensity.astype(float) > 0).astype(int)
    labels.name = LABEL_COLUMN
    return labels


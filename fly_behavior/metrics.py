"""Metric computation helpers."""

from __future__ import annotations

import numpy as np

from .envelope import DEFAULT_SMOOTHING_FPS

__all__ = ["compute_metrics"]


def compute_metrics(signal: np.ndarray, fps: float, threshold: float) -> dict | None:
    """Compute core reaction metrics from a 1D signal."""
    signal = np.asarray(signal).astype(float)
    n = len(signal)
    if n == 0:
        return None
    fps_val = fps if fps and fps > 0 else DEFAULT_SMOOTHING_FPS
    duration = n / fps_val if fps_val > 0 else float(n)

    thresh = float(threshold) if threshold is not None else 0.0
    above = signal > thresh
    time_fraction = float(np.count_nonzero(above)) / float(n)

    if above.any():
        auc = float(np.sum(signal[above] - thresh)) / (fps_val if fps_val > 0 else 1.0)
        first_cross_idx = int(np.flatnonzero(above)[0])
        time_to_threshold = (
            first_cross_idx / fps_val if fps_val > 0 else float(first_cross_idx)
        )
    else:
        auc = 0.0
        first_cross_idx = None
        time_to_threshold = None

    return {
        'time_fraction': time_fraction,
        'auc': auc,
        'duration': duration,
        'crossed_threshold': bool(first_cross_idx is not None),
        'time_to_threshold': time_to_threshold,
    }

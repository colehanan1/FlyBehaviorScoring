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

    peak_value = None
    time_to_peak = None
    rise_speed = None
    rise_acceleration = None

    if first_cross_idx is not None:
        post_cross = signal[first_cross_idx:]
        if post_cross.size:
            rel_peak_idx = int(np.argmax(post_cross))
            peak_idx = first_cross_idx + rel_peak_idx
            peak_value = float(signal[peak_idx])
            if fps_val > 0:
                time_to_peak = (peak_idx - first_cross_idx) / fps_val
            else:
                time_to_peak = float(peak_idx - first_cross_idx)

            if time_to_peak and time_to_peak > 0:
                value_at_cross = float(signal[first_cross_idx])
                rise_amplitude = max(0.0, peak_value - value_at_cross)
                rise_speed = rise_amplitude / time_to_peak

                if peak_idx > first_cross_idx and fps_val > 0:
                    window = signal[first_cross_idx : peak_idx + 1]
                    if window.size >= 2:
                        velocities = np.gradient(window, 1.0 / fps_val)
                        rise_acceleration = (velocities[-1] - velocities[0]) / time_to_peak
                    else:
                        rise_acceleration = 0.0
                else:
                    rise_acceleration = 0.0
            else:
                time_to_peak = 0.0
                rise_speed = 0.0
                rise_acceleration = 0.0

    return {
        'time_fraction': time_fraction,
        'auc': auc,
        'duration': duration,
        'crossed_threshold': bool(first_cross_idx is not None),
        'time_to_threshold': time_to_threshold,
        'time_to_peak': time_to_peak,
        'peak_value': peak_value,
        'rise_speed': rise_speed,
        'rise_acceleration': rise_acceleration,
    }

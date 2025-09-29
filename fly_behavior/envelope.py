"""Envelope extraction and threshold helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SMOOTHING_FPS: float = 40.0
ENVELOPE_WINDOW_SECONDS: float = 0.25


def _analytic_signal(values: np.ndarray) -> np.ndarray:
    """Return analytic signal via FFT-based Hilbert transform (SciPy-free)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return arr.astype(complex)
    spectrum = np.fft.fft(arr, n)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * h)


def compute_envelope(signal: np.ndarray, fps: float) -> np.ndarray:
    """Compute clipped analytic envelope smoothed over ENVELOPE_WINDOW_SECONDS."""
    arr = np.asarray(signal, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 100.0)
    analytic = _analytic_signal(arr)
    env = np.abs(analytic)
    fps_for_window = fps if fps and fps > 0 else DEFAULT_SMOOTHING_FPS
    window_frames = max(int(round(ENVELOPE_WINDOW_SECONDS * fps_for_window)), 1)
    if window_frames > 1:
        env = (
            pd.Series(env)
            .rolling(window=window_frames, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
    return env


def compute_threshold(
    time_s: np.ndarray,
    envelope: np.ndarray,
    fps: float,
    baseline_seconds: float,
) -> float:
    """Compute μ_baseline + 4σ_baseline using the baseline window."""
    env = np.asarray(envelope, dtype=float)
    if env.size == 0:
        return 0.0
    times = np.asarray(time_s, dtype=float)
    if times.size != env.size:
        times = np.arange(env.size, dtype=float) / (
            fps if fps and fps > 0 else DEFAULT_SMOOTHING_FPS
        )
    baseline_cutoff = float(max(baseline_seconds, 0.0))
    pre_mask = times < baseline_cutoff
    if np.any(pre_mask):
        pre_vals = env[pre_mask]
    else:
        n_pre = max(
            int(round(baseline_cutoff * (fps if fps and fps > 0 else DEFAULT_SMOOTHING_FPS))),
            1,
        )
        pre_vals = env[:n_pre]
    if pre_vals.size == 0:
        return 0.0
    mu = float(np.nanmean(pre_vals))
    sigma = float(np.nanstd(pre_vals, ddof=0))
    return mu + 4.0 * sigma


__all__ = [
    "DEFAULT_SMOOTHING_FPS",
    "ENVELOPE_WINDOW_SECONDS",
    "compute_envelope",
    "compute_threshold",
]

"""Preprocessing utilities for flypca."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, filtfilt, hilbert, savgol_filter

from .io import TrialTimeseries

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SmoothingConfig:
    """Configuration for smoothing filters."""

    enable: bool = True
    savgol_window_ms: int = 151
    savgol_poly: int = 3
    lowpass_hz: Optional[float] = None


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for trial extraction and preprocessing."""

    fps: float
    pre_s: float
    post_s: float
    smoothing: SmoothingConfig


def compute_savgol_window(fps: float, window_ms: int) -> int:
    """Compute the odd Savitzky-Golay window length."""

    samples = max(int(round(window_ms / 1000 * fps)), 1)
    if samples % 2 == 0:
        samples += 1
    if samples <= 1:
        return 3
    return max(samples, 3)


def savgol_smooth(trace: np.ndarray, fps: float, config: SmoothingConfig) -> np.ndarray:
    """Apply Savitzky-Golay smoothing and optional low-pass filtering."""

    if not config.enable:
        return trace
    if trace.size < 3:
        return trace
    window = min(compute_savgol_window(fps, config.savgol_window_ms), trace.size)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return trace
    poly = min(config.savgol_poly, window - 1)
    smoothed = savgol_filter(trace, window_length=window, polyorder=poly, mode="interp")
    if config.lowpass_hz is not None:
        nyquist = 0.5 * fps
        normal_cutoff = config.lowpass_hz / nyquist
        if not 0 < normal_cutoff < 1:
            raise ValueError("Low-pass cutoff must be between 0 and Nyquist frequency.")
        b, a = butter(N=3, Wn=normal_cutoff, btype="low", analog=False)
        smoothed = filtfilt(b, a, smoothed, method="gust")
    return smoothed


def baseline_zscore(trace: np.ndarray, baseline_slice: slice) -> Tuple[np.ndarray, float, float]:
    """Z-score a trace using the values in baseline_slice."""

    baseline_values = trace[baseline_slice]
    mean = float(np.mean(baseline_values))
    std = float(np.std(baseline_values, ddof=0))
    if std == 0:
        std = 1.0
    zscored = (trace - mean) / std
    return zscored, mean, std


def extract_aligned_window(trial: TrialTimeseries, pre_s: float, post_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a window around the odor onset."""

    samples_pre = int(round(pre_s * trial.fps))
    samples_post = int(round(post_s * trial.fps))
    start = max(trial.odor_on_idx - samples_pre, 0)
    end = min(trial.odor_on_idx + samples_post, len(trial.distance))
    distance = trial.distance[start:end]
    time = trial.time[start:end] - trial.time[trial.odor_on_idx]
    if len(distance) == 0:
        raise ValueError(f"Empty window extracted for trial {trial.trial_id}.")
    return time, distance


def preprocess_trial(
    trial: TrialTimeseries,
    config: PreprocessConfig,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Align, smooth, and z-score a trial."""

    time, distance = extract_aligned_window(trial, config.pre_s, config.post_s)
    smoothed = savgol_smooth(distance, trial.fps, config.smoothing)
    pre_samples = int(round(config.pre_s * trial.fps))
    baseline_end = min(pre_samples, len(smoothed))
    baseline_slice = slice(0, baseline_end)
    zscored, mean, std = baseline_zscore(smoothed, baseline_slice)
    return time, zscored, mean, std


def compute_velocity(trace: ArrayLike, fps: float) -> np.ndarray:
    """Compute the first derivative of a trace."""

    arr = np.asarray(trace, dtype=float)
    if arr.size < 2:
        return np.zeros_like(arr)
    dt = 1.0 / fps
    velocity = np.gradient(arr, dt)
    return velocity


def compute_hilbert_envelope(trace: ArrayLike) -> np.ndarray:
    """Compute the Hilbert envelope of a trace."""

    arr = np.asarray(trace, dtype=float)
    analytic = hilbert(arr)
    return np.abs(analytic)

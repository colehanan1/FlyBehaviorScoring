"""Feature engineering for flypca."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import signal

from .io import TrialTimeseries
from .lagpca import LagPCAResult, project_trial
from .preprocess import PreprocessConfig, SmoothingConfig, compute_hilbert_envelope, compute_velocity, preprocess_trial

LOGGER = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature computation."""

    pre_s: float
    post_s: float
    threshold_k: float
    smoothing: SmoothingConfig
    frequency: Dict[str, object]


def _auc(time: np.ndarray, values: np.ndarray) -> float:
    if time.size < 2:
        return float(np.nan)
    return float(np.trapezoid(values, time))


def _first_crossing(time: np.ndarray, values: np.ndarray, threshold: float) -> float:
    for t, v in zip(time, values):
        if v >= threshold:
            return float(t)
    return float("nan")


def _rise_time(time: np.ndarray, values: np.ndarray, baseline: float, peak: float) -> float:
    amplitude = peak - baseline
    if not np.isfinite(amplitude) or amplitude <= 0:
        return float("nan")
    lower = baseline + 0.1 * amplitude
    upper = baseline + 0.9 * amplitude
    lower_time = _first_crossing(time, values, lower)
    upper_time = _first_crossing(time, values, upper)
    if np.isnan(lower_time) or np.isnan(upper_time):
        return float("nan")
    return float(upper_time - lower_time)


def _decay_half_life(time: np.ndarray, values: np.ndarray, peak_idx: int, baseline: float) -> float:
    peak_value = values[peak_idx]
    target = baseline + 0.5 * (peak_value - baseline)
    for t, v in zip(time[peak_idx:], values[peak_idx:]):
        if v <= target:
            return float(t - time[peak_idx])
    return float("nan")


def _excursion_stats(time: np.ndarray, values: np.ndarray, threshold: float) -> Tuple[int, float, float]:
    above = values >= threshold
    if not np.any(above):
        return 0, float("nan"), float("nan")
    indices = np.where(above)[0]
    gaps = np.diff(indices)
    split_points = np.where(gaps > 1)[0]
    segments = np.split(indices, split_points + 1)
    count = len(segments)
    onset_times = [time[seg[0]] for seg in segments]
    if len(onset_times) < 2:
        return count, float("nan"), float("nan")
    inter = np.diff(onset_times)
    return count, float(np.mean(inter)), float(np.std(inter))


def _frequency_features(time: np.ndarray, values: np.ndarray, fps: float, bands: Iterable[Tuple[float, float]]) -> Dict[str, float]:
    if values.size < 4:
        return {}
    freq, psd = signal.welch(values, fs=fps, nperseg=min(256, values.size))
    features: Dict[str, float] = {}
    total_power = np.trapezoid(psd, freq)
    for idx, (low, high) in enumerate(bands):
        mask = (freq >= low) & (freq <= high)
        band_power = np.trapezoid(psd[mask], freq[mask]) if np.any(mask) else 0.0
        features[f"bandpower_{idx}"] = float(band_power)
        features[f"bandpower_ratio_{idx}"] = float(band_power / total_power) if total_power > 0 else float("nan")
    return features


def _baseline_stats(time: np.ndarray, values: np.ndarray) -> Dict[str, float]:
    if time.size < 2:
        return {"baseline_mean": float(np.mean(values)), "baseline_std": float(np.std(values)), "baseline_slope": float("nan")}
    mean = float(np.mean(values))
    std = float(np.std(values))
    slope = float(np.polyfit(time, values, 1)[0]) if time.size >= 2 else float("nan")
    return {
        "baseline_mean": mean,
        "baseline_std": std,
        "baseline_slope": slope,
    }


def compute_trial_features(
    trial: TrialTimeseries,
    config: FeatureConfig,
    result: Optional[LagPCAResult] = None,
    projections: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, float]:
    preproc = PreprocessConfig(
        fps=trial.fps,
        pre_s=config.pre_s,
        post_s=config.post_s,
        smoothing=config.smoothing,
    )
    time, zscored, _, _ = preprocess_trial(trial, preproc)
    velocity = compute_velocity(zscored, trial.fps)
    envelope = compute_hilbert_envelope(zscored)
    before_mask = time < 0
    during_mask = (time >= 0) & (time <= config.post_s)
    after_mask = time > config.post_s
    baseline_features = _baseline_stats(time[before_mask], zscored[before_mask])
    baseline_mean = baseline_features["baseline_mean"]
    baseline_std = baseline_features["baseline_std"] if baseline_features["baseline_std"] > 0 else 1.0
    threshold = baseline_mean + config.threshold_k * baseline_std
    during_time = time[during_mask]
    during_values = zscored[during_mask]
    latency = _first_crossing(during_time, during_values, threshold)
    if np.isfinite(latency) and latency <= 0 and during_values.size:
        latency = float(latency + 1.0 / trial.fps)
    if during_values.size:
        peak_idx = int(np.nanargmax(during_values))
        peak_value = float(during_values[peak_idx])
        peak_time = float(during_time[peak_idx])
        rise_time = _rise_time(during_time, during_values, baseline_mean, peak_value)
        decay_half = _decay_half_life(during_time, during_values, peak_idx, baseline_mean)
        duty_cycle = float(np.mean(during_values > threshold))
        num_excursions, inter_mean, inter_std = _excursion_stats(during_time, during_values, threshold)
    else:
        peak_idx = 0
        peak_value = float("nan")
        peak_time = float("nan")
        rise_time = float("nan")
        decay_half = float("nan")
        duty_cycle = float("nan")
        num_excursions, inter_mean, inter_std = 0, float("nan"), float("nan")
    velocity_during = np.abs(velocity[during_mask]) if during_values.size else np.array([])
    auc_features = {
        "auc_before": _auc(time[before_mask], zscored[before_mask]),
        "auc_during": _auc(time[during_mask], zscored[during_mask]),
        "auc_after": _auc(time[after_mask], zscored[after_mask]),
    }
    snr = float(np.sqrt(np.mean(during_values ** 2)) / baseline_std) if (during_values.size and baseline_std > 0) else float("nan")
    freq_features: Dict[str, float] = {}
    if config.frequency.get("enable", False):
        bands = config.frequency.get("bands_hz", [])
        freq_features = _frequency_features(time[during_mask], zscored[during_mask], trial.fps, bands)
    def _safe_ratio(num: float, den: float) -> float:
        if not np.isfinite(num) or not np.isfinite(den) or den == 0:
            return float("nan")
        return float(num / den)

    interaction_features = {
        "during_mean": float(np.mean(during_values)) if during_values.size else float("nan"),
        "during_abs_velocity_mean": float(np.mean(velocity_during)) if velocity_during.size else float("nan"),
        "auc_diff_during_before": auc_features["auc_during"] - auc_features["auc_before"],
        "auc_ratio_during_before": _safe_ratio(auc_features["auc_during"], auc_features["auc_before"]),
    }
    hilbert_during = envelope[during_mask] if np.any(during_mask) else np.array([])
    hilbert_features = {
        "hilbert_mean": float(np.mean(hilbert_during)) if hilbert_during.size else float("nan"),
        "hilbert_max": float(np.max(hilbert_during)) if hilbert_during.size else float("nan"),
        "hilbert_var": float(np.var(hilbert_during)) if hilbert_during.size else float("nan"),
        "hilbert_tmax": float(time[during_mask][int(np.argmax(hilbert_during))]) if hilbert_during.size else float("nan"),
    }
    if velocity.size:
        max_idx = int(np.argmax(np.abs(velocity)))
        velocity_features = {
            "velocity_max_abs": float(np.max(np.abs(velocity))),
            "velocity_tmax_abs": float(time[max_idx]),
            "velocity_total_abs": float(np.trapezoid(np.abs(velocity), time)),
        }
    else:
        velocity_features = {
            "velocity_max_abs": float("nan"),
            "velocity_tmax_abs": float("nan"),
            "velocity_total_abs": float("nan"),
        }
    pc_features: Dict[str, float] = {}
    if result is not None:
        if projections and trial.trial_id in projections:
            pc_time, pc_values = projections[trial.trial_id]
        else:
            pc_time, pc_values = project_trial(trial, result)
        if pc_values.size:
            pc1 = pc_values[:, 0]
            mask_0_1 = (pc_time >= 0) & (pc_time <= 1.0)
            mask_1_2 = (pc_time > 1.0) & (pc_time <= 2.0)
            pc_features = {
                "pc1_auc_0_1": _auc(pc_time[mask_0_1], pc1[mask_0_1]),
                "pc1_auc_1_2": _auc(pc_time[mask_1_2], pc1[mask_1_2]),
                "pc1_max": float(np.max(pc1)) if pc1.size else float("nan"),
                "pc1_time_to_threshold": _first_crossing(pc_time, pc1, threshold),
                "pc_path_length": float(np.sum(np.linalg.norm(np.diff(pc_values[:, :2], axis=0), axis=1))) if pc_values.shape[1] >= 2 else float("nan"),
            }
    features = {
        "trial_id": trial.trial_id,
        "fly_id": trial.fly_id,
        "latency": latency,
        "peak_value": peak_value,
        "peak_time": peak_time,
        "rise_time_10_90": rise_time,
        "decay_half_life": decay_half,
        "duty_cycle": duty_cycle,
        "num_excursions": float(num_excursions),
        "inter_excursion_mean": inter_mean,
        "inter_excursion_std": inter_std,
        "peak_to_baseline": float(peak_value - baseline_mean),
        "snr": snr,
    }
    features.update(auc_features)
    features.update(baseline_features)
    features.update(interaction_features)
    features.update(hilbert_features)
    features.update(velocity_features)
    features.update(freq_features)
    features.update(pc_features)
    return features


def compute_feature_table(
    trials: Iterable[TrialTimeseries],
    config: dict,
    result: Optional[LagPCAResult] = None,
    projections: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> pd.DataFrame:
    """Compute features for all trials and return a tidy DataFrame."""

    smoothing_cfg = config.get("smoothing", {})
    smoothing = SmoothingConfig(
        enable=bool(smoothing_cfg.get("enable", True)),
        savgol_window_ms=int(smoothing_cfg.get("savgol_window_ms", 151)),
        savgol_poly=int(smoothing_cfg.get("savgol_poly", 3)),
        lowpass_hz=smoothing_cfg.get("lowpass_hz"),
    )
    feature_config = FeatureConfig(
        pre_s=float(config.get("pre_s", 2.0)),
        post_s=float(config.get("post_s", 2.0)),
        threshold_k=float(config.get("threshold_k", 4.0)),
        smoothing=smoothing,
        frequency=config.get("frequency", {}),
    )
    records = []
    for trial in trials:
        record = compute_trial_features(trial, feature_config, result=result, projections=projections)
        records.append(record)
    df = pd.DataFrame.from_records(records)
    return df

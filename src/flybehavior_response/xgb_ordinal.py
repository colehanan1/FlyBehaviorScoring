"""XGBoost ordinal scoring model — inference utilities.

Provides everything needed to load a trained XGBoost multiclass ordinal model,
compute the 24-feature input (11 engineered + 13 signal), and predict scores
on the {-1, 0, 1, 2, 3, 4, 5} scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import signal as scipy_signal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ORIGINAL_CLASSES = np.array([-1, 0, 1, 2, 3, 4, 5])
N_CLASSES = len(ORIGINAL_CLASSES)

ORIGINAL_TO_XGBOOST = {orig: idx for idx, orig in enumerate(ORIGINAL_CLASSES)}
XGBOOST_TO_ORIGINAL = {idx: orig for idx, orig in enumerate(ORIGINAL_CLASSES)}

FPS = 40
ODOR_ON_SEC = 32
ODOR_OFF_SEC = 62
ODOR_ON_IDX = int(ODOR_ON_SEC * FPS)   # 1280
ODOR_OFF_IDX = int(ODOR_OFF_SEC * FPS)  # 2480

ENGINEERED_FEATURES: list[str] = [
    "AUC-During-Before-Ratio",
    "AUC-After-Before-Ratio",
    "global_max",
    "trimmed_global_min",
    "local_min",
    "local_max",
    "local_min_before",
    "local_max_before",
    "local_min_during",
    "local_max_over_global_min",
    "local_max_during_over_global_min",
]

SIGNAL_FEATURES: list[str] = [
    "mean_shift_z",
    "std_ratio",
    "peak_z",
    "auc_ratio",
    "total_power_ratio",
    "power_ratio_vlow",
    "power_ratio_low",
    "power_ratio_mid",
    "power_ratio_high",
    "power_ratio_vhigh",
    "frac_above_baseline",
    "persistence",
    "time_to_peak_frac",
]

ALL_FEATURES: list[str] = ENGINEERED_FEATURES + SIGNAL_FEATURES

_FREQ_BANDS = {
    "vlow": (0.0, 0.5),
    "low": (0.5, 1.0),
    "mid": (1.0, 3.0),
    "high": (3.0, 5.0),
    "vhigh": (5.0, 10.0),
}

# ---------------------------------------------------------------------------
# Signal feature helpers
# ---------------------------------------------------------------------------


def interpolate_nans(trace: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN gaps in a 1-D trace."""
    x = np.asarray(trace, dtype=float).copy()
    nans = np.isnan(x)
    if not nans.any():
        return x
    if nans.all():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    """Sum PSD values within [fmin, fmax)."""
    m = (freqs >= fmin) & (freqs < fmax)
    if not np.any(m):
        return 0.0
    return float(np.sum(psd[m]))


def compute_signal_features(trace: np.ndarray, fps: int = FPS) -> Dict[str, float]:
    """Compute 13 signal-derived features from a behavioural trace.

    Parameters
    ----------
    trace : 1-D array
        Raw distance/angle trace (e.g. 3600 frames at 40 fps).
    fps : int
        Frames per second.

    Returns
    -------
    dict with keys matching ``SIGNAL_FEATURES``.
    """
    tr = interpolate_nans(trace)
    before = tr[:ODOR_ON_IDX]
    during = tr[ODOR_ON_IDX:ODOR_OFF_IDX]
    after = tr[ODOR_OFF_IDX:]

    eps = 1e-10
    b_mean, b_std = float(np.mean(before)), float(np.std(before))
    d_mean, d_std = float(np.mean(during)), float(np.std(during))
    a_mean = float(np.mean(after))

    auc_before = float(np.trapz(before)) / max(len(before), 1)
    auc_during = float(np.trapz(during)) / max(len(during), 1)

    f_b, pxx_b = scipy_signal.welch(
        before - b_mean,
        fs=fps,
        nperseg=min(256, len(before)),
        noverlap=min(128, max(len(before) // 2, 1)),
    )
    f_d, pxx_d = scipy_signal.welch(
        during - d_mean,
        fs=fps,
        nperseg=min(256, len(during)),
        noverlap=min(128, max(len(during) // 2, 1)),
    )

    out: Dict[str, float] = {}
    out["mean_shift_z"] = (d_mean - b_mean) / (b_std + eps)
    out["std_ratio"] = d_std / (b_std + eps)
    out["peak_z"] = (float(np.max(during)) - b_mean) / (b_std + eps)
    out["auc_ratio"] = auc_during / (auc_before + eps)

    before_total = 0.0
    during_total = 0.0
    for bn, (fmin, fmax) in _FREQ_BANDS.items():
        pb = band_power(pxx_b, f_b, fmin, fmax)
        pdw = band_power(pxx_d, f_d, fmin, fmax)
        out[f"power_ratio_{bn}"] = pdw / (pb + eps)
        before_total += pb
        during_total += pdw

    out["total_power_ratio"] = during_total / (before_total + eps)
    out["frac_above_baseline"] = float(np.mean(during > (b_mean + 2.0 * b_std)))
    out["persistence"] = (a_mean - b_mean) / ((d_mean - b_mean) + eps)
    out["time_to_peak_frac"] = float(np.argmax(during)) / max(len(during), 1)

    return out


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_features(df: pd.DataFrame, fps: int = FPS) -> pd.DataFrame:
    """Build the (N, 24) feature matrix from a wide-format DataFrame.

    The input *df* must contain:
    - The 11 engineered feature columns (``ENGINEERED_FEATURES``)
    - ``dir_val_*`` trace columns for signal feature computation

    Returns a DataFrame with columns ordered as ``ALL_FEATURES``.
    """
    # --- Engineered features ---
    missing_eng = [c for c in ENGINEERED_FEATURES if c not in df.columns]
    if missing_eng:
        raise ValueError(f"Missing engineered columns: {missing_eng}")
    X_eng = df[ENGINEERED_FEATURES].copy().reset_index(drop=True)

    # --- Signal features from dir_val_* traces ---
    dir_cols = sorted(
        [c for c in df.columns if c.startswith("dir_val_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if not dir_cols:
        raise ValueError("No dir_val_* trace columns found in input DataFrame.")

    signal_rows = []
    for _, row in df.iterrows():
        trace = row[dir_cols].values.astype(float)
        signal_rows.append(compute_signal_features(trace, fps=fps))
    X_sig = pd.DataFrame(signal_rows)[SIGNAL_FEATURES].reset_index(drop=True)

    X = pd.concat([X_eng, X_sig], axis=1)

    # --- Clean: inf → NaN, then median-impute ---
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True)
    for col in X.columns:
        X[col] = X[col].fillna(medians.get(col, 0.0))

    return X


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_ordinal_scores(booster: xgb.Booster, feature_df: pd.DataFrame) -> np.ndarray:
    """Return ordinal scores in {-1, 0, 1, 2, 3, 4, 5} for each row."""
    dmat = xgb.DMatrix(feature_df)
    raw = booster.predict(dmat, output_margin=True).reshape(-1, N_CLASSES)
    xgb_idx = np.argmax(raw, axis=1)
    return np.array([XGBOOST_TO_ORIGINAL[int(i)] for i in xgb_idx])


def score_to_binary(scores: np.ndarray, threshold: int = 2) -> np.ndarray:
    """Map ordinal scores to binary reaction flags.

    Scores >= *threshold* → 1 (reaction), else → 0 (no reaction).
    Default threshold=2 means {-1, 0, 1} → 0 and {2, 3, 4, 5} → 1.
    """
    return (scores >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------


def save_xgb_model(booster: xgb.Booster, path: str | Path) -> Path:
    """Save an XGBoost Booster in native JSON format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    print(f"Saved XGBoost model to {path}")
    return path


def load_xgb_model(path: str | Path) -> xgb.Booster:
    """Load an XGBoost Booster from a native JSON/UBJ file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster

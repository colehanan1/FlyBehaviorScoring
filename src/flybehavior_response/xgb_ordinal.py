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


def _welch_psd(x: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD that degrades gracefully for very short segments.

    Per-trial odor timing and ``n_frames`` truncation can collapse a slice down
    to zero or one sample (e.g. a trial whose real duration is shorter than its
    configured odor-off time). ``scipy.signal.welch`` requires
    ``noverlap < nperseg`` and ``nperseg >= 1``, so for segments shorter than 2
    samples a flat zero spectrum is returned instead of raising. The overlap is
    always clamped strictly below ``nperseg`` to preserve that invariant.
    """
    n = len(x)
    if n < 2:
        return np.zeros(1), np.zeros(1)
    nperseg = min(256, n)
    noverlap = min(nperseg - 1, max(nperseg // 2, 1))
    return scipy_signal.welch(x, fs=fps, nperseg=nperseg, noverlap=noverlap)


def band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    """Sum PSD values within [fmin, fmax)."""
    m = (freqs >= fmin) & (freqs < fmax)
    if not np.any(m):
        return 0.0
    return float(np.sum(psd[m]))


def compute_signal_features(
    trace: np.ndarray,
    fps: float = FPS,
    *,
    odor_on_idx: int | None = None,
    odor_off_idx: int | None = None,
    n_frames: int | None = None,
) -> Dict[str, float]:
    """Compute 13 signal-derived features from a behavioural trace.

    Parameters
    ----------
    trace : 1-D array
        Raw distance/angle trace.
    fps : float
        Frames per second (per-trial when known, else ``FPS``).
    odor_on_idx, odor_off_idx : int, optional
        Per-trial odor-on / odor-off frame indices. When omitted the legacy
        constants (``ODOR_ON_IDX=1280``, ``ODOR_OFF_IDX=2480``) are used —
        appropriate for the 90 s / 30 s-odor schedule the model was trained
        on. For datasets recorded under a different schedule (e.g. RandomPanel
        with ~40 s trials and a ~10 s odor) the caller must pass the actual
        per-trial indices so before / during / after slices are correct.
    n_frames : int, optional
        Real recording length. If given, the trace is truncated here so that
        the trailing NaN / zero padding ``dir_val_*`` columns acquire when the
        wide-matrix time axis is longer than the trial does not pollute
        baseline / post-odor statistics.
    """

    tr = interpolate_nans(trace)
    if n_frames is not None and n_frames > 0:
        tr = tr[: int(n_frames)]
    total = len(tr)

    on_idx = int(odor_on_idx) if odor_on_idx is not None else ODOR_ON_IDX
    off_idx = int(odor_off_idx) if odor_off_idx is not None else ODOR_OFF_IDX
    # Clamp into the real trace bounds; preserve before < during < after
    on_idx = max(0, min(on_idx, max(total - 1, 0)))
    off_idx = max(on_idx + 1, min(off_idx, total))

    before = tr[:on_idx]
    during = tr[on_idx:off_idx]
    after = tr[off_idx:]

    eps = 1e-10
    b_mean, b_std = float(np.mean(before)), float(np.std(before))
    d_mean, d_std = float(np.mean(during)), float(np.std(during))
    a_mean = float(np.mean(after))

    auc_before = float(np.trapz(before)) / max(len(before), 1)
    auc_during = float(np.trapz(during)) / max(len(during), 1)

    f_b, pxx_b = _welch_psd(before - b_mean, fps)
    f_d, pxx_d = _welch_psd(during - d_mean, fps)

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


def preprocess_features(df: pd.DataFrame, fps: float = FPS) -> pd.DataFrame:
    """Build the (N, 24) feature matrix from a wide-format DataFrame.

    The input *df* must contain:
    - The 11 engineered feature columns (``ENGINEERED_FEATURES``)
    - ``dir_val_*`` trace columns for signal feature computation

    Optional per-row columns honoured when present (added by the upstream
    pipeline ``build_wide_csv`` so each trial's actual rig timing is used
    instead of the legacy 32 / 62 s constants):
    - ``trial_odor_on_s`` / ``trial_odor_off_s`` — odor on / off in seconds
    - ``trial_duration_s`` — real recording length in seconds
    - ``fps`` — per-trial fps (defaults to ``FPS`` when missing)

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

    has_on = "trial_odor_on_s" in df.columns
    has_off = "trial_odor_off_s" in df.columns
    has_dur = "trial_duration_s" in df.columns
    has_fps = "fps" in df.columns

    signal_rows = []
    for _, row in df.iterrows():
        trace = row[dir_cols].values.astype(float)
        row_fps = float(row["fps"]) if has_fps and pd.notna(row["fps"]) and float(row["fps"]) > 0 else float(fps)
        on_idx: int | None = None
        off_idx: int | None = None
        n_frames: int | None = None
        if has_on and pd.notna(row["trial_odor_on_s"]):
            on_idx = max(0, int(round(float(row["trial_odor_on_s"]) * row_fps)))
        if has_off and pd.notna(row["trial_odor_off_s"]):
            off_idx = int(round(float(row["trial_odor_off_s"]) * row_fps))
        if has_dur and pd.notna(row["trial_duration_s"]):
            n_frames = int(round(float(row["trial_duration_s"]) * row_fps))
        signal_rows.append(
            compute_signal_features(
                trace,
                fps=row_fps,
                odor_on_idx=on_idx,
                odor_off_idx=off_idx,
                n_frames=n_frames,
            )
        )
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

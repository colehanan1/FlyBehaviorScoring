"""Build (and cache) the exact dataset the combined multiclass scorer is trained on.

This mirrors ``notebooks/05_combined_two_datasets_multiclass.ipynb`` cells 3-7:
both blinded-scoring sources are loaded, filtered to their scoring-trace rows,
merged to labels on ``(dataset, fly, fly_number, testing_N)``, and turned into
the 11 engineered + 13 signal features.  The raw ``dir_val_*`` traces are kept
too, because the spectral figures in the supplement are computed on the same
rows that feed the model — otherwise the "why this feature" argument would be
made on a different dataset than the one the model saw.

Run directly to (re)build the cache::

    python scripts/figures/prep_data.py [--force]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "outputs" / "thesis_figures" / "cache"

# --- settings, verbatim from notebook 05 -----------------------------------
SEED = 42
FPS = 40
ODOR_ON_SEC, ODOR_OFF_SEC = 30, 60
ODOR_ON_IDX, ODOR_OFF_IDX = ODOR_ON_SEC * FPS, ODOR_OFF_SEC * FPS
EXCLUDE_TRIAL_NUM_FROM_EVAL = 11
REMAP_6_TO_5 = True
SCORE_LEVELS = [-1, 0, 1, 2, 3, 4, 5]
LABEL_COL = "user_score"

# Trials whose comment marks the response as abdomen movement (not PER) are
# excluded everywhere — train, val and test. The alternation catches the
# observed misspellings: abdomen/abdo, abodmen, abeomen, badomen.
EXCLUDE_COMMENT_REGEX = r"abdo|abodm|abeom|badom"

ENGINEERED_FEATURES = [
    "AUC-During-Before-Ratio", "AUC-After-Before-Ratio", "global_max",
    "trimmed_global_min", "local_min", "local_max", "local_min_before",
    "local_max_before", "local_min_during", "local_max_over_global_min",
    "local_max_during_over_global_min",
]
SIGNAL_FEATURES = [
    "mean_shift_z", "std_ratio", "peak_z", "auc_ratio", "total_power_ratio",
    "power_ratio_vlow", "power_ratio_low", "power_ratio_mid", "power_ratio_high",
    "power_ratio_vhigh", "frac_above_baseline", "persistence", "time_to_peak_frac",
]

BANDS = {
    "vlow": (0.0, 0.5), "low": (0.5, 1.0), "mid": (1.0, 3.0),
    "high": (3.0, 5.0), "vhigh": (5.0, 10.0),
}
BAND_LABELS = {
    "vlow": "0–0.5 Hz", "low": "0.5–1 Hz", "mid": "1–3 Hz",
    "high": "3–5 Hz", "vhigh": "5–10 Hz",
}

DISTANCE_TRACE_REGEX = (
    r"^testing_\d+_fly\d+_distances_fly\d+_angle_distance_rms_envelope(?:\.csv)?$"
)
DATASET_MAP = {
    "opto_3-oct": "3OCT-Training", "opto_ACV": "ACV-Training", "opto_AIR": "AIR-Training",
    "Benz_control": "Benz-Control", "opto_benz_1": "Benz-Training",
    "opto_benz_1-flagged": "Benz-Training-flagged", "EB_control": "EB-Control",
    "opto_EB": "EB-Training", "opto_EB(6-training)": "EB-Training(No-Operant)",
    "opto_EB-flagged": "EB-Training-flagged", "hex_control": "Hex-Control",
    "hex_control-flagged": "Hex-Control-flagged", "opto_hex": "Hex-Training",
    "opto_hex-flagged": "Hex-Training-flagged",
}
SOURCES = [
    {
        "name": "old",
        "data": PROJECT_ROOT / "data/all_envelope_rows_wide_combined_base.csv",
        "labels": Path("/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/blinded_video_scores.csv"),
        "trace_regex": DISTANCE_TRACE_REGEX,
        "dataset_map": DATASET_MAP,
    },
    {
        "name": "new",
        "data": Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/all_envelope_rows_wide_combined_base.csv"),
        "labels": Path("/home/ramanlab/Documents/cole/Data/CSVs-New-Opto-Flys/blinded_video_scores.csv"),
        "trace_regex": None,
        "dataset_map": {},
    },
]


# --- feature computation, verbatim from notebook 05 ------------------------
def read_table(p: Path) -> pd.DataFrame:
    p = Path(p)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def interpolate_nans(trace):
    x = np.asarray(trace, dtype=float).copy()
    nans = np.isnan(x)
    if not nans.any():
        return x
    if nans.all():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[nans] = np.interp(idx[nans], idx[~nans], x[~nans])
    return x


def band_power(psd, freqs, fmin, fmax) -> float:
    m = (freqs >= fmin) & (freqs < fmax)
    return float(np.sum(psd[m])) if np.any(m) else 0.0


def compute_signal_features(trace, fps: int = FPS) -> dict:
    tr = interpolate_nans(trace)
    before, during, after = tr[:ODOR_ON_IDX], tr[ODOR_ON_IDX:ODOR_OFF_IDX], tr[ODOR_OFF_IDX:]
    eps = 1e-10
    b_mean, b_std = np.mean(before), np.std(before)
    d_mean, d_std = np.mean(during), np.std(during)
    a_mean = np.mean(after)
    auc_before = np.trapz(before) / max(len(before), 1)
    auc_during = np.trapz(during) / max(len(during), 1)
    f_b, pxx_b = sps.welch(before - b_mean, fs=fps, nperseg=min(256, len(before)),
                           noverlap=min(128, max(len(before) // 2, 1)))
    f_d, pxx_d = sps.welch(during - d_mean, fs=fps, nperseg=min(256, len(during)),
                           noverlap=min(128, max(len(during) // 2, 1)))
    out = {
        "mean_shift_z": float((d_mean - b_mean) / (b_std + eps)),
        "std_ratio": float(d_std / (b_std + eps)),
        "peak_z": float((np.max(during) - b_mean) / (b_std + eps)),
        "auc_ratio": float(auc_during / (auc_before + eps)),
    }
    bt = dt = 0.0
    for bn, (fmin, fmax) in BANDS.items():
        pb = band_power(pxx_b, f_b, fmin, fmax)
        pdw = band_power(pxx_d, f_d, fmin, fmax)
        out[f"power_ratio_{bn}"] = float(pdw / (pb + eps))
        bt += pb
        dt += pdw
    out["total_power_ratio"] = float(dt / (bt + eps))
    out["frac_above_baseline"] = float(np.mean(during > (b_mean + 2.0 * b_std)))
    out["persistence"] = float((a_mean - b_mean) / ((d_mean - b_mean) + eps))
    out["time_to_peak_frac"] = float(np.argmax(during) / max(len(during), 1))
    return out


def build_source(src: dict) -> pd.DataFrame:
    data = read_table(src["data"])
    raw = read_table(src["labels"])
    lab = raw[["dataset", "fly", "fly_number", "trial_label", LABEL_COL]].copy()
    if "comment" in raw.columns:
        abdo = raw["comment"].astype(str).str.contains(
            EXCLUDE_COMMENT_REGEX, case=False, regex=True, na=False)
        if abdo.any():
            print(f"  [{src['name']}] excluding {int(abdo.sum())} abdomen-comment trials")
            lab = lab[~abdo].copy()
    lab[LABEL_COL] = pd.to_numeric(lab[LABEL_COL], errors="coerce")
    if REMAP_6_TO_5:
        lab[LABEL_COL] = lab[LABEL_COL].replace(6, 5)
    lab = lab.dropna(subset=[LABEL_COL]).copy()
    lab[LABEL_COL] = lab[LABEL_COL].astype(int)

    regex = src["trace_regex"] or r"testing_\d+"
    data = data[data["trial_label"].astype(str).str.match(regex, na=False)].copy()

    dm = src["dataset_map"]

    def norm(df):
        for c in ["dataset", "fly", "trial_label"]:
            df[c] = df[c].astype(str).str.strip()
        df["core"] = df["trial_label"].str.extract(r"(testing_\d+)", expand=False)
        df["ds"] = df["dataset"].map(dm).fillna(df["dataset"])
        df["fly_number"] = pd.to_numeric(df["fly_number"], errors="coerce")
        return df

    data, lab = norm(data), norm(lab)
    keys = ["ds", "fly", "fly_number", "core"]
    merged = data.merge(lab[keys + [LABEL_COL]].drop_duplicates(keys), on=keys, how="inner")
    print(f"  [{src['name']}] trace rows={len(data):,}  labels={len(lab):,}  ->  merged={len(merged):,}")
    return merged


def build() -> dict:
    """Load both sources, compute features + trace matrix, return the bundle."""
    print("Building combined dataset (this reads two large wide CSVs)…")
    Xe_list, Xs_list, y_list, meta_list, trace_list = [], [], [], [], []
    n_frames = None

    for src in SOURCES:
        merged = build_source(src)
        dir_cols = sorted([c for c in merged.columns if c.startswith("dir_val_")],
                          key=lambda x: int(x.split("_")[-1]))
        traces = merged[dir_cols].to_numpy(dtype=np.float64)
        traces = np.vstack([interpolate_nans(t) for t in traces])
        n_frames = traces.shape[1] if n_frames is None else min(n_frames, traces.shape[1])
        trace_list.append(traces)

        sig = pd.DataFrame([compute_signal_features(t) for t in traces])
        Xe_list.append(merged[ENGINEERED_FEATURES].reset_index(drop=True))
        Xs_list.append(sig[SIGNAL_FEATURES].reset_index(drop=True))
        y_list.append(merged[LABEL_COL].reset_index(drop=True))

        meta = merged[["dataset", "fly", "fly_number", "trial_label"]].reset_index(drop=True).copy()
        meta["source"] = src["name"]
        meta["trial_num"] = pd.to_numeric(
            meta["trial_label"].astype(str).str.extract(r"testing_(\d+)")[0], errors="coerce"
        ).fillna(-1).astype(int)
        meta_list.append(meta)
        del merged

    traces = np.vstack([t[:, :n_frames] for t in trace_list]).astype(np.float32)
    X_engineered = pd.concat(Xe_list, ignore_index=True)
    X_signal = pd.concat(Xs_list, ignore_index=True)
    y = pd.concat(y_list, ignore_index=True).astype(int)
    meta = pd.concat(meta_list, ignore_index=True)

    for X in (X_engineered, X_signal):
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(numeric_only=True), inplace=True)

    print(f"\n  traces:     {traces.shape}  ({n_frames / FPS:.1f} s at {FPS} fps)")
    print(f"  engineered: {X_engineered.shape}   signal: {X_signal.shape}")
    print(f"  label counts:\n{y.value_counts().sort_index().to_string()}")
    return {"traces": traces, "y": y, "meta": meta,
            "X_engineered": X_engineered, "X_signal": X_signal}


def load(force: bool = False) -> dict:
    """Return the bundle, using the on-disk cache when it is present."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tr_p = CACHE_DIR / "traces.npy"
    if not force and tr_p.exists():
        print(f"Loading cached dataset from {CACHE_DIR}")
        return {
            "traces": np.load(tr_p),
            "y": pd.read_parquet(CACHE_DIR / "labels.parquet")["y"],
            "meta": pd.read_parquet(CACHE_DIR / "meta.parquet"),
            "X_engineered": pd.read_parquet(CACHE_DIR / "X_engineered.parquet"),
            "X_signal": pd.read_parquet(CACHE_DIR / "X_signal.parquet"),
        }
    bundle = build()
    np.save(tr_p, bundle["traces"])
    bundle["y"].to_frame("y").to_parquet(CACHE_DIR / "labels.parquet")
    bundle["meta"].to_parquet(CACHE_DIR / "meta.parquet")
    bundle["X_engineered"].to_parquet(CACHE_DIR / "X_engineered.parquet")
    bundle["X_signal"].to_parquet(CACHE_DIR / "X_signal.parquet")
    print(f"Cached to {CACHE_DIR}")
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    load(force=ap.parse_args().force)

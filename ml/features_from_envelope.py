"""Utilities for turning raw envelope traces into engineered features."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

ODOR_ONSET_S: float = 2.0
ODOR_OFFSET_S: float = 7.0
BASELINE_END_S: float = 1.5
GLOBAL_PCA_COMPONENTS: int = 5
FOLD_PCA_COMPONENTS: int = 3
PCA_MIN_SAMPLES: int = 5
RANDOM_SEED: int = 42


def load_envelope(npy_path: str, mmap: bool = True) -> np.ndarray:
    """Load the envelope matrix with optional memory mapping."""
    path = Path(npy_path)
    if not path.exists():
        raise FileNotFoundError(f"Envelope matrix not found at {path}")
    logger.info("Loading envelope matrix from %s (mmap=%s)", path, mmap)
    kwargs = {"mmap_mode": "r"} if mmap else {}
    matrix = np.load(path, **kwargs)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D envelope matrix, got shape {matrix.shape}")
    logger.info("Envelope matrix shape: trials=%s, width=%s", matrix.shape[0], matrix.shape[1])
    return matrix


def _extract_columns(code_map: Dict) -> List[str]:
    if isinstance(code_map, dict):
        for key in ("columns", "column_names", "col_names", "headers"):
            if key in code_map and isinstance(code_map[key], Sequence):
                return list(code_map[key])
        if "schema" in code_map and isinstance(code_map["schema"], dict):
            schema = code_map["schema"]
            if "fields" in schema and isinstance(schema["fields"], Sequence):
                return [field.get("name") for field in schema["fields"]]
    if isinstance(code_map, Sequence):
        return list(code_map)
    raise ValueError("Unable to determine column order from code map JSON")


def load_code_map(json_path: str) -> Dict:
    """Load the code map JSON and extract ordered column names."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Code map JSON not found at {path}")
    logger.info("Loading code map from %s", path)
    with path.open("r", encoding="utf-8") as f:
        code_map = json.load(f)
    columns = _extract_columns(code_map)
    if not columns:
        raise ValueError("Code map did not provide any columns")
    logger.info("Code map defines %d columns", len(columns))
    return {"columns": columns, "raw": code_map}


def _validate_schema(columns: Sequence[str]) -> None:
    required = ["dataset", "fly", "trial_type", "trial_label", "fps"]
    for name in required:
        if name not in columns:
            raise ValueError(f"Required column '{name}' missing from code map")
    dir_cols = [c for c in columns if c.startswith("dir_val_")]
    if not dir_cols:
        raise ValueError("No dir_val_* columns found in code map")


def assemble_envelope_df(npy_path: str, json_path: str) -> pd.DataFrame:
    """Load envelope matrix and assemble a fully-labeled DataFrame of features."""
    np.random.seed(RANDOM_SEED)
    matrix = load_envelope(npy_path, mmap=True)
    code_map = load_code_map(json_path)
    columns = code_map["columns"]
    _validate_schema(columns)
    if matrix.shape[1] != len(columns):
        raise ValueError(
            f"Envelope width {matrix.shape[1]} does not match column count {len(columns)}"
        )

    logger.info("Building envelope DataFrame with %d rows", matrix.shape[0])
    df = pd.DataFrame(matrix, columns=columns)
    df["dataset"] = df["dataset"].astype(str)
    df["fly"] = df["fly"].astype(str)
    df["trial_type"] = df["trial_type"].astype(str)
    df["trial_label"] = df["trial_label"].astype(str)
    df["fps"] = df["fps"].astype(float)

    dir_cols = [c for c in df.columns if c.startswith("dir_val_")]
    df[dir_cols] = df[dir_cols].astype(np.float32)

    feature_records: List[Dict[str, float]] = []
    odor_vectors: List[np.ndarray] = []
    timepoint_counts: List[int] = []

    for idx, row in df.iterrows():
        fps = float(row["fps"])
        if fps <= 0:
            raise ValueError(f"Non-positive fps ({fps}) at row {idx}")
        signal = row[dir_cols].to_numpy(dtype=np.float32, copy=True)
        n_time = signal.shape[0]
        i_on = int(round(ODOR_ONSET_S * fps))
        i_off = int(round(ODOR_OFFSET_S * fps))
        i_base_end = int(round(BASELINE_END_S * fps))
        if i_off > n_time:
            raise ValueError(
                f"Odor offset index {i_off} exceeds available timepoints {n_time} for row {idx}"
            )
        if i_base_end <= 0:
            raise ValueError(f"Invalid baseline end index {i_base_end} for row {idx}")
        baseline = signal[:i_base_end]
        odor_window = signal[i_on:i_off]
        if odor_window.size == 0:
            raise ValueError(f"Empty odor window for row {idx}")

        baseline_mean = float(baseline.mean())
        baseline_std = float(baseline.std(ddof=0))
        odor_mean = float(odor_window.mean())
        odor_peak = float(odor_window.max())
        auc = float(np.trapz(odor_window, dx=1.0 / fps))
        threshold = baseline_mean + 4.0 * baseline_std
        odor_above = np.clip(odor_window - threshold, a_min=0.0, a_max=None)
        auc_above = float(np.trapz(odor_above, dx=1.0 / fps))
        pct_above = float((odor_window >= threshold).mean() * 100.0)

        post_onset = signal[i_on:i_off]
        cross_indices = np.where(post_onset >= threshold)[0]
        if cross_indices.size > 0:
            latency_sec = float(cross_indices[0] / fps)
            crossed = 1
        else:
            latency_sec = math.inf
            crossed = 0
        rise_window_end = min(i_on + int(round(1.0 * fps)) + 1, n_time)
        rise_segment = signal[i_on:rise_window_end]
        if rise_segment.size <= 1:
            rise_slope = 0.0
        else:
            diffs = np.diff(rise_segment) * fps
            rise_slope = float(np.max(diffs))

        odor_duration = float((i_off - i_on) / fps)
        feature_records.append(
            {
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "odor_mean": odor_mean,
                "odor_peak": odor_peak,
                "odor_auc": auc,
                "odor_auc_above_thr": auc_above,
                "odor_pct_above_thr": pct_above,
                "latency_to_thr": latency_sec,
                "latency_crossed": crossed,
                "rise_slope": rise_slope,
                "odor_window_len_s": odor_duration,
            }
        )
        odor_vectors.append(odor_window.astype(np.float32, copy=True))
        timepoint_counts.append(odor_window.size)

    feature_df = pd.DataFrame(feature_records, index=df.index)
    df = pd.concat([df, feature_df], axis=1)
    df["odor_window_vector"] = odor_vectors
    df["trial_key"] = (
        df["dataset"].astype(str)
        + "__"
        + df["fly"].astype(str)
        + "__"
        + df["trial_type"].astype(str)
        + "__"
        + df["trial_label"].astype(str)
    )

    lengths = set(timepoint_counts)
    logger.info("Odor window length summary (samples): %s", sorted(lengths))
    min_len = min(lengths)
    if len(lengths) > 1:
        logger.warning(
            "Found odor windows with varying length; PCA will truncate/pad to %d samples",
            min_len,
        )

    if len(odor_vectors) >= PCA_MIN_SAMPLES:
        target_len = min_len
        matrix = np.zeros((len(odor_vectors), target_len), dtype=np.float32)
        for i, vec in enumerate(odor_vectors):
            if vec.size >= target_len:
                matrix[i] = vec[:target_len]
            else:
                matrix[i, : vec.size] = vec
                fill = vec[-1] if vec.size else 0.0
                matrix[i, vec.size :] = fill
        n_components = min(GLOBAL_PCA_COMPONENTS, matrix.shape[0], matrix.shape[1])
        if n_components > 0:
            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            comps = pca.fit_transform(matrix)
            for comp_idx in range(comps.shape[1]):
                df[f"global_pca_{comp_idx + 1}"] = comps[:, comp_idx]
            logger.info(
                "Global PCA fitted with %d components for visualization", comps.shape[1]
            )
    else:
        logger.warning("Insufficient trials (%d) for global PCA", len(odor_vectors))

    sample_indices = np.random.choice(df.index, size=min(3, len(df)), replace=False)
    for idx in sample_indices:
        row = df.loc[idx]
        logger.debug(
            "Sample trial %s: baseline_mean=%.3f odor_peak=%.3f latency=%s",
            row["trial_key"],
            row["baseline_mean"],
            row["odor_peak"],
            row["latency_to_thr"],
        )

    return df


def prepare_pca_matrix(vectors: Sequence[np.ndarray], target_len: int) -> np.ndarray:
    """Prepare a matrix of odor window vectors cropped/padded to target length."""
    matrix = np.zeros((len(vectors), target_len), dtype=np.float32)
    for i, vec in enumerate(vectors):
        if vec.size >= target_len:
            matrix[i] = vec[:target_len]
        else:
            matrix[i, : vec.size] = vec
            fill = vec[-1] if vec.size else 0.0
            matrix[i, vec.size :] = fill
    return matrix


__all__ = [
    "assemble_envelope_df",
    "load_code_map",
    "load_envelope",
    "prepare_pca_matrix",
]

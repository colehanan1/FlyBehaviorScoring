"""Tests for feature engineering."""

from __future__ import annotations

import numpy as np

from flypca.features import compute_feature_table


def test_latency_correlates_with_ground_truth(synthetic_dataset) -> None:
    trials, meta = synthetic_dataset
    config = {
        "pre_s": 2.0,
        "post_s": 2.0,
        "threshold_k": 2.5,
        "smoothing": {"enable": True, "savgol_window_ms": 151, "savgol_poly": 3},
    }
    features = compute_feature_table(trials, config)
    merged = features.merge(meta, on=["trial_id", "fly_id"], suffixes=("", "_gt"))
    valid = merged["latency"].notna()
    corr = np.corrcoef(merged.loc[valid, "latency"], merged.loc[valid, "latency_gt"])[0, 1]
    assert corr > 0.6
    assert (merged.loc[valid, "latency"] > 0).all()

"""End-to-end test on synthetic data."""

from __future__ import annotations

import numpy as np

from flypca.cluster import cluster_features, evaluate_with_labels
from flypca.features import compute_feature_table
from flypca.lagpca import fit_lag_pca_for_trials, project_trial


def test_end_to_end_pipeline(synthetic_dataset) -> None:
    trials, meta = synthetic_dataset
    config = {
        "pre_s": 2.0,
        "post_s": 2.0,
        "lag_ms": 250,
        "n_components": 5,
        "threshold_k": 2.5,
        "smoothing": {"enable": True, "savgol_window_ms": 151, "savgol_poly": 3},
    }
    result = fit_lag_pca_for_trials(trials, config)
    projections = {trial.trial_id: project_trial(trial, result) for trial in trials}
    feature_table = compute_feature_table(trials, config, result=result, projections=projections)
    feature_table = feature_table.merge(meta, on=["trial_id", "fly_id"], suffixes=("", "_gt"))
    clustering = cluster_features(feature_table.drop(columns=["reaction", "amplitude", "latency_gt"]), method="gmm", n_components=2)
    assert clustering.metrics["silhouette"] > 0.3
    evaluations = evaluate_with_labels(
        feature_table.drop(columns=["reaction", "amplitude", "latency_gt"]),
        feature_table["reaction"],
        feature_table["fly_id"],
    )
    assert evaluations["auroc"] > 0.8
    assert evaluations["auprc"] > 0.8

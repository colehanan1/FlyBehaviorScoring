"""Unit tests for clustering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flypca.cluster import cluster_features


def _make_feature_table() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=0.0, scale=0.2, size=(60, 2))
    cluster_b = rng.normal(loc=2.0, scale=0.2, size=(60, 2))
    data = np.vstack([cluster_a, cluster_b])
    df = pd.DataFrame(data, columns=["feat_a", "feat_b"])
    df["flat"] = 1.0  # low-variance column should be dropped automatically
    df["trial_id"] = [f"trial_{i}" for i in range(len(df))]
    df["fly_id"] = ["fly_0" if i < 60 else "fly_1" for i in range(len(df))]
    return df


def test_cluster_features_adaptive_selection() -> None:
    table = _make_feature_table()
    result = cluster_features(
        table,
        method="gmm",
        n_components=2,
        component_range=[1, 2, 3],
        covariance_types=["full", "diag"],
    )
    unique = np.unique(result.assignments)
    assert unique.size >= 2


def test_cluster_features_with_projection_matrix() -> None:
    table = _make_feature_table()
    # Collapse numeric features so clustering relies on projection matrix.
    table["feat_a"] = 0.0
    table["feat_b"] = 0.0
    projection_a = np.repeat(np.array([[0.0, 0.0]]), repeats=60, axis=0)
    projection_b = np.repeat(np.array([[3.0, 3.0]]), repeats=60, axis=0)
    projections = np.vstack([projection_a, projection_b])
    projection_matrix = projections.reshape(len(table), -1)
    result = cluster_features(
        table,
        method="gmm",
        n_components=2,
        projection_matrix=projection_matrix,
        combine_projection=False,
    )
    unique = np.unique(result.assignments)
    assert unique.size == 2

"""Clustering utilities for flypca."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

LOGGER = logging.getLogger(__name__)

try:
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover
    hdbscan = None


@dataclass
class ClusterResult:
    assignments: np.ndarray
    metrics: Dict[str, float]
    model: object


def _prepare_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_df.fillna(numeric_df.mean(), inplace=True)
    numeric_df.fillna(0.0, inplace=True)
    return numeric_df.to_numpy(dtype=float)


def cluster_features(
    features: pd.DataFrame,
    method: str = "gmm",
    n_components: int = 2,
    random_state: int = 0,
) -> ClusterResult:
    """Cluster feature table and compute unsupervised metrics."""

    X = _prepare_matrix(features)
    if method == "gmm":
        model = GaussianMixture(n_components=n_components, covariance_type="full", random_state=random_state)
        assignments = model.fit_predict(X)
    elif method == "hdbscan":
        if hdbscan is None:
            raise ImportError("hdbscan is not installed.")
        model = hdbscan.HDBSCAN(min_cluster_size=max(5, n_components))
        assignments = model.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering method {method}")
    silhouette = metrics.silhouette_score(X, assignments) if len(set(assignments)) > 1 else float("nan")
    calinski = metrics.calinski_harabasz_score(X, assignments) if len(set(assignments)) > 1 else float("nan")
    return ClusterResult(
        assignments=assignments,
        metrics={
            "silhouette": float(silhouette),
            "calinski_harabasz": float(calinski),
        },
        model=model,
    )


def evaluate_with_labels(
    features: pd.DataFrame,
    labels: pd.Series,
    fly_ids: pd.Series,
    n_folds: int = 5,
) -> Dict[str, float]:
    """Evaluate AUROC/AUPRC with leave-one-fly-out cross-validation."""

    df = features.copy()
    df["label"] = labels.to_numpy()
    df["fly_id"] = fly_ids.to_numpy()
    unique_flies = df["fly_id"].unique()
    if unique_flies.size < 2:
        raise ValueError("At least two flies required for evaluation.")
    folds = min(n_folds, unique_flies.size)
    splits = np.array_split(unique_flies, folds)
    y_scores: list[np.ndarray] = []
    y_true: list[np.ndarray] = []
    for holdout in splits:
        train_mask = ~df["fly_id"].isin(holdout)
        test_mask = df["fly_id"].isin(holdout)
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        X_train = _prepare_matrix(df.loc[train_mask].drop(columns=["label", "fly_id"]))
        X_test = _prepare_matrix(df.loc[test_mask].drop(columns=["label", "fly_id"]))
        y_train = df.loc[train_mask, "label"].to_numpy()
        y_test = df.loc[test_mask, "label"].to_numpy()
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
        ])
        pipeline.fit(X_train, y_train)
        y_scores.append(pipeline.predict_proba(X_test)[:, 1])
        y_true.append(y_test)
    if not y_scores:
        raise ValueError("No folds evaluated; check data balance.")
    y_scores_concat = np.concatenate(y_scores)
    y_true_concat = np.concatenate(y_true)
    auroc = metrics.roc_auc_score(y_true_concat, y_scores_concat)
    auprc = metrics.average_precision_score(y_true_concat, y_scores_concat)
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
    }

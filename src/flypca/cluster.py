"""Clustering utilities for flypca."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def _drop_low_variance(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if threshold <= 0:
        return df
    variances = df.var(axis=0)
    keep_mask = variances > threshold
    dropped = df.columns[~keep_mask]
    if len(dropped) > 0:
        LOGGER.debug("Dropping low-variance columns: %s", list(dropped))
    filtered = df.loc[:, keep_mask]
    if filtered.empty:
        raise ValueError("All features removed by variance threshold; lower min_variance.")
    return filtered


def _prepare_matrix(df: pd.DataFrame) -> np.ndarray:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_df.fillna(numeric_df.mean(), inplace=True)
    numeric_df.fillna(0.0, inplace=True)
    return numeric_df.to_numpy(dtype=float)


def _prepare_feature_matrix(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    min_variance: float = 1e-6,
) -> Tuple[np.ndarray, List[str]]:
    if columns is not None:
        numeric_df = df.loc[:, list(columns)].copy()
    else:
        numeric_df = df.select_dtypes(include=[np.number]).copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors="coerce")
    numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_df.fillna(numeric_df.mean(), inplace=True)
    numeric_df.fillna(0.0, inplace=True)
    numeric_df = _drop_low_variance(numeric_df, min_variance)
    return numeric_df.to_numpy(dtype=float), numeric_df.columns.tolist()


def _standardize_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(matrix), scaler


def _iterate_component_grid(
    X: np.ndarray,
    component_values: Iterable[int],
    covariance_types: Iterable[str],
    random_state: int,
) -> List[Tuple[GaussianMixture, np.ndarray, float, float, int, str]]:
    candidates: List[Tuple[GaussianMixture, np.ndarray, float, float, int, str]] = []
    for cov_type in covariance_types:
        for n_components in component_values:
            if n_components < 1:
                continue
            model = GaussianMixture(
                n_components=n_components,
                covariance_type=cov_type,
                random_state=random_state,
            )
            model.fit(X)
            assignments = model.predict(X)
            unique = np.unique(assignments).size
            try:
                silhouette = (
                    metrics.silhouette_score(X, assignments)
                    if unique > 1
                    else float("nan")
                )
            except ValueError:
                silhouette = float("nan")
            bic = model.bic(X)
            candidates.append((model, assignments, bic, float(silhouette), n_components, cov_type))
    if not candidates:
        raise ValueError("No valid GMM candidates evaluated; adjust component range.")
    return candidates


def _select_best_candidate(
    candidates: List[Tuple[GaussianMixture, np.ndarray, float, float, int, str]]
) -> Tuple[GaussianMixture, np.ndarray, float, float, int, str]:
    # Prefer the lowest BIC; if ties, pick the one with higher silhouette.
    candidates.sort(key=lambda item: (item[2], -np.nan_to_num(item[3], nan=-np.inf)))
    best = candidates[0]
    # If best collapses to a single cluster, try to find an alternative with multiple clusters.
    if np.unique(best[1]).size > 1:
        return best
    for cand in candidates[1:]:
        if np.unique(cand[1]).size > 1 and not np.isnan(cand[3]):
            return cand
    # Fall back to the original best even if it is degenerate.
    return best


def cluster_features(
    features: pd.DataFrame,
    method: str = "gmm",
    n_components: int = 2,
    random_state: int = 0,
    feature_columns: Optional[Sequence[str]] = None,
    min_variance: float = 1e-6,
    standardize: bool = True,
    component_range: Optional[Sequence[int]] = None,
    covariance_types: Optional[Sequence[str]] = None,
    projection_matrix: Optional[np.ndarray] = None,
    combine_projection: bool = False,
) -> ClusterResult:
    """Cluster feature table and compute unsupervised metrics."""

    X_features: Optional[np.ndarray] = None
    if projection_matrix is None or combine_projection:
        X_features, _ = _prepare_feature_matrix(
            features,
            columns=feature_columns,
            min_variance=min_variance,
        )
    if projection_matrix is not None:
        if combine_projection:
            if X_features is None:
                raise ValueError("combine_projection requires feature matrix")
            if projection_matrix.shape[0] != X_features.shape[0]:
                raise ValueError("Projection matrix rows must match feature rows")
            X = np.concatenate([X_features, projection_matrix], axis=1)
        else:
            X = projection_matrix
    else:
        if X_features is None:
            raise ValueError("No features available for clustering")
        X = X_features
    if X.size == 0:
        raise ValueError("Empty feature matrix after preprocessing")
    if standardize:
        X, _ = _standardize_matrix(X)
    unique_assignments: int
    if method == "gmm":
        covariance_options: Sequence[str] = covariance_types or ("full",)
        if component_range is not None:
            component_values = sorted({int(v) for v in component_range if int(v) > 0})
        else:
            component_values = [int(n_components)]
        candidates = _iterate_component_grid(
            X,
            component_values,
            covariance_options,
            random_state=random_state,
        )
        model, assignments, bic, silhouette, best_components, best_cov = _select_best_candidate(candidates)
        LOGGER.info(
            "Selected GMM with n_components=%d covariance=%s (BIC=%.3f, silhouette=%.3f)",
            best_components,
            best_cov,
            bic,
            silhouette,
        )
        unique_assignments = np.unique(assignments).size
    elif method == "hdbscan":
        if hdbscan is None:
            raise ImportError("hdbscan is not installed.")
        model = hdbscan.HDBSCAN(min_cluster_size=max(5, n_components))
        assignments = model.fit_predict(X)
        unique_assignments = np.unique(assignments).size
    else:
        raise ValueError(f"Unknown clustering method {method}")
    silhouette = (
        metrics.silhouette_score(X, assignments)
        if unique_assignments > 1
        else float("nan")
    )
    calinski = (
        metrics.calinski_harabasz_score(X, assignments)
        if unique_assignments > 1
        else float("nan")
    )
    LOGGER.debug(
        "Clustering complete with %d unique assignments; silhouette=%.3f calinski=%.3f",
        unique_assignments,
        silhouette,
        calinski,
    )
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

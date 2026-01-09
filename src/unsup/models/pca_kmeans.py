"""PCA + k-means clustering pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from ..metrics import compute_ari, compute_silhouette
from ..pca_core import PCAResults


@dataclass
class ModelOutputs:
    labels: np.ndarray
    metrics: Dict[str, float | int | None]


def _candidate_range(n_samples: int, min_clusters: int, max_clusters: int) -> range:
    if n_samples <= 1:
        return range(1, min(2, n_samples + 1))

    upper = max(2, min(max_clusters, n_samples))
    lower = max(2, min(min_clusters, upper))
    if lower > upper:
        lower = upper
    return range(lower, upper + 1)


def _select_kmeans(
    embedding: np.ndarray,
    seed: Optional[int],
    min_clusters: int,
    max_clusters: int,
) -> tuple[np.ndarray, int]:
    candidates = _candidate_range(embedding.shape[0], min_clusters, max_clusters)

    best_score = -np.inf
    best_labels: Optional[np.ndarray] = None
    best_k = max(1, min_clusters)

    for k in candidates:
        clusterer = KMeans(n_clusters=k, random_state=seed)
        labels = clusterer.fit_predict(embedding)
        if np.unique(labels).size < 2:
            continue
        score = silhouette_score(embedding, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    if best_labels is None:
        fallback_k = min(max(1, min_clusters), embedding.shape[0])
        clusterer = KMeans(n_clusters=fallback_k, random_state=seed)
        best_labels = clusterer.fit_predict(embedding)
        best_k = fallback_k

    return best_labels, best_k


def run_model(
    pca_results: PCAResults,
    dataset_labels: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    min_clusters: int = 2,
    max_clusters: int = 10,
) -> ModelOutputs:
    pcs_to_use = pca_results.pcs_80pct or 1
    embedding = pca_results.scores[:, :pcs_to_use]

    if embedding.shape[0] == 0:
        raise ValueError("No samples available for k-means clustering.")

    labels, selected_k = _select_kmeans(
        embedding, seed=seed, min_clusters=min_clusters, max_clusters=max_clusters
    )

    silhouette = compute_silhouette(embedding, labels)
    ari = compute_ari(labels, dataset_labels)

    metrics_dict: Dict[str, float | int | None] = {
        "n_clusters": int(len(np.unique(labels))),
        "selected_k": int(selected_k),
        "noise_fraction": 0.0,
        "silhouette": silhouette,
        "ARI_vs_true": ari,
        "logreg_cv_acc": None,
    }
    return ModelOutputs(labels=labels, metrics=metrics_dict)


__all__ = ["run_model", "ModelOutputs"]

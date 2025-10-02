"""PCA + density-based clustering pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import DBSCAN

from ..metrics import compute_ari, compute_silhouette
from ..pca_core import PCAResults

try:  # Optional dependency
    import hdbscan  # type: ignore
except ImportError:  # pragma: no cover
    hdbscan = None


@dataclass
class ModelOutputs:
    labels: np.ndarray
    metrics: Dict[str, float | int | None]


def run_model(
    pca_results: PCAResults,
    dataset_labels: Optional[np.ndarray] = None,
    min_cluster_size: int = 5,
    seed: Optional[int] = None,
) -> ModelOutputs:
    pcs_to_use = pca_results.pcs_80pct or 1
    embedding = pca_results.scores[:, :pcs_to_use]

    if hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=False)
        labels = clusterer.fit_predict(embedding)
    else:
        clusterer = DBSCAN(min_samples=min_cluster_size)
        labels = clusterer.fit_predict(embedding)

    silhouette = compute_silhouette(embedding, labels)
    ari = compute_ari(labels, dataset_labels)

    unique_labels = np.unique(labels)
    cluster_labels = unique_labels[unique_labels != -1]
    noise_fraction = float(np.mean(labels == -1))

    metrics_dict: Dict[str, float | int | None] = {
        "n_clusters": int(cluster_labels.size),
        "noise_fraction": noise_fraction,
        "silhouette": silhouette,
        "ARI_vs_true": ari,
        "logreg_cv_acc": None,
    }

    return ModelOutputs(labels=labels, metrics=metrics_dict)


__all__ = ["run_model", "ModelOutputs"]

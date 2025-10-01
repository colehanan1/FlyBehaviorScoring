"""PCA + k-means clustering pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans

from ..metrics import compute_ari, compute_silhouette
from ..pca_core import PCAResults


@dataclass
class ModelOutputs:
    labels: np.ndarray
    metrics: Dict[str, float | int | None]


def run_model(
    pca_results: PCAResults,
    dataset_labels: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> ModelOutputs:
    pcs_to_use = pca_results.pcs_80pct or 1
    embedding = pca_results.scores[:, :pcs_to_use]

    clusterer = KMeans(n_clusters=2, random_state=seed)
    labels = clusterer.fit_predict(embedding)

    silhouette = compute_silhouette(embedding, labels)
    ari = compute_ari(labels, dataset_labels)

    metrics_dict: Dict[str, float | int | None] = {
        "n_clusters": int(len(np.unique(labels))),
        "noise_fraction": 0.0,
        "silhouette": silhouette,
        "ARI_vs_true": ari,
        "logreg_cv_acc": None,
    }
    return ModelOutputs(labels=labels, metrics=metrics_dict)


__all__ = ["run_model", "ModelOutputs"]

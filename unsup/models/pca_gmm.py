"""PCA + Gaussian mixture clustering pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.mixture import GaussianMixture

from ..metrics import compute_ari, compute_silhouette, logistic_probe_cv
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

    gmm = GaussianMixture(n_components=2, random_state=seed)
    gmm.fit(embedding)
    labels = gmm.predict(embedding)

    silhouette = compute_silhouette(embedding, labels)
    ari = compute_ari(labels, dataset_labels)
    logreg_acc = None
    if dataset_labels is not None and np.unique(dataset_labels).size == 2:
        logreg_acc = logistic_probe_cv(embedding, dataset_labels, seed=seed)

    metrics_dict: Dict[str, float | int | None] = {
        "n_clusters": int(len(np.unique(labels))),
        "noise_fraction": 0.0,
        "silhouette": silhouette,
        "ARI_vs_true": ari,
        "logreg_cv_acc": logreg_acc,
    }

    return ModelOutputs(labels=labels, metrics=metrics_dict)


__all__ = ["run_model", "ModelOutputs"]

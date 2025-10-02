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


def _candidate_range(n_samples: int, min_components: int, max_components: int) -> range:
    if n_samples == 0:
        return range(0)

    upper = max(1, min(max_components, n_samples))
    lower = max(1, min(min_components, upper))
    if lower > upper:
        lower = upper
    return range(lower, upper + 1)


def _select_gmm(
    embedding: np.ndarray,
    seed: Optional[int],
    min_components: int,
    max_components: int,
) -> tuple[GaussianMixture, np.ndarray]:
    candidates = _candidate_range(embedding.shape[0], min_components, max_components)

    best_bic = np.inf
    best_model: Optional[GaussianMixture] = None
    best_labels: Optional[np.ndarray] = None

    for n_components in candidates:
        model = GaussianMixture(n_components=n_components, random_state=seed)
        model.fit(embedding)
        bic = model.bic(embedding)
        labels = model.predict(embedding)
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_labels = labels

    if best_model is None or best_labels is None:
        fallback_components = max(1, min_components)
        best_model = GaussianMixture(n_components=fallback_components, random_state=seed)
        best_model.fit(embedding)
        best_labels = best_model.predict(embedding)

    return best_model, best_labels


def run_model(
    pca_results: PCAResults,
    dataset_labels: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    min_components: int = 2,
    max_components: int = 10,
) -> ModelOutputs:
    pcs_to_use = pca_results.pcs_80pct or 1
    embedding = pca_results.scores[:, :pcs_to_use]

    if embedding.shape[0] == 0:
        raise ValueError("No samples available for Gaussian mixture clustering.")

    model, labels = _select_gmm(
        embedding,
        seed=seed,
        min_components=min_components,
        max_components=max_components,
    )

    silhouette = compute_silhouette(embedding, labels)
    ari = compute_ari(labels, dataset_labels)
    logreg_acc = None
    if dataset_labels is not None and np.unique(dataset_labels).size == 2:
        logreg_acc = logistic_probe_cv(embedding, dataset_labels, seed=seed)

    metrics_dict: Dict[str, float | int | None] = {
        "n_clusters": int(len(np.unique(labels))),
        "selected_components": int(model.n_components),
        "noise_fraction": 0.0,
        "silhouette": silhouette,
        "ARI_vs_true": ari,
        "logreg_cv_acc": logreg_acc,
    }

    return ModelOutputs(labels=labels, metrics=metrics_dict)


__all__ = ["run_model", "ModelOutputs"]

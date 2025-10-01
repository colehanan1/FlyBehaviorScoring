"""Evaluation metrics for unsupervised models."""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def compute_silhouette(embedding: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute silhouette score, excluding noise labels."""

    mask = labels != -1
    unique_labels = np.unique(labels[mask]) if mask.any() else np.array([])
    if unique_labels.size < 2:
        return None
    return float(metrics.silhouette_score(embedding[mask], labels[mask]))


def compute_ari(pred_labels: np.ndarray, true_labels: Optional[np.ndarray]) -> float | None:
    if true_labels is None:
        return None
    if np.unique(true_labels).size < 2:
        return None
    return float(metrics.adjusted_rand_score(true_labels, pred_labels))


def logistic_probe_cv(
    features: np.ndarray,
    labels: np.ndarray,
    seed: Optional[int] = None,
) -> float:
    """Evaluate linear separability with logistic regression."""

    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(classifier, features, labels, cv=cv, scoring="accuracy")
    return float(np.mean(scores))


__all__ = ["compute_silhouette", "compute_ari", "logistic_probe_cv"]

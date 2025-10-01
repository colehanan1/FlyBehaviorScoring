"""Principal component analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class PCAResults:
    """Bundle PCA outputs for downstream models and reporting."""

    pca: PCA
    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    pcs_80pct: int
    pcs_90pct: int

    @property
    def components(self) -> np.ndarray:
        return self.pca.components_


def _select_components(cumulative: np.ndarray, threshold: float, cap: int) -> int:
    above = np.where(cumulative >= threshold)[0]
    if above.size == 0:
        return min(cap, cumulative.size)
    return min(cap, int(above[0] + 1))


def compute_pca(
    X: np.ndarray,
    max_pcs: int = 10,
    random_state: int | None = None,
) -> PCAResults:
    """Fit PCA and determine component counts for analysis."""

    n_components = min(max_pcs, X.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    scores = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    pcs_80 = _select_components(cumulative, 0.80, cap=max_pcs)
    pcs_90 = _select_components(cumulative, 0.90, cap=min(5, n_components))

    return PCAResults(
        pca=pca,
        scores=scores,
        explained_variance_ratio=explained,
        cumulative_variance=cumulative,
        pcs_80pct=pcs_80,
        pcs_90pct=pcs_90,
    )


def compute_time_importance(pca_results: PCAResults, time_columns: Sequence[str]) -> pd.DataFrame:
    """Calculate mean absolute loadings across important PCs."""

    if pca_results.pcs_90pct == 0:
        raise ValueError("Unable to determine PCs covering 90% variance.")

    top_components = pca_results.components[: pca_results.pcs_90pct]
    importance = np.mean(np.abs(top_components), axis=0)

    return pd.DataFrame({
        "time_index": [int(col.split("_")[-1]) for col in time_columns],
        "importance": importance,
    })


__all__ = ["PCAResults", "compute_pca", "compute_time_importance"]

"""Visualization utilities for flypca."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lagpca import LagPCAResult

LOGGER = logging.getLogger(__name__)


def _ensure_directory(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def scree_plot(result: LagPCAResult, out_dir: str | Path) -> Path:
    out_dir = _ensure_directory(out_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(result.explained_variance_ratio_) + 1), result.explained_variance_ratio_, marker="o")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Lag-PCA Scree Plot")
    path = out_dir / "scree_plot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved scree plot to %s", path)
    return path


def pc_loadings_plot(result: LagPCAResult, out_dir: str | Path, n_loadings: int = 10) -> Path:
    out_dir = _ensure_directory(out_dir)
    loadings = result.model.components_[:n_loadings]
    fig, axes = plt.subplots(len(loadings), 1, figsize=(6, 2 * len(loadings)), sharex=True)
    if len(loadings) == 1:
        axes = [axes]
    for idx, (ax, loading) in enumerate(zip(axes, loadings)):
        ax.plot(loading)
        ax.set_ylabel(f"PC{idx+1}")
    axes[-1].set_xlabel("Lag Index")
    fig.tight_layout()
    path = out_dir / "pc_loadings.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved loadings plot to %s", path)
    return path


def pc_trajectories_plot(
    trajectories: Iterable[tuple[np.ndarray, np.ndarray]],
    out_dir: str | Path,
    n_examples: int = 5,
) -> Path:
    out_dir = _ensure_directory(out_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, (time, pcs) in enumerate(trajectories):
        if idx >= n_examples:
            break
        ax.plot(time, pcs[:, 0], alpha=0.7)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PC1")
    ax.set_title("Example PC1 Trajectories")
    fig.tight_layout()
    path = out_dir / "pc_trajectories.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved trajectory plot to %s", path)
    return path


AssignmentInput = Union[np.ndarray, Sequence[int | float], pd.Series]


def _normalize_assignments(assignments: AssignmentInput, expected_length: int) -> np.ndarray:
    """Convert cluster assignments to a NumPy array with validation."""

    if isinstance(assignments, pd.Series):
        values = assignments.to_numpy()
    else:
        values = np.asarray(assignments)
    if values.size != expected_length:
        raise ValueError(
            f"Assignment count {values.size} does not match feature rows {expected_length}"
        )
    return values


def pc_scatter(features: pd.DataFrame, assignments: AssignmentInput, out_dir: str | Path) -> Path:
    out_dir = _ensure_directory(out_dir)
    fig, ax = plt.subplots(figsize=(6, 5))
    lower_map = {c.lower(): c for c in features.columns}
    numeric = features.select_dtypes(include=[np.number])
    if "pc1" in lower_map and "pc2" in lower_map:
        x = features[lower_map["pc1"]]
        y = features[lower_map["pc2"]]
    elif numeric.shape[1] >= 2:
        x = numeric.iloc[:, 0]
        y = numeric.iloc[:, 1]
    else:
        raise ValueError("PC scatter requires at least two numeric feature columns.")
    assignment_array = _normalize_assignments(assignments, len(features))
    scatter = ax.scatter(x, y, c=assignment_array, cmap="viridis", alpha=0.8)
    ax.set_xlabel(x.name)
    ax.set_ylabel(y.name)
    ax.set_title("PC Scatter with Clusters")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    path = out_dir / "pc_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved scatter plot to %s", path)
    return path


def feature_violin(
    features: pd.DataFrame,
    assignments: AssignmentInput,
    columns: Optional[Iterable[str]],
    out_dir: str | Path,
) -> Path:
    out_dir = _ensure_directory(out_dir)
    cols = list(columns) if columns is not None else [c for c in features.columns if c not in {"trial_id", "fly_id"}]
    fig, axes = plt.subplots(len(cols), 1, figsize=(6, 2 * len(cols)))
    if len(cols) == 1:
        axes = [axes]
    assignment_array = _normalize_assignments(assignments, len(features))
    for ax, col in zip(axes, cols):
        data = [
            features.loc[assignment_array == cluster, col].dropna()
            for cluster in np.unique(assignment_array)
        ]
        ax.violinplot(data, showmeans=True)
        ax.set_title(col)
    axes[-1].set_xlabel("Cluster")
    fig.tight_layout()
    path = out_dir / "feature_violins.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    LOGGER.info("Saved violin plots to %s", path)
    return path

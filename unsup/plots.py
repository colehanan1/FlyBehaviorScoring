"""Plotting utilities for unsupervised analysis."""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pca_core import PCAResults


ODOR_ON_FRAME = 1230
ODOR_OFF_FRAME = 2430


def plot_variance(pca_results: PCAResults, path: str) -> None:
    indices = np.arange(1, len(pca_results.explained_variance_ratio) + 1)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(indices, pca_results.explained_variance_ratio, alpha=0.7, label="Explained variance")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance ratio")
    ax1.set_title("PCA explained variance")

    ax2 = ax1.twinx()
    ax2.plot(indices, pca_results.cumulative_variance, color="black", marker="o", label="Cumulative")
    ax2.set_ylabel("Cumulative variance")
    ax2.set_ylim(0, 1.05)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_time_importance(importance_df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(importance_df["time_index"], importance_df["importance"], marker="o")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Mean |loading|")
    ax.set_title("Timepoint importance")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_embedding(
    embedding: np.ndarray,
    labels: Sequence[int],
    path: str,
) -> None:
    if embedding.shape[1] < 2:
        padding = np.zeros((embedding.shape[0], 2 - embedding.shape[1]))
        embedding = np.hstack([embedding, padding])
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = np.asarray(labels)
    mask_noise = labels == -1
    unique_labels = np.unique(labels[~mask_noise])

    if unique_labels.size == 0:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=20, c="grey", alpha=0.7)
    else:
        colors = plt.cm.get_cmap("tab10", unique_labels.size)
        for idx, label in enumerate(unique_labels):
            subset = labels == label
            ax.scatter(
                embedding[subset, 0],
                embedding[subset, 1],
                s=30,
                color=colors(idx),
                label=f"Cluster {label}",
                alpha=0.8,
            )
        if mask_noise.any():
            ax.scatter(
                embedding[mask_noise, 0],
                embedding[mask_noise, 1],
                s=20,
                color="lightgrey",
                label="Noise",
                alpha=0.6,
                marker="x",
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PC1 vs PC2 embedding")
    if unique_labels.size > 0 or mask_noise.any():
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_components(time_points: np.ndarray, components: np.ndarray, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, pattern in enumerate(components):
        ax.plot(time_points, pattern, label=f"Component {idx+1}")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Component magnitude")
    ax.set_title("Odor-aligned motifs")
    if components.shape[0] > 1:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_cluster_traces(
    time_points: Sequence[float],
    traces: np.ndarray,
    labels: Sequence[int],
    path: str,
    odor_on: int = ODOR_ON_FRAME,
    odor_off: int = ODOR_OFF_FRAME,
) -> None:
    time_points = np.asarray(time_points)
    labels = np.asarray(labels)
    if traces.ndim != 2:
        raise ValueError("traces must be a 2D array of shape (n_trials, n_timepoints)")
    if traces.shape[0] != labels.shape[0]:
        raise ValueError("labels length must match number of trials")

    fig, ax = plt.subplots(figsize=(6, 4))

    unique_labels = np.unique(labels)
    plotted = False
    cluster_labels = [label for label in unique_labels if label != -1]

    for label in sorted(cluster_labels):
        mask = labels == label
        if not np.any(mask):
            continue
        mean_trace = traces[mask].mean(axis=0)
        ax.plot(time_points, mean_trace, linewidth=1.8, label=f"Cluster {label}")
        plotted = True

    if np.any(labels == -1):
        noise_mean = traces[labels == -1].mean(axis=0)
        ax.plot(
            time_points,
            noise_mean,
            color="grey",
            linewidth=1.5,
            linestyle="--",
            label="Noise",
        )
        plotted = True

    if not plotted:
        overall_mean = traces.mean(axis=0)
        ax.plot(time_points, overall_mean, color="black", linewidth=1.8, label="All trials")

    for frame in (odor_on, odor_off):
        if time_points.size and time_points.min() <= frame <= time_points.max():
            ax.axvline(frame, color="red", linestyle="--", linewidth=1.2)

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Mean z-scored dir_val")
    ax.set_title("Average cluster traces")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


__all__ = [
    "plot_variance",
    "plot_time_importance",
    "plot_embedding",
    "plot_components",
    "plot_cluster_traces",
]

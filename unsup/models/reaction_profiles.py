"""Odor-aligned reaction profiling models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import StandardScaler

from ..metrics import compute_ari, compute_silhouette, logistic_probe_cv


ODOR_ON_FRAME = 1230
ODOR_OFF_FRAME = 2430
MOTIF_BIN_SIZE = 30
MAX_MOTIF_COMPONENTS = 5


@dataclass
class ReactionModelOutputs:
    labels: np.ndarray
    metrics: Dict[str, float | int | None]
    embedding: np.ndarray
    feature_names: list[str]
    features: np.ndarray
    component_time: Optional[np.ndarray] = None
    components: Optional[np.ndarray] = None
    component_weights: Optional[np.ndarray] = None


def _slice(traces: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, min(start, traces.shape[1]))
    end = max(start, min(end, traces.shape[1]))
    return traces[:, start:end]


def _safe_stat(segment: np.ndarray, reducer) -> np.ndarray:
    if segment.size == 0:
        length = segment.shape[0] if segment.ndim > 0 else 0
        return np.zeros(length, dtype=float)
    return reducer(segment)


def _segment_statistics(traces: np.ndarray, odor_on: int, odor_off: int) -> dict[str, np.ndarray]:
    baseline = _slice(traces, 0, odor_on)
    response = _slice(traces, odor_on, odor_off)
    recovery = _slice(traces, odor_off, traces.shape[1])

    stats: dict[str, np.ndarray] = {}

    stats["baseline_mean"] = _safe_stat(baseline, lambda arr: arr.mean(axis=1))
    stats["baseline_std"] = _safe_stat(baseline, lambda arr: arr.std(axis=1))

    stats["odor_mean"] = _safe_stat(response, lambda arr: arr.mean(axis=1))
    stats["odor_std"] = _safe_stat(response, lambda arr: arr.std(axis=1))
    stats["odor_max"] = _safe_stat(response, lambda arr: arr.max(axis=1))
    if response.size:
        peak_indices = np.argmax(response, axis=1)
        stats["odor_peak_latency"] = peak_indices.astype(float) / max(response.shape[1], 1)
        stats["odor_auc"] = np.trapz(response, axis=1)
    else:
        stats["odor_peak_latency"] = np.zeros(traces.shape[0])
        stats["odor_auc"] = np.zeros(traces.shape[0])

    stats["post_mean"] = _safe_stat(recovery, lambda arr: arr.mean(axis=1))
    stats["post_std"] = _safe_stat(recovery, lambda arr: arr.std(axis=1))
    stats["post_min"] = _safe_stat(recovery, lambda arr: arr.min(axis=1))
    stats["post_auc"] = (
        np.trapz(recovery, axis=1) if recovery.size else np.zeros(traces.shape[0])
    )

    stats["response_delta"] = stats["odor_mean"] - stats["baseline_mean"]
    stats["peak_delta"] = stats["odor_max"] - stats["baseline_mean"]
    stats["recovery_delta"] = stats["post_mean"] - stats["baseline_mean"]
    stats["post_vs_odor"] = stats["post_mean"] - stats["odor_mean"]
    stats["post_auc_delta"] = stats["post_auc"] - stats["odor_auc"]

    return stats


def _aggregate_window(traces: np.ndarray, start: int, end: int, bin_size: int) -> tuple[np.ndarray, np.ndarray]:
    window = _slice(traces, start, end)
    if window.size == 0:
        return np.empty((traces.shape[0], 0)), np.empty(0)

    n_bins = int(np.ceil(window.shape[1] / bin_size))
    aggregated = np.zeros((traces.shape[0], n_bins), dtype=float)
    centers = np.zeros(n_bins, dtype=float)

    for idx in range(n_bins):
        s = idx * bin_size
        e = min((idx + 1) * bin_size, window.shape[1])
        aggregated[:, idx] = window[:, s:e].mean(axis=1)
        centers[idx] = start + (s + e - 1) / 2.0

    return aggregated, centers


def _select_kmeans(features: np.ndarray, seed: Optional[int], min_clusters: int, max_clusters: int) -> tuple[np.ndarray, int]:
    n_samples = features.shape[0]
    if n_samples <= 1:
        labels = np.zeros(n_samples, dtype=int)
        return labels, 1

    lower = max(2, min(min_clusters, n_samples))
    upper = max(lower, min(max_clusters, n_samples))

    best_labels: Optional[np.ndarray] = None
    best_score = -np.inf
    best_k = lower

    for k in range(lower, upper + 1):
        clusterer = KMeans(n_clusters=k, random_state=seed)
        labels = clusterer.fit_predict(features)
        if np.unique(labels).size < 2:
            continue
        score = compute_silhouette(features, labels)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    if best_labels is None:
        clusterer = KMeans(n_clusters=lower, random_state=seed)
        best_labels = clusterer.fit_predict(features)
        best_k = lower

    return best_labels, best_k


def _cluster_reactions(
    traces: np.ndarray,
    dataset_labels: Optional[np.ndarray],
    seed: Optional[int],
    min_clusters: int,
    max_clusters: int,
) -> tuple[ReactionModelOutputs, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    odor_on = min(max(0, ODOR_ON_FRAME), traces.shape[1])
    odor_off = min(max(odor_on + 1, ODOR_OFF_FRAME), traces.shape[1])

    stats = _segment_statistics(traces, odor_on, odor_off)
    feature_names = list(stats.keys())
    feature_matrix = np.column_stack([stats[name] for name in feature_names])
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    labels, selected_k = _select_kmeans(scaled, seed, min_clusters, max_clusters)

    silhouette = compute_silhouette(scaled, labels)
    ari = compute_ari(labels, dataset_labels)

    metrics: Dict[str, float | int | None] = {
        "n_clusters": int(np.unique(labels).size),
        "selected_k": int(selected_k),
        "noise_fraction": 0.0,
        "silhouette": silhouette,
        "ARI_vs_true": ari,
        "logreg_cv_acc": None,
    }

    if dataset_labels is not None and np.unique(dataset_labels).size == 2:
        metrics["logreg_cv_acc"] = logistic_probe_cv(scaled, dataset_labels, seed=seed)

    projector = PCA(n_components=min(2, scaled.shape[1]), random_state=seed)
    embedding = projector.fit_transform(scaled) if scaled.shape[1] >= 1 else np.zeros((scaled.shape[0], 2))

    outputs = ReactionModelOutputs(
        labels=labels,
        metrics=metrics,
        embedding=embedding,
        feature_names=feature_names,
        features=scaled,
    )

    motif_start = max(0, odor_on - 300)
    motif_end = min(traces.shape[1], odor_off + 600)
    aggregated, centers = _aggregate_window(traces, motif_start, motif_end, MOTIF_BIN_SIZE)

    return outputs, stats, aggregated, centers


def run_model_with_motifs(
    traces: np.ndarray,
    *,
    dataset_labels: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    min_clusters: int = 2,
    max_clusters: int = 10,
) -> ReactionModelOutputs:
    outputs, _, aggregated, centers = _cluster_reactions(
        traces,
        dataset_labels,
        seed,
        min_clusters,
        max_clusters,
    )

    if aggregated.size == 0:
        return outputs

    shifted = aggregated - aggregated.min(axis=1, keepdims=True)
    shifted += 1e-6

    n_components = min(
        MAX_MOTIF_COMPONENTS,
        aggregated.shape[0],
        aggregated.shape[1],
    )
    n_components = max(1, n_components)

    model = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=seed,
        max_iter=1000,
    )
    weights = model.fit_transform(shifted)
    components = model.components_

    outputs.components = components
    outputs.component_time = centers
    outputs.component_weights = weights
    return outputs


def run_model_clusters_only(
    traces: np.ndarray,
    *,
    dataset_labels: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    min_clusters: int = 2,
    max_clusters: int = 10,
) -> ReactionModelOutputs:
    outputs, stats, aggregated, centers = _cluster_reactions(
        traces,
        dataset_labels,
        seed,
        min_clusters,
        max_clusters,
    )

    # Preserve scaled features from clustering for downstream inspection.
    outputs.feature_names = list(stats.keys())
    outputs.components = None
    outputs.component_time = centers if aggregated.size else None
    outputs.component_weights = None
    return outputs


__all__ = [
    "run_model_with_motifs",
    "run_model_clusters_only",
    "ReactionModelOutputs",
]

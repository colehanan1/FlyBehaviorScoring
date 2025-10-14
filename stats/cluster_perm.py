"""Cluster-based permutation testing for within-fly time-series."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp, wilcoxon

from . import plotting
from .utils import diff_matrix

LOG = logging.getLogger("stats.cluster_perm")


@dataclass
class Cluster:
    """Container describing a contiguous supra-threshold cluster."""

    cluster_id: int
    indices: np.ndarray
    sign: int
    mass: float
    p_cluster: float

    def to_row(self, time_s: np.ndarray) -> Dict[str, float]:
        start_idx = int(self.indices[0])
        end_idx = int(self.indices[-1])
        return {
            "cluster_id": int(self.cluster_id),
            "start_s": float(time_s[start_idx]),
            "end_s": float(time_s[end_idx]),
            "duration_s": float(time_s[end_idx] - time_s[start_idx]),
            "sign": int(self.sign),
            "mass": float(self.mass),
            "p_cluster": float(self.p_cluster),
        }


@dataclass
class ClusterPermutationResult:
    """Result bundle for the cluster-based permutation test."""

    stats: np.ndarray
    p_uncorrected: np.ndarray
    clusters: List[Cluster]
    cluster_ids: np.ndarray
    n_flies: int


def _compute_statistic(diff: np.ndarray, method: str) -> Tuple[np.ndarray, np.ndarray]:
    flies, timepoints = diff.shape
    stats = np.full(timepoints, np.nan, dtype=float)
    pvals = np.full(timepoints, np.nan, dtype=float)
    for t in range(timepoints):
        series = diff[:, t]
        series = series[np.isfinite(series)]
        if series.size < 2:
            continue
        if method == "t":
            stat, pval = ttest_1samp(series, 0.0, nan_policy="omit")
        elif method == "wilcoxon":
            if np.allclose(series, 0.0):
                continue
            try:
                stat, pval = wilcoxon(series, zero_method="wilcox", alternative="two-sided", correction=True)
            except ValueError:
                continue
        else:
            raise ValueError(f"Unsupported cluster statistic '{method}'.")
        stats[t] = stat
        pvals[t] = pval
    return stats, pvals


def _find_clusters(stats: np.ndarray, pvals: np.ndarray, alpha: float) -> List[Tuple[int, np.ndarray]]:
    mask = (pvals < alpha) & ~np.isnan(stats)
    clusters: List[Tuple[int, np.ndarray]] = []
    if not np.any(mask):
        return clusters
    for sign in (1, -1):
        signed_mask = mask & ((stats > 0) if sign > 0 else (stats < 0))
        start: Optional[int] = None
        for idx, flag in enumerate(signed_mask):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                clusters.append((sign, np.arange(start, idx)))
                start = None
        if start is not None:
            clusters.append((sign, np.arange(start, signed_mask.size)))
    clusters.sort(key=lambda item: item[1][0])
    return clusters


def _cluster_masses(stats: np.ndarray, clusters: List[Tuple[int, np.ndarray]]) -> List[Tuple[int, float, np.ndarray]]:
    masses: List[Tuple[int, float, np.ndarray]] = []
    for sign, idxs in clusters:
        if idxs.size == 0:
            continue
        mass = float(np.sum(np.abs(stats[idxs])))
        masses.append((sign, mass, idxs))
    return masses


def _max_cluster_mass(diff: np.ndarray, method: str, alpha: float) -> float:
    stats, pvals = _compute_statistic(diff, method)
    clusters = _find_clusters(stats, pvals, alpha)
    masses = _cluster_masses(stats, clusters)
    if not masses:
        return 0.0
    return max(mass for _, mass, _ in masses)


def _critical_threshold(method: str, n: int, alpha: float) -> float:
    if method == "t" and n > 1:
        return float(student_t.ppf(1 - alpha / 2.0, n - 1))
    if method == "wilcoxon" and n > 1:
        from scipy.stats import norm

        return float(norm.ppf(1 - alpha / 2.0))
    return 0.0


def cluster_permutation_test(
    groups: Sequence,
    time_s: np.ndarray,
    n_perm: int,
    alpha: float = 0.05,
    method: str = "t",
    rng: Optional[np.random.Generator] = None,
) -> ClusterPermutationResult:
    """Run a cluster-based permutation test across time."""

    if rng is None:
        rng = np.random.default_rng(0)
    diff = diff_matrix(groups)
    flies = diff.shape[0]
    if flies < 2:
        raise ValueError("Cluster permutation test requires at least two flies.")
    stats, pvals = _compute_statistic(diff, method)
    raw_clusters = _find_clusters(stats, pvals, alpha)
    masses = _cluster_masses(stats, raw_clusters)
    if method == "t":
        LOG.info("Cluster permutation using t-statistic across %d flies (%d permutations).", flies, n_perm)
    else:
        LOG.info(
            "Cluster permutation using Wilcoxon statistic across %d flies (%d permutations).",
            flies,
            n_perm,
        )
    null_max = np.zeros(n_perm, dtype=float)
    for perm in range(n_perm):
        signs = rng.choice([-1, 1], size=flies)
        perm_diff = diff * signs[:, None]
        null_max[perm] = _max_cluster_mass(perm_diff, method, alpha)
    null_max.sort()
    clusters: List[Cluster] = []
    cluster_ids = np.zeros_like(stats, dtype=float)
    for idx, (sign, mass, indices) in enumerate(masses, start=1):
        greater = np.sum(null_max >= mass)
        p_cluster = (1 + greater) / (1 + n_perm)
        clusters.append(Cluster(cluster_id=idx, indices=indices, sign=sign, mass=mass, p_cluster=p_cluster))
        cluster_ids[indices] = idx
    return ClusterPermutationResult(
        stats=stats,
        p_uncorrected=pvals,
        clusters=clusters,
        cluster_ids=cluster_ids,
        n_flies=flies,
    )


def save_outputs(
    result: ClusterPermutationResult,
    time_s: np.ndarray,
    out_dir: str,
    alpha: float,
    method: str,
) -> Tuple[str, str, str]:
    """Persist CSV and PNG outputs for the cluster permutation test."""

    import os
    import pandas as pd

    clusters_rows = [cluster.to_row(time_s) for cluster in result.clusters]
    clusters_csv = os.path.join(out_dir, "cluster_perm_clusters.csv")
    if clusters_rows:
        clusters_df = pd.DataFrame(clusters_rows)
    else:
        clusters_df = pd.DataFrame(
            columns=["cluster_id", "start_s", "end_s", "duration_s", "sign", "mass", "p_cluster"]
        )
    clusters_df.to_csv(clusters_csv, index=False)

    per_time = pd.DataFrame(
        {
            "time_s": time_s,
            "statistic": result.stats,
            "p_uncorrected": result.p_uncorrected,
            "cluster_id": result.cluster_ids,
            "p_cluster": np.nan,
        }
    )
    for cluster in result.clusters:
        per_time.loc[
            np.isclose(per_time["cluster_id"], cluster.cluster_id),
            "p_cluster"
        ] = cluster.p_cluster
    time_csv = os.path.join(out_dir, "cluster_perm_timewise.csv")
    per_time.to_csv(time_csv, index=False)

    threshold = _critical_threshold(method, result.n_flies, alpha)
    clusters_for_plot = [
        {"mask": np.isclose(result.cluster_ids, cluster.cluster_id), "sign": cluster.sign, "p_cluster": cluster.p_cluster}
        for cluster in result.clusters
    ]
    plot_path = os.path.join(out_dir, "cluster_perm_plot.png")
    plotting.plot_cluster_map(time_s, result.stats, threshold, clusters_for_plot, plot_path, alpha)
    return clusters_csv, time_csv, plot_path


__all__ = ["cluster_permutation_test", "save_outputs", "ClusterPermutationResult", "Cluster"]

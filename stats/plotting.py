"""Plotting utilities for the stats package."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter

from .utils import GROUP_A_LABEL, GROUP_B_LABEL

LOG = logging.getLogger("stats.plotting")


def _prepare_axes(figsize: Tuple[int, int] = (10, 4)) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    return fig, ax


def plot_p_q(
    time_s: np.ndarray,
    pvals: np.ndarray,
    qvals: np.ndarray,
    title: str,
    out_path: str,
    alpha: float = 0.05,
    highlight_sign: Optional[np.ndarray] = None,
) -> None:
    """Plot p- and q-values across time, optionally marking significant regions."""

    fig, ax = _prepare_axes()
    ax.plot(time_s, pvals, label="p-value")
    ax.plot(time_s, qvals, label="BH q-value", linestyle="--")
    ax.axhline(alpha, linestyle=":", color="red", alpha=0.6, label=f"alpha={alpha}")
    if highlight_sign is not None and highlight_sign.size == time_s.size:
        above = highlight_sign > 0
        below = highlight_sign < 0
        for mask, color in ((above, "green"), (below, "purple")):
            if np.any(mask):
                ax.scatter(
                    time_s[mask],
                    np.minimum(pvals[mask], alpha * 0.8),
                    s=10,
                    color=color,
                    alpha=0.9,
                    label="median>0" if color == "green" else "median<0",
                )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_effect_curve(time_s: np.ndarray, effect: np.ndarray, title: str, out_path: str) -> None:
    """Plot an effect size curve."""

    fig, ax = _prepare_axes()
    ax.plot(time_s, effect, label="A - B (mean across flies)")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Effect (A - B)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_discordant_counts(
    time_s: np.ndarray,
    a_gt_b: np.ndarray,
    b_gt_a: np.ndarray,
    out_path: str,
) -> None:
    """Plot discordant counts for McNemar analysis."""

    fig, ax = _prepare_axes()
    ax.plot(time_s, a_gt_b, label=f"{GROUP_A_LABEL} > {GROUP_B_LABEL}")
    ax.plot(time_s, b_gt_a, label=f"{GROUP_B_LABEL} > {GROUP_A_LABEL}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Count (flies)")
    ax.set_title("Discordant counts per timepoint")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cluster_map(
    time_s: np.ndarray,
    stats: np.ndarray,
    threshold: float,
    clusters: Sequence[Dict[str, Any]],
    out_path: str,
    alpha: float,
) -> None:
    """Plot cluster permutation statistics with highlighted clusters."""

    fig, ax = _prepare_axes()
    ax.plot(time_s, stats, label="Statistic", color="steelblue")
    ax.axhline(threshold, linestyle=":", color="red", label="Cluster threshold")
    ax.axhline(-threshold, linestyle=":", color="red")
    for cluster in clusters:
        mask = cluster["mask"]
        color = "green" if cluster.get("sign", 1) > 0 else "purple"
        ax.fill_between(time_s[mask], stats[mask], 0, color=color, alpha=0.3)
        ax.text(
            float(time_s[mask][len(time_s[mask]) // 2]),
            float(np.max(stats[mask])),
            f"p={cluster.get('p_cluster', np.nan):.3f}",
            ha="center",
            va="bottom",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Statistic")
    ax.set_title(f"Cluster-based permutation test (alpha={alpha:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_km(
    lat_a: np.ndarray,
    evt_a: np.ndarray,
    lat_b: np.ndarray,
    evt_b: np.ndarray,
    out_path: str,
) -> Tuple[float, float]:
    """Plot Kaplan–Meier curves and return median latencies."""

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    km_a = KaplanMeierFitter(label=GROUP_A_LABEL)
    km_b = KaplanMeierFitter(label=GROUP_B_LABEL)
    km_a.fit(lat_a, event_observed=evt_a)
    km_b.fit(lat_b, event_observed=evt_b)
    km_a.plot_survival_function(ax=ax)
    km_b.plot_survival_function(ax=ax)
    ax.set_xlabel("Time to crossing (s)")
    ax.set_ylabel("Survival (1 - P[crossed])")
    ax.set_title("Kaplan–Meier latency curves")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    median_a = float(km_a.median_survival_time_)
    median_b = float(km_b.median_survival_time_)
    return median_a, median_b


def plot_randomization(time_s: np.ndarray, stat: np.ndarray, p: np.ndarray, q: np.ndarray, out_path: str) -> None:
    """Plot statistics from the randomization test."""

    fig, ax = _prepare_axes()
    ax.plot(time_s, stat, label="Statistic")
    ax2 = ax.twinx()
    ax2.plot(time_s, p, color="red", linestyle=":", label="p-value")
    ax2.plot(time_s, q, color="purple", linestyle="--", label="q-value")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Statistic")
    ax2.set_ylabel("Probability")
    ax.set_title("Time-wise randomization test")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_diff_curve(
    time_s: np.ndarray,
    diff: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    out_path: str,
    title: str = "Predicted difference (A - B)",
) -> None:
    """Plot predicted difference curves with confidence intervals."""

    fig, ax = _prepare_axes()
    ax.plot(time_s, diff, label="Predicted difference")
    ax.fill_between(time_s, ci_low, ci_high, color="steelblue", alpha=0.3, label="CI")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Difference")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_equivalence(time_s: np.ndarray, p_tost: np.ndarray, decision: np.ndarray, epsilon: np.ndarray, out_path: str) -> None:
    """Plot TOST results shading timepoints deemed equivalent."""

    fig, ax = _prepare_axes()
    ax.plot(time_s, p_tost, label="TOST p-value")
    ax.axhline(0.05, linestyle=":", color="red", label="alpha=0.05")
    mask = decision.astype(bool)
    if np.any(mask):
        ax.fill_between(time_s, 0, 0.05, where=mask, color="green", alpha=0.3, label="Equivalent")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("p-value")
    ax.set_ylim(0, 1)
    ax.set_title("Pre-odor equivalence test (TOST)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_yuen(time_s: np.ndarray, t_stat: np.ndarray, p: np.ndarray, q: np.ndarray, out_path: str) -> None:
    """Plot Yuen trimmed-mean statistics."""

    fig, ax = _prepare_axes()
    ax.plot(time_s, t_stat, label="Yuen t-statistic")
    ax2 = ax.twinx()
    ax2.plot(time_s, p, color="red", linestyle=":", label="p-value")
    ax2.plot(time_s, q, color="purple", linestyle="--", label="q-value")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("t-statistic")
    ax2.set_ylabel("Probability")
    ax.set_title("Yuen trimmed-mean test")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


__all__ = [
    "plot_p_q",
    "plot_effect_curve",
    "plot_discordant_counts",
    "plot_cluster_map",
    "plot_km",
    "plot_randomization",
    "plot_diff_curve",
    "plot_equivalence",
    "plot_yuen",
]

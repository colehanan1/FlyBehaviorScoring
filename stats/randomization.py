"""Time-wise paired randomization test using within-fly sign flips."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from . import plotting
from .utils import bh_fdr, diff_matrix

LOG = logging.getLogger("stats.randomization")


@dataclass
class RandomizationResult:
    """Result container for the randomization test."""

    statistic: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray


def _compute_statistic(diff: np.ndarray, method: str) -> np.ndarray:
    if method == "median":
        return np.nanmedian(diff, axis=0)
    if method == "mean":
        return np.nanmean(diff, axis=0)
    raise ValueError(f"Unsupported randomization statistic '{method}'.")


def randomization_test(
    groups: Sequence,
    time_s: np.ndarray,
    n_perm: int,
    method: str = "median",
    rng: Optional[np.random.Generator] = None,
) -> RandomizationResult:
    """Run the time-wise randomization test using paired sign flips."""

    if rng is None:
        rng = np.random.default_rng(0)
    diff = diff_matrix(groups)
    flies = diff.shape[0]
    if flies < 2:
        raise ValueError("Randomization test requires at least two flies for pairing.")
    observed = _compute_statistic(diff, method)
    extreme = np.zeros_like(observed, dtype=float)
    mask = np.isfinite(observed)

    for perm in range(n_perm):
        signs = rng.choice([-1, 1], size=flies)
        perm_diff = diff * signs[:, None]
        perm_stat = _compute_statistic(perm_diff, method)
        extreme[mask] += np.abs(perm_stat[mask]) >= np.abs(observed[mask])

    pvals = np.full_like(observed, np.nan, dtype=float)
    pvals[mask] = (1.0 + extreme[mask]) / (1.0 + n_perm)
    qvals = bh_fdr(pvals)
    LOG.info("Randomization test completed (%d permutations, method=%s).", n_perm, method)
    return RandomizationResult(statistic=observed, p_value=pvals, q_value=qvals)


def save_outputs(result: RandomizationResult, time_s: np.ndarray, out_dir: str) -> Tuple[str, str]:
    """Persist CSV and PNG outputs for the randomization analysis."""

    import os
    import pandas as pd

    csv_path = os.path.join(out_dir, "randomization.csv")
    pd.DataFrame({"time_s": time_s, "stat": result.statistic, "p": result.p_value, "q": result.q_value}).to_csv(
        csv_path, index=False
    )
    plot_path = os.path.join(out_dir, "randomization_plot.png")
    plotting.plot_randomization(time_s, result.statistic, result.p_value, result.q_value, plot_path)
    return csv_path, plot_path


__all__ = ["randomization_test", "save_outputs", "RandomizationResult"]

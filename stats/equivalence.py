"""Pre-odor equivalence testing using paired TOST per timepoint."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import t as student_t

from . import plotting
from .utils import diff_matrix

LOG = logging.getLogger("stats.equivalence")


@dataclass
class EquivalenceResult:
    p_tost: np.ndarray
    decision: np.ndarray
    epsilon: np.ndarray


def _tost_pvalue(samples: np.ndarray, epsilon: float, alpha: float) -> Tuple[float, bool]:
    samples = samples[np.isfinite(samples)]
    n = samples.size
    if n < 2:
        return np.nan, False
    mean = float(np.mean(samples))
    sd = float(np.std(samples, ddof=1))
    if sd == 0:
        within = abs(mean) <= epsilon
        return 0.0 if within else 1.0, within
    se = sd / np.sqrt(n)
    t_lower = (mean + epsilon) / se
    t_upper = (mean - epsilon) / se
    p_lower = 1 - student_t.cdf(t_lower, df=n - 1)
    p_upper = student_t.cdf(t_upper, df=n - 1)
    p_tost = max(p_lower, p_upper)
    decision = p_lower < alpha and p_upper < alpha
    return p_tost, decision


def run_equivalence(
    groups,
    time_s: np.ndarray,
    alpha: float,
    epsilon: Optional[float],
    epsilon_mult: Optional[float],
    pre_window: Tuple[int, int],
) -> EquivalenceResult:
    diff = diff_matrix(groups)
    start, end = pre_window
    if start < 0 or end <= start or end > diff.shape[1]:
        raise ValueError("Invalid pre-odor window for equivalence testing.")
    if epsilon is None:
        if epsilon_mult is None:
            raise ValueError("Equivalence margin not provided (use --equiv-margin or --equiv-margin-mult).")
        window_diffs = diff[:, start:end].ravel()
        baseline_sd = float(np.nanstd(window_diffs, ddof=1))
        epsilon = epsilon_mult * baseline_sd
        LOG.info("Equivalence margin set to %.6g (baseline SD %.6g * %.3f)", epsilon, baseline_sd, epsilon_mult)
    else:
        LOG.info("Equivalence margin set to absolute %.6g", epsilon)
    pvals = np.full(diff.shape[1], np.nan, dtype=float)
    decisions = np.zeros(diff.shape[1], dtype=int)
    eps_array = np.full(diff.shape[1], float(epsilon), dtype=float)
    for idx in range(start, end):
        p, dec = _tost_pvalue(diff[:, idx], float(epsilon), alpha)
        pvals[idx] = p
        decisions[idx] = int(dec)
    return EquivalenceResult(p_tost=pvals, decision=decisions, epsilon=eps_array)


def save_outputs(result: EquivalenceResult, time_s: np.ndarray, out_dir: str) -> Tuple[str, str]:
    """Write CSV and plot for equivalence testing."""

    import os
    import pandas as pd

    csv_path = os.path.join(out_dir, "equivalence_preodor.csv")
    pd.DataFrame(
        {
            "time_s": time_s,
            "p_tost": result.p_tost,
            "eq_decision": result.decision,
            "epsilon": result.epsilon,
        }
    ).to_csv(csv_path, index=False)
    plot_path = os.path.join(out_dir, "equivalence_plot.png")
    plotting.plot_equivalence(time_s, result.p_tost, result.decision, result.epsilon, plot_path)
    return csv_path, plot_path


__all__ = ["run_equivalence", "save_outputs", "EquivalenceResult"]

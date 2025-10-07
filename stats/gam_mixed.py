"""Mixed-effects spline model contrasting condition-specific trajectories."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import patsy
from statsmodels.regression.mixed_linear_model import MixedLM
from scipy.stats import chi2

from . import plotting
from .utils import stack_mean_traces

LOG = logging.getLogger("stats.gam_mixed")


@dataclass
class GamResult:
    """Container holding model fit details and predictions."""

    summary_text: str
    wald_stat: float
    wald_df: float
    wald_p: float
    diff: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray


def _build_dataframe(groups: Sequence, time_s: np.ndarray) -> pd.DataFrame:
    mean_a, mean_b = stack_mean_traces(groups)
    flies = len(groups)
    timepoints = len(time_s)
    records: List[dict] = []
    for fly_idx, group in enumerate(groups):
        fly = group.fly_id
        for t_idx in range(timepoints):
            records.append(
                {
                    "fly": fly,
                    "time": float(time_s[t_idx]),
                    "condition": 1,
                    "value": float(mean_a[fly_idx, t_idx]),
                }
            )
            records.append(
                {
                    "fly": fly,
                    "time": float(time_s[t_idx]),
                    "condition": 0,
                    "value": float(mean_b[fly_idx, t_idx]),
                }
            )
    return pd.DataFrame(records)


def fit_gam(
    groups: Sequence,
    time_s: np.ndarray,
    knots: int,
) -> GamResult:
    """Fit the mixed-effects spline model and return predictions."""

    if len(groups) < 2:
        raise ValueError("Mixed-effects GAM requires at least two flies.")
    data = _build_dataframe(groups, time_s)
    formula = "value ~ condition + bs(time, df=%d, include_intercept=False) + condition:bs(time, df=%d, include_intercept=False)" % (
        knots,
        knots,
    )
    LOG.info("Fitting MixedLM with %d knots and %d observations.", knots, len(data))
    model = MixedLM.from_formula(formula, groups="fly", data=data, re_formula="1")
    result = model.fit(method="lbfgs", maxiter=200, disp=False)
    fe_params = result.fe_params
    cov = result.cov_params()
    interaction_names = [name for name in fe_params.index if "condition:bs(time" in name]
    if not interaction_names:
        raise ValueError("Failed to locate condition spline interaction terms.")
    R = np.zeros((len(interaction_names), len(fe_params)))
    for row_idx, name in enumerate(interaction_names):
        col_idx = fe_params.index.get_loc(name)
        R[row_idx, col_idx] = 1.0
    summary_text = result.summary().as_text()
    wald_df = float(np.linalg.matrix_rank(R))
    if wald_df == 0:
        wald_stat = 0.0
        wald_p = 1.0
    else:
        fe_vector = fe_params.loc[fe_params.index].to_numpy()
        cov_matrix = cov.loc[fe_params.index, fe_params.index].to_numpy()
        Rb = R @ fe_vector
        RCovRT = R @ cov_matrix @ R.T
        if np.allclose(Rb, 0.0) or np.allclose(RCovRT, 0.0):
            wald_stat = 0.0
            wald_p = 1.0
        else:
            try:
                RCovRT_inv = np.linalg.pinv(RCovRT, hermitian=True)
            except TypeError:
                # hermitian flag added in newer numpy; fall back for compatibility
                RCovRT_inv = np.linalg.pinv(RCovRT)
            wald_stat = float(Rb.T @ RCovRT_inv @ Rb)
            wald_p = float(chi2.sf(wald_stat, df=wald_df))
    basis = patsy.dmatrix(
        f"bs(time, df={knots}, include_intercept=False)",
        {"time": time_s},
        return_type="dataframe",
    )
    diff = np.full(time_s.shape[0], np.nan, dtype=float)
    ci_low = np.full_like(diff, np.nan)
    ci_high = np.full_like(diff, np.nan)
    cov_matrix = cov.loc[fe_params.index, fe_params.index].to_numpy()
    for idx in range(time_s.size):
        design = np.zeros(len(fe_params), dtype=float)
        if "condition" in fe_params.index:
            design[fe_params.index.get_loc("condition")] = 1.0
        for col_idx, name in enumerate(basis.columns):
            param_name = f"condition:{name}"
            if param_name in fe_params.index:
                design[fe_params.index.get_loc(param_name)] = float(basis.iloc[idx, col_idx])
        diff[idx] = float(design @ fe_params.to_numpy())
        var = float(design @ cov_matrix @ design)
        se = np.sqrt(max(var, 0.0))
        ci_low[idx] = diff[idx] - 1.96 * se
        ci_high[idx] = diff[idx] + 1.96 * se
    return GamResult(
        summary_text=summary_text,
        wald_stat=wald_stat,
        wald_df=wald_df,
        wald_p=wald_p,
        diff=diff,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def save_outputs(result: GamResult, time_s: np.ndarray, out_dir: str) -> Tuple[str, str, str]:
    """Persist GAM outputs to disk."""

    import os
    import pandas as pd

    summary_path = os.path.join(out_dir, "gam_mixed_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(result.summary_text)
        handle.write("\n\nCondition × spline Wald test:\n")
        handle.write(f"statistic={result.wald_stat:.6g}, df={result.wald_df:.3f}, p={result.wald_p:.6g}\n")
    csv_path = os.path.join(out_dir, "gam_predicted_diff.csv")
    pd.DataFrame(
        {
            "time_s": time_s,
            "diff": result.diff,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
        }
    ).to_csv(csv_path, index=False)
    plot_path = os.path.join(out_dir, "gam_diff_plot.png")
    plotting.plot_diff_curve(time_s, result.diff, result.ci_low, result.ci_high, plot_path)
    return summary_path, csv_path, plot_path


__all__ = ["fit_gam", "save_outputs", "GamResult"]

#!/usr/bin/env python3
"""Command-line entry point for time-series statistical comparisons.

The script aggregates trial traces into within-fly means for two groups of
trials (Group A vs Group B) and runs per-timepoint statistical tests:

* Paired t-test and Wilcoxon signed-rank when at least two flies contribute to
  both groups. When fewer than two flies remain after filtering, tests fall back
  to the unpaired t-test (Welch) and Mann–Whitney U test, emitting a warning.
* McNemar-as-sign test comparing the direction of the effect across flies.
  Skipped when fewer than two flies have both groups.
* Optional Kaplan–Meier survival analysis with a log-rank test on the latency
  to first threshold crossing.

Outputs include CSVs with per-timepoint statistics, PNG figures, and optional
Kaplan–Meier summaries. Extensive DEBUG logging provides insight into data
selection, group sizes, and any skipped analyses.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import binom_test as sm_binom_test

LOG = logging.getLogger("stats.run_all")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class FlyGroup:
    """Container storing Group A/B trial matrices for a single fly."""

    fly_id: str
    trials: np.ndarray

    def mean_trace(self) -> np.ndarray:
        if self.trials.size == 0:
            raise ValueError(f"Fly {self.fly_id} has no trials in requested subset.")
        return np.nanmean(self.trials, axis=0)


@dataclass
class FlyGroups:
    """Encapsulates aligned Group A and Group B data for a fly."""

    fly_id: str
    group_a: FlyGroup
    group_b: FlyGroup

    @property
    def has_both(self) -> bool:
        return self.group_a.trials.size > 0 and self.group_b.trials.size > 0


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def parse_trial_list(raw: str) -> List[int]:
    trials = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            trials.append(int(item))
        except ValueError as exc:  # pragma: no cover - defensive logging
            raise ValueError(f"Invalid trial identifier '{item}' (must be int).") from exc
    if not trials:
        raise ValueError("Target trials list cannot be empty.")
    return trials


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_matrix(path: str) -> np.ndarray:
    LOG.debug("Loading matrix from %s", path)
    mat = np.load(path)
    if mat.ndim != 2:
        raise ValueError(f"Expected a 2D matrix [rows, time]; got shape={mat.shape}")
    if not np.issubdtype(mat.dtype, np.floating):
        LOG.warning("Matrix dtype %s is not float; casting to float32 for safety.", mat.dtype)
        mat = mat.astype(np.float32, copy=False)
    return np.asarray(mat)


def load_metadata(path: str) -> List[dict]:
    LOG.debug("Loading metadata from %s", path)
    with open(path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if isinstance(meta, dict):
        if "rows" in meta and isinstance(meta["rows"], list):
            return meta["rows"]
        if "row_index_to_meta" in meta and isinstance(meta["row_index_to_meta"], dict):
            ordered = sorted(meta["row_index_to_meta"].items(), key=lambda kv: int(kv[0]))
            return [entry for _, entry in ordered]
    if isinstance(meta, list):
        return meta
    raise ValueError(
        "Metadata JSON must contain either a `rows` list, a `row_index_to_meta` "
        "mapping, or be a list aligned to the matrix rows."
    )


def rows_to_dataframe(rows: Sequence[dict], fly_field: str, dataset_field: str, trial_field: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    missing = [field for field in (fly_field, dataset_field, trial_field) if field not in df.columns]
    if missing:
        raise KeyError(
            "Metadata missing required fields: "
            + ", ".join(missing)
            + f". Available keys: {sorted(df.columns.tolist())}"
        )
    df.insert(0, "row", np.arange(len(df), dtype=int))
    return df


def select_datasets(df: pd.DataFrame, datasets: Sequence[str], dataset_field: str) -> pd.DataFrame:
    datasets_set = {str(ds) for ds in datasets}
    filtered = df[df[dataset_field].astype(str).isin(datasets_set)].copy()
    LOG.debug("Rows after dataset filter: %d / %d", len(filtered), len(df))
    if filtered.empty:
        raise ValueError(
            "No metadata rows matched the requested datasets. "
            f"Requested={sorted(datasets_set)}, available={sorted(df[dataset_field].unique())}"
        )
    return filtered


def build_groups(
    matrix: np.ndarray,
    meta_df: pd.DataFrame,
    fly_field: str,
    trial_field: str,
    target_trials: Sequence[int],
) -> List[FlyGroups]:
    by_fly: Dict[str, pd.DataFrame] = {}
    for _, row in meta_df.iterrows():
        fly = str(row[fly_field])
        by_fly.setdefault(fly, []).append(row)
    groups: List[FlyGroups] = []
    for fly, rows in sorted(by_fly.items()):
        fly_df = pd.DataFrame(rows)
        rows_a = fly_df[fly_df[trial_field].astype(int).isin(target_trials)]["row"].to_numpy(dtype=int)
        rows_b = fly_df[~fly_df[trial_field].astype(int).isin(target_trials)]["row"].to_numpy(dtype=int)
        trials_a = matrix[rows_a, :] if rows_a.size else np.empty((0, matrix.shape[1]), dtype=matrix.dtype)
        trials_b = matrix[rows_b, :] if rows_b.size else np.empty((0, matrix.shape[1]), dtype=matrix.dtype)
        LOG.debug(
            "Fly %s: Group A trials=%s, Group B trials=%s",
            fly,
            rows_a.tolist(),
            rows_b.tolist(),
        )
        groups.append(
            FlyGroups(
                fly_id=fly,
                group_a=FlyGroup(fly_id=fly, trials=trials_a),
                group_b=FlyGroup(fly_id=fly, trials=trials_b),
            )
        )
    usable = [g for g in groups if g.has_both]
    dropped = [g.fly_id for g in groups if not g.has_both]
    if dropped:
        LOG.warning(
            "Dropping flies without at least one trial in both groups: %s",
            ", ".join(dropped),
        )
    if not usable:
        raise ValueError("No flies retained after filtering for both Group A and Group B trials.")
    return usable


# ---------------------------------------------------------------------------
# Statistical routines
# ---------------------------------------------------------------------------
def _paired_time_tests(
    group_means_a: np.ndarray,
    group_means_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    t_p = np.full(group_means_a.shape[1], np.nan, dtype=float)
    w_p = np.full_like(t_p, np.nan)
    for idx in range(group_means_a.shape[1]):
        a = group_means_a[:, idx]
        b = group_means_b[:, idx]
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if a.size < 2:
            continue
        try:
            _, p_t = ttest_rel(a, b, nan_policy="omit")
            t_p[idx] = p_t
        except Exception as exc:  # pragma: no cover - should never trigger
            LOG.debug("Paired t-test failed at t=%d: %s", idx, exc)
        diffs = a - b
        if np.allclose(diffs, 0.0):
            continue
        try:
            _, p_w = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", correction=True)
            w_p[idx] = p_w
        except ValueError as exc:
            LOG.debug("Wilcoxon skipped at t=%d: %s", idx, exc)
    return t_p, w_p


def _unpaired_time_tests(
    group_trials_a: np.ndarray,
    group_trials_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    t_p = np.full(group_trials_a.shape[1], np.nan, dtype=float)
    u_p = np.full_like(t_p, np.nan)
    for idx in range(group_trials_a.shape[1]):
        a = group_trials_a[:, idx]
        b = group_trials_b[:, idx]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            continue
        try:
            _, p_t = ttest_ind(a, b, equal_var=False, nan_policy="omit")
            t_p[idx] = p_t
        except Exception as exc:  # pragma: no cover
            LOG.debug("Welch t-test failed at t=%d: %s", idx, exc)
        try:
            _, p_u = mannwhitneyu(a, b, alternative="two-sided")
            u_p[idx] = p_u
        except ValueError as exc:
            LOG.debug("Mann–Whitney U skipped at t=%d: %s", idx, exc)
    return t_p, u_p


def paired_or_unpaired_tests(
    groups: List[FlyGroups],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Run per-timepoint tests, returning p-values and effect estimates.

    Returns a tuple of (primary_p, secondary_p, effect_mean, effect_median, paired).
    When ``paired`` is ``False`` the tests correspond to Welch t-test and
    Mann–Whitney U instead of paired tests.
    """

    mean_a = np.vstack([g.group_a.mean_trace() for g in groups])
    mean_b = np.vstack([g.group_b.mean_trace() for g in groups])
    diff = mean_a - mean_b
    effect_mean = np.nanmean(diff, axis=0)
    effect_median = np.nanmedian(diff, axis=0)

    if len(groups) >= 2:
        LOG.info("Running paired per-timepoint tests across %d flies.", len(groups))
        primary_p, secondary_p = _paired_time_tests(mean_a, mean_b)
        paired = True
    else:
        LOG.warning(
            "Fewer than two flies available (%d). Falling back to unpaired tests and "
            "skipping McNemar.",
            len(groups),
        )
        trials_a = np.vstack([g.group_a.trials for g in groups])
        trials_b = np.vstack([g.group_b.trials for g in groups])
        effect_mean = np.nanmean(trials_a, axis=0) - np.nanmean(trials_b, axis=0)
        effect_median = np.nanmedian(trials_a, axis=0) - np.nanmedian(trials_b, axis=0)
        primary_p, secondary_p = _unpaired_time_tests(trials_a, trials_b)
        paired = False
    return primary_p, secondary_p, effect_mean, effect_median, paired


def mcnemar_sign_test(groups: List[FlyGroups]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flies = [g.fly_id for g in groups]
    LOG.info("Running McNemar sign test across %d flies.", len(flies))
    if len(groups) < 2:
        raise ValueError("McNemar test requires at least two flies with both groups.")
    mean_a = np.vstack([g.group_a.mean_trace() for g in groups])
    mean_b = np.vstack([g.group_b.mean_trace() for g in groups])
    timepoints = mean_a.shape[1]
    b_counts = np.zeros(timepoints, dtype=int)
    c_counts = np.zeros(timepoints, dtype=int)
    pvals = np.ones(timepoints, dtype=float)
    for idx in range(timepoints):
        a = mean_a[:, idx]
        b = mean_b[:, idx]
        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]
        gt = np.sum(a > b)
        lt = np.sum(b > a)
        b_counts[idx] = int(gt)
        c_counts[idx] = int(lt)
        if gt + lt == 0:
            pvals[idx] = 1.0
            continue
        k = min(gt, lt)
        n = gt + lt
        pvals[idx] = sm_binom_test(k, n, 0.5, alternative="two-sided")
    return pvals, b_counts.astype(float), c_counts.astype(float)


def compute_kaplan_meier(
    groups: List[FlyGroups],
    threshold: float,
    time_hz: float,
    window: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def _stack(trial_getter: Iterable[np.ndarray]) -> np.ndarray:
        stacked = np.vstack(list(trial_getter))
        if stacked.ndim != 2:
            raise ValueError("Stacked trial matrix must be 2D.")
        return stacked

    LOG.info("Computing Kaplan–Meier latency with threshold %s", threshold)
    trials_a = _stack(g.group_a.trials for g in groups)
    trials_b = _stack(g.group_b.trials for g in groups)
    lat_a, evt_a = latency_to_threshold(trials_a, threshold, window)
    lat_b, evt_b = latency_to_threshold(trials_b, threshold, window)
    lat_a_sec = lat_a / float(time_hz)
    lat_b_sec = lat_b / float(time_hz)
    evt_a = evt_a.astype(int)
    evt_b = evt_b.astype(int)
    return lat_a_sec, evt_a, lat_b_sec, evt_b


def latency_to_threshold(
    traces: np.ndarray,
    threshold: float,
    window: Optional[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    if window is None:
        start, end = 0, traces.shape[1]
    else:
        start, end = window
        if start < 0 or end <= start or end > traces.shape[1]:
            raise ValueError(
                f"Invalid KM window {window}; must satisfy 0 <= start < end <= trace length {traces.shape[1]}"
            )
    latencies = np.empty(traces.shape[0], dtype=float)
    events = np.zeros(traces.shape[0], dtype=bool)
    search_len = end - start
    for idx, trace in enumerate(traces):
        segment = trace[start:end]
        crossings = np.where(segment >= threshold)[0]
        if crossings.size:
            latencies[idx] = start + crossings[0]
            events[idx] = True
        else:
            latencies[idx] = start + search_len - 1
            events[idx] = False
    return latencies, events


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def bh_correction(pvals: np.ndarray) -> np.ndarray:
    qvals = np.full_like(pvals, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    if mask.sum() == 0:
        return qvals
    _, q, _, _ = multipletests(pvals[mask], alpha=0.05, method="fdr_bh")
    qvals[mask] = q.astype(float)
    return qvals


def plot_series(time_s: np.ndarray, series: Dict[str, np.ndarray], title: str, ylabel: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    for label, values in series.items():
        ax.plot(time_s, values, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0 if np.all(series[next(iter(series))] >= 0) else None)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.debug("Saved plot to %s", out_path)


def plot_effect(time_s: np.ndarray, effect: np.ndarray, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(time_s, effect, label="Effect: A - B")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (A - B)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.debug("Saved effect plot to %s", out_path)


def plot_km_curves(
    lat_a_sec: np.ndarray,
    evt_a: np.ndarray,
    lat_b_sec: np.ndarray,
    evt_b: np.ndarray,
    out_path: str,
) -> Tuple[float, float]:
    km_a = KaplanMeierFitter(label="Group A (target trials)")
    km_b = KaplanMeierFitter(label="Group B (other trials)")
    km_a.fit(lat_a_sec, event_observed=evt_a)
    km_b.fit(lat_b_sec, event_observed=evt_b)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    km_a.plot_survival_function(ax=ax)
    km_b.plot_survival_function(ax=ax)
    ax.set_xlabel("Time to threshold crossing (s)")
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan–Meier survival curves")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    LOG.debug("Saved Kaplan–Meier plot to %s", out_path)
    median_a = km_a.median_survival_time_
    median_b = km_b.median_survival_time_
    return (
        float(median_a) if median_a is not None and not math.isnan(median_a) else float("nan"),
        float(median_b) if median_b is not None and not math.isnan(median_b) else float("nan"),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run time-series statistical comparisons between trial groups.")
    parser.add_argument("--npy", required=True, help="Path to numpy matrix [rows, time].")
    parser.add_argument("--meta", required=True, help="Metadata JSON with a `rows` list aligned to the matrix.")
    parser.add_argument("--out", required=True, help="Directory to write outputs.")
    parser.add_argument(
        "--datasets",
        default="testing",
        help="Comma-separated dataset labels to include (default: testing).",
    )
    parser.add_argument(
        "--target-trials",
        default="2,4,5",
        help="Comma-separated trial IDs defining Group A (default: 2,4,5).",
    )
    parser.add_argument("--fly-field", default="fly", help="Metadata key for fly identifier.")
    parser.add_argument("--dataset-field", default="dataset", help="Metadata key for dataset label.")
    parser.add_argument("--trial-field", default="trial", help="Metadata key for trial number (1-based).")
    parser.add_argument("--time-hz", type=float, default=40.0, help="Sampling rate in Hz (for axis labels).")
    parser.add_argument(
        "--km-threshold",
        default="None",
        help="Threshold for latency analysis: numeric, percentile:Q, or None to skip.",
    )
    parser.add_argument(
        "--km-window",
        default=None,
        help="Window for latency search as 'start:end' in samples (defaults to full trace).",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default INFO).",
    )
    return parser.parse_args(argv)


def parse_datasets(raw: str) -> List[str]:
    datasets = [item.strip() for item in raw.split(",") if item.strip()]
    if not datasets:
        raise ValueError("At least one dataset label must be provided.")
    return datasets


def parse_threshold(raw: str, matrix: np.ndarray) -> Optional[float]:
    if raw is None or str(raw).lower() == "none":
        return None
    raw = raw.strip()
    if raw.lower().startswith("percentile:"):
        try:
            q = float(raw.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("Percentile threshold must be numeric, e.g., percentile:95") from exc
        value = float(np.nanpercentile(matrix, q))
        LOG.info("Resolved percentile threshold %.2f -> %.6g", q, value)
        return value
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Threshold '{raw}' is neither numeric nor percentile:Q.") from exc


def parse_window(raw: Optional[str]) -> Optional[Tuple[int, int]]:
    if raw is None:
        return None
    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError("--km-window must be formatted as start:end")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError("Window bounds must be integers.") from exc
    return start, end


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.loglevel), format="[%(levelname)s] %(message)s")
    LOG.info("Starting analysis.")

    ensure_out_dir(args.out)
    matrix = load_matrix(args.npy)
    rows = load_metadata(args.meta)
    df = rows_to_dataframe(rows, args.fly_field, args.dataset_field, args.trial_field)
    datasets = parse_datasets(args.datasets)
    target_trials = parse_trial_list(args.target_trials)
    LOG.debug("Target datasets=%s, target trials=%s", datasets, target_trials)

    df = select_datasets(df, datasets, args.dataset_field)
    groups = build_groups(matrix, df, args.fly_field, args.trial_field, target_trials)
    LOG.info("Retained %d flies for paired analyses.", len(groups))

    primary_p, secondary_p, effect_mean, effect_median, paired = paired_or_unpaired_tests(groups)
    timepoints = matrix.shape[1]
    time_axis = np.arange(timepoints, dtype=float) / float(args.time_hz)
    primary_q = bh_correction(primary_p)
    secondary_q = bh_correction(secondary_p)

    t_label = "paired_t" if paired else "welch_t"
    w_label = "wilcoxon" if paired else "mannwhitneyu"

    primary_df = pd.DataFrame(
        {
            "time_s": time_axis,
            "p_value": primary_p,
            "q_value": primary_q,
            "effect_mean_A_minus_B": effect_mean,
            "effect_median_A_minus_B": effect_median,
        }
    )
    primary_csv = os.path.join(args.out, f"{t_label}.csv")
    primary_df.to_csv(primary_csv, index=False)
    LOG.info("Saved %s", primary_csv)

    secondary_df = pd.DataFrame({"time_s": time_axis, "p_value": secondary_p, "q_value": secondary_q})
    secondary_csv = os.path.join(args.out, f"{w_label}.csv")
    secondary_df.to_csv(secondary_csv, index=False)
    LOG.info("Saved %s", secondary_csv)

    plot_series(
        time_axis,
        {"p-value": primary_p, "BH q-value": primary_q},
        "Primary test p-values over time",
        "p",
        os.path.join(args.out, f"{t_label}_plot.png"),
    )
    plot_series(
        time_axis,
        {"p-value": secondary_p, "BH q-value": secondary_q},
        "Secondary test p-values over time",
        "p",
        os.path.join(args.out, f"{w_label}_plot.png"),
    )
    plot_effect(
        time_axis,
        effect_mean,
        "Effect size (mean across subjects)",
        os.path.join(args.out, "effect_mean_plot.png"),
    )

    if paired and len(groups) >= 2:
        mcnemar_p, b_counts, c_counts = mcnemar_sign_test(groups)
        mcnemar_q = bh_correction(mcnemar_p)
        mcnemar_df = pd.DataFrame(
            {
                "time_s": time_axis,
                "p_value": mcnemar_p,
                "q_value": mcnemar_q,
                "discordant_A_gt_B": b_counts,
                "discordant_B_gt_A": c_counts,
            }
        )
        mcnemar_csv = os.path.join(args.out, "mcnemar.csv")
        mcnemar_df.to_csv(mcnemar_csv, index=False)
        LOG.info("Saved %s", mcnemar_csv)
        plot_series(
            time_axis,
            {"p-value": mcnemar_p, "BH q-value": mcnemar_q},
            "McNemar sign-test p-values over time",
            "p",
            os.path.join(args.out, "mcnemar_plot.png"),
        )
        plot_series(
            time_axis,
            {"A>B": b_counts, "B>A": c_counts},
            "Discordant counts per timepoint",
            "count (flies)",
            os.path.join(args.out, "mcnemar_bc_plot.png"),
        )
    else:
        LOG.warning("McNemar analysis skipped due to insufficient flies for pairing.")

    threshold = parse_threshold(args.km_threshold, matrix)
    if threshold is None:
        LOG.info("Kaplan–Meier analysis skipped (no threshold provided).")
    else:
        window = parse_window(args.km_window)
        lat_a_sec, evt_a, lat_b_sec, evt_b = compute_kaplan_meier(groups, threshold, args.time_hz, window)
        median_a, median_b = plot_km_curves(
            lat_a_sec,
            evt_a,
            lat_b_sec,
            evt_b,
            os.path.join(args.out, "km_plot.png"),
        )
        lr = logrank_test(lat_a_sec, lat_b_sec, event_observed_A=evt_a, event_observed_B=evt_b)
        LOG.info("Log-rank test statistic=%.4f, p=%.6g", lr.test_statistic, lr.p_value)
        km_df = pd.DataFrame(
            {
                "group": ["A", "B"],
                "n_trials": [lat_a_sec.size, lat_b_sec.size],
                "median_latency_s": [median_a, median_b],
                "logrank_test_stat": [float(lr.test_statistic)] * 2,
                "logrank_p_value": [float(lr.p_value)] * 2,
            }
        )
        km_csv = os.path.join(args.out, "km_logrank.csv")
        km_df.to_csv(km_csv, index=False)
        LOG.info("Saved %s", km_csv)

    LOG.info("Analysis complete. Outputs written to %s", os.path.abspath(args.out))


if __name__ == "__main__":  # pragma: no cover
    main()

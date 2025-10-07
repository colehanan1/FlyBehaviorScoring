#!/usr/bin/env python3
"""Command-line orchestrator for advanced time-series analyses."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.proportion import binom_test as sm_binom_test

from stats import cluster_perm, equivalence, gam_mixed, plotting, randomization, survival, utils

LOG = logging.getLogger("stats.run_all")


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------

def paired_or_unpaired_tests(
    groups: Sequence[utils.FlyGroups],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Run paired or unpaired per-timepoint tests depending on fly count."""

    mean_a, mean_b = utils.stack_mean_traces(groups)
    diff = mean_a - mean_b
    effect_mean = np.nanmean(diff, axis=0)
    effect_median = np.nanmedian(diff, axis=0)

    if len(groups) >= 2:
        LOG.info("Running paired per-timepoint tests across %d flies.", len(groups))
        t_p = np.full(mean_a.shape[1], np.nan, dtype=float)
        w_p = np.full_like(t_p, np.nan)
        for idx in range(mean_a.shape[1]):
            a = mean_a[:, idx]
            b = mean_b[:, idx]
            mask = np.isfinite(a) & np.isfinite(b)
            a = a[mask]
            b = b[mask]
            if a.size < 2:
                continue
            try:
                _, p_t = ttest_rel(a, b, nan_policy="omit")
                t_p[idx] = p_t
            except Exception as exc:  # pragma: no cover - defensive logging
                LOG.debug("Paired t-test failed at idx=%d: %s", idx, exc)
            diffs = a - b
            if np.allclose(diffs, 0.0):
                continue
            try:
                _, p_w = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", correction=True)
                w_p[idx] = p_w
            except ValueError as exc:
                LOG.debug("Wilcoxon skipped at idx=%d: %s", idx, exc)
        return t_p, w_p, effect_mean, effect_median, True

    LOG.warning(
        "Fewer than two flies available (%d). Falling back to unpaired tests and skipping McNemar.",
        len(groups),
    )
    trials_a = np.vstack([g.group_a.trials for g in groups])
    trials_b = np.vstack([g.group_b.trials for g in groups])
    effect_mean = np.nanmean(trials_a, axis=0) - np.nanmean(trials_b, axis=0)
    effect_median = np.nanmedian(trials_a, axis=0) - np.nanmedian(trials_b, axis=0)
    t_p = np.full(trials_a.shape[1], np.nan, dtype=float)
    u_p = np.full_like(t_p, np.nan)
    for idx in range(trials_a.shape[1]):
        a = trials_a[:, idx]
        b = trials_b[:, idx]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            continue
        try:
            _, p_t = ttest_ind(a, b, equal_var=False, nan_policy="omit")
            t_p[idx] = p_t
        except Exception as exc:
            LOG.debug("Welch t-test failed at idx=%d: %s", idx, exc)
        try:
            _, p_u = mannwhitneyu(a, b, alternative="two-sided")
            u_p[idx] = p_u
        except ValueError as exc:
            LOG.debug("Mann–Whitney U skipped at idx=%d: %s", idx, exc)
    return t_p, u_p, effect_mean, effect_median, False


def mcnemar_sign_test(groups: Sequence[utils.FlyGroups]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run McNemar-as-sign test at each timepoint across flies."""

    if len(groups) < 2:
        raise ValueError("McNemar test requires at least two flies with both groups.")
    mean_a, mean_b = utils.stack_mean_traces(groups)
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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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
        raise ValueError("Window specification must be formatted as start:end")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError("Window bounds must be integers.") from exc
    return start, end


def parse_float_list(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    values: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value '{token}'.") from exc
    return values


def earliest_onset(time_s: np.ndarray, qvals: np.ndarray, alpha: float) -> Optional[float]:
    mask = np.isfinite(qvals) & (qvals < alpha)
    if not np.any(mask):
        return None
    return float(time_s[mask][0])


def save_manifest(manifest: Dict[str, object], out_dir: str) -> str:
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Time-series stats for proboscis distance traces.")
    parser.add_argument("--npy", required=True, help="Path to envelope_matrix_float16.npy [rows,time].")
    parser.add_argument("--meta", required=True, help="Path to code_maps.json or similar metadata export.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--datasets", default="testing", help="Datasets to include (comma-separated).")
    parser.add_argument("--target-trials", default="2,4,5", help="Trials considered Group A (comma-separated).")
    parser.add_argument("--fly-field", default="fly", help="Metadata key for fly identifier.")
    parser.add_argument("--dataset-field", default="dataset", help="Metadata key for dataset label.")
    parser.add_argument("--trial-field", default="trial", help="Metadata key for trial number (1-based).")
    parser.add_argument("--time-hz", type=float, default=40.0, help="Sampling rate in Hz (for axis labels).")
    parser.add_argument("--km-threshold", default="None", help="Threshold for latency analysis (numeric or percentile:Q).")
    parser.add_argument("--km-window", default=None, help="Window for latency search 'start:end' in samples.")
    parser.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Family-wise alpha for significance summaries.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for resampling-based methods.")

    # Cluster permutation
    parser.add_argument("--do-cluster-perm", action="store_true", help="Run cluster-based permutation test.")
    parser.add_argument("--cluster-stat", choices=["t", "wilcoxon"], default="t", help="Statistic for cluster permutation.")
    parser.add_argument("--n-perm", type=int, default=2000, help="Number of permutations for cluster test.")

    # Randomization test
    parser.add_argument("--do-randomization", action="store_true", help="Run time-wise randomization test.")
    parser.add_argument("--rand-stat", choices=["median", "mean"], default="median", help="Statistic for randomization test.")
    parser.add_argument("--rand-n-perm", type=int, default=5000, help="Number of permutations for randomization test.")

    # GAM mixed model
    parser.add_argument("--do-gam", action="store_true", help="Fit mixed-effects GAM across time.")
    parser.add_argument("--gam-knots", type=int, default=12, help="Number of spline knots for GAM model.")

    # Survival sweep
    parser.add_argument("--do-survival-sweep", action="store_true", help="Run latency threshold sweep with Cox/Log-rank analyses.")
    parser.add_argument("--km-thresholds", default=None, help="Comma-separated numeric thresholds for survival sweep.")
    parser.add_argument("--km-percentiles", default=None, help="Comma-separated percentiles for threshold sweep.")
    parser.add_argument("--km-adaptive-k", type=float, default=None, help="Adaptive threshold multiplier (baseline mean + k*std).")
    parser.add_argument("--baseline-window", default=None, help="Baseline window 'start:end' for adaptive threshold.")

    # Equivalence testing
    parser.add_argument("--do-equivalence", action="store_true", help="Run pre-odor equivalence (TOST).")
    parser.add_argument("--equiv-margin", type=float, default=None, help="Absolute equivalence margin.")
    parser.add_argument("--equiv-margin-mult", type=float, default=None, help="Margin multiplier of baseline SD.")
    parser.add_argument("--pre-window", default="0:1200", help="Pre-odor window 'start:end' in samples.")

    # Yuen trimmed-mean test
    parser.add_argument("--do-yuen", action="store_true", help="Run Yuen trimmed-mean test per timepoint.")
    parser.add_argument("--trim-pct", type=float, default=0.2, help="Trim proportion for Yuen test (0-0.5).")

    return parser


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.loglevel), format="[%(levelname)s] %(message)s")
    LOG.info("Starting analysis.")

    rng = np.random.default_rng(args.seed)
    utils.ensure_out_dir(args.out)
    matrix = utils.load_matrix(args.npy)
    metadata = utils.load_metadata(args.meta)

    if metadata.rows is not None:
        LOG.info("Loaded %d metadata rows from JSON export.", len(metadata.rows))
        df = utils.rows_to_dataframe(metadata.rows, args.fly_field, args.dataset_field, args.trial_field)
        time_matrix = matrix
    else:
        if metadata.column_order is None:
            raise ValueError("Metadata file lacked row records and column order information.")
        LOG.info("Decoding columnar metadata via column_order/code_maps heuristics.")
        time_matrix, df, _ = utils.dataframe_from_columnar_matrix(
            matrix,
            metadata.column_order,
            metadata.code_maps,
            args.fly_field,
            args.dataset_field,
            args.trial_field,
        )
    datasets = utils.parse_datasets(args.datasets)
    target_trials = utils.parse_trial_list(args.target_trials)
    df = utils.select_datasets(df, datasets, args.dataset_field)
    groups = utils.build_groups(time_matrix, df, args.fly_field, args.trial_field, target_trials)
    LOG.info("Retained %d flies for analyses.", len(groups))

    time_s = utils.compute_time_axis(time_matrix.shape[1], args.time_hz)
    manifest: Dict[str, object] = {}

    # Primary tests
    primary_p, secondary_p, effect_mean, effect_median, paired = paired_or_unpaired_tests(groups)
    primary_q = utils.bh_fdr(primary_p)
    secondary_q = utils.bh_fdr(secondary_p)
    t_label = "paired_t" if paired else "welch_t"
    w_label = "wilcoxon" if paired else "mannwhitneyu"

    primary_df = pd.DataFrame(
        {
            "time_s": time_s,
            "p_value": primary_p,
            "q_value": primary_q,
            "effect_mean_A_minus_B": effect_mean,
            "effect_median_A_minus_B": effect_median,
        }
    )
    primary_csv = os.path.join(args.out, f"{t_label}.csv")
    primary_df.to_csv(primary_csv, index=False)
    primary_plot = os.path.join(args.out, f"{t_label}_plot.png")
    plotting.plot_p_q(
        time_s,
        primary_p,
        primary_q,
        f"Primary test p-values: {utils.GROUP_A_LABEL} vs {utils.GROUP_B_LABEL}",
        primary_plot,
        alpha=args.alpha,
    )
    effect_plot = os.path.join(args.out, "effect_mean_plot.png")
    plotting.plot_effect_curve(
        time_s,
        effect_mean,
        f"Effect size: {utils.GROUP_A_LABEL} minus {utils.GROUP_B_LABEL}",
        effect_plot,
    )
    manifest["primary_test"] = {
        "csv": primary_csv,
        "plot": primary_plot,
        "effect_plot": effect_plot,
        "earliest_onset_s": earliest_onset(time_s, primary_q, args.alpha),
    }

    secondary_df = pd.DataFrame(
        {
            "time_s": time_s,
            "p_value": secondary_p,
            "q_value": secondary_q,
            "effect_median_A_minus_B": effect_median,
        }
    )
    secondary_csv = os.path.join(args.out, f"{w_label}.csv")
    secondary_df.to_csv(secondary_csv, index=False)
    sign_indicator = np.where(secondary_q < args.alpha, np.sign(effect_median), 0.0)
    secondary_plot = os.path.join(args.out, f"{w_label}_plot.png")
    plotting.plot_p_q(
        time_s,
        secondary_p,
        secondary_q,
        f"Secondary test p-values: {utils.GROUP_A_LABEL} vs {utils.GROUP_B_LABEL}",
        secondary_plot,
        alpha=args.alpha,
        highlight_sign=sign_indicator,
    )
    manifest[w_label] = {
        "csv": secondary_csv,
        "plot": secondary_plot,
        "earliest_onset_s": earliest_onset(time_s, secondary_q, args.alpha),
    }

    # McNemar
    if paired and len(groups) >= 2:
        mcnemar_p, b_counts, c_counts = mcnemar_sign_test(groups)
        mcnemar_q = utils.bh_fdr(mcnemar_p)
        mcnemar_df = pd.DataFrame(
            {
                "time_s": time_s,
                "p_value": mcnemar_p,
                "q_value": mcnemar_q,
                "discordant_A_gt_B": b_counts,
                "discordant_B_gt_A": c_counts,
            }
        )
        mcnemar_csv = os.path.join(args.out, "mcnemar.csv")
        mcnemar_df.to_csv(mcnemar_csv, index=False)
        mcnemar_plot = os.path.join(args.out, "mcnemar_plot.png")
        plotting.plot_p_q(
            time_s,
            mcnemar_p,
            mcnemar_q,
            f"McNemar sign-test p-values: {utils.GROUP_A_LABEL} vs {utils.GROUP_B_LABEL}",
            mcnemar_plot,
            alpha=args.alpha,
        )
        mcnemar_bc_plot = os.path.join(args.out, "mcnemar_bc_plot.png")
        plotting.plot_discordant_counts(
            time_s,
            b_counts,
            c_counts,
            mcnemar_bc_plot,
        )
        manifest["mcnemar"] = {
            "csv": mcnemar_csv,
            "plot": mcnemar_plot,
            "bc_plot": mcnemar_bc_plot,
            "earliest_onset_s": earliest_onset(time_s, mcnemar_q, args.alpha),
        }
    else:
        LOG.warning("McNemar analysis skipped due to insufficient flies for pairing.")

    # Optional KM/log-rank baseline
    threshold = parse_threshold(args.km_threshold, time_matrix)
    if threshold is None:
        LOG.info("Kaplan–Meier analysis skipped (no threshold provided).")
    else:
        window = parse_window(args.km_window)
        trials_a = np.vstack([g.group_a.trials for g in groups])
        trials_b = np.vstack([g.group_b.trials for g in groups])
        lat_a, evt_a = utils.latency_to_threshold(trials_a, threshold, window)
        lat_b, evt_b = utils.latency_to_threshold(trials_b, threshold, window)
        lat_a_sec = lat_a / float(args.time_hz)
        lat_b_sec = lat_b / float(args.time_hz)
        km_plot = os.path.join(args.out, "km_plot.png")
        median_a, median_b = plotting.plot_km(lat_a_sec, evt_a.astype(int), lat_b_sec, evt_b.astype(int), km_plot)
        lr = logrank_test(lat_a_sec, lat_b_sec, event_observed_A=evt_a, event_observed_B=evt_b)
        km_csv = os.path.join(args.out, "km_logrank.csv")
        pd.DataFrame(
            {
                "group": [utils.GROUP_A_LABEL, utils.GROUP_B_LABEL],
                "n_trials": [lat_a_sec.size, lat_b_sec.size],
                "median_latency_s": [median_a, median_b],
                "logrank_stat": [float(lr.test_statistic)] * 2,
                "logrank_p_value": [float(lr.p_value)] * 2,
            }
        ).to_csv(km_csv, index=False)
        manifest["km_single"] = {"threshold": threshold, "plot": km_plot, "csv": km_csv}

    # Cluster permutation
    if args.do_cluster_perm:
        LOG.info("Running cluster permutation analysis.")
        cluster_result = cluster_perm.cluster_permutation_test(
            groups,
            time_s,
            n_perm=args.n_perm,
            alpha=args.alpha,
            method=args.cluster_stat,
            rng=rng,
        )
        clusters_csv, time_csv, plot_path = cluster_perm.save_outputs(
            cluster_result,
            time_s,
            args.out,
            alpha=args.alpha,
            method=args.cluster_stat,
        )
        sig_clusters = [c for c in cluster_result.clusters if c.p_cluster < args.alpha]
        total_duration = 0.0
        earliest = None
        for cluster in sig_clusters:
            start_idx = int(cluster.indices[0])
            end_idx = int(cluster.indices[-1])
            duration = float(time_s[end_idx] - time_s[start_idx])
            total_duration += max(duration, 0.0)
            onset = float(time_s[start_idx])
            if earliest is None or onset < earliest:
                earliest = onset
        manifest["cluster_perm"] = {
            "time_csv": time_csv,
            "clusters_csv": clusters_csv,
            "plot": plot_path,
            "n_clusters": len(cluster_result.clusters),
            "n_significant": len(sig_clusters),
            "total_significant_duration_s": total_duration,
            "earliest_onset_s": earliest,
        }

    # Randomization test
    if args.do_randomization:
        LOG.info("Running time-wise randomization test.")
        rand_result = randomization.randomization_test(
            groups,
            time_s,
            n_perm=args.rand_n_perm,
            method=args.rand_stat,
            rng=rng,
        )
        rand_csv, rand_plot = randomization.save_outputs(rand_result, time_s, args.out)
        manifest["randomization"] = {
            "csv": rand_csv,
            "plot": rand_plot,
            "earliest_onset_s": earliest_onset(time_s, rand_result.q_value, args.alpha),
        }

    # Mixed-effects GAM
    if args.do_gam:
        LOG.info("Running mixed-effects GAM model.")
        gam_result = gam_mixed.fit_gam(groups, time_s, args.gam_knots)
        gam_summary, gam_csv, gam_plot = gam_mixed.save_outputs(gam_result, time_s, args.out)
        manifest["gam"] = {
            "summary": gam_summary,
            "csv": gam_csv,
            "plot": gam_plot,
            "wald_stat": gam_result.wald_stat,
            "wald_p": gam_result.wald_p,
        }

    # Survival sweep
    if args.do_survival_sweep:
        LOG.info("Running survival threshold sweep.")
        thresholds = survival.resolve_thresholds(
            time_matrix,
            parse_float_list(args.km_thresholds),
            parse_float_list(args.km_percentiles),
            args.km_adaptive_k,
            parse_window(args.baseline_window) if args.baseline_window else None,
        )
        if not thresholds:
            raise ValueError("Survival sweep requested but no thresholds resolved.")
        sweep_window = parse_window(args.km_window)
        outcomes = survival.run_survival_sweep(groups, time_matrix, args.time_hz, thresholds, sweep_window, args.out)
        best = None
        for outcome in outcomes:
            entry = {
                "label": outcome.label,
                "threshold": outcome.threshold,
                "km_plot": outcome.km_plot,
                "km_csv": outcome.km_csv,
                "cox_txt": outcome.cox_txt,
                "logrank_stat": outcome.logrank_stat,
                "logrank_p": outcome.logrank_p,
                "hazard_ratio": outcome.hazard_ratio,
                "hr_ci": outcome.hr_ci,
                "cox_p": outcome.cox_p,
            }
            utils.update_manifest_entry(manifest, "survival", entry)
            if outcome.hazard_ratio and outcome.hazard_ratio > 0:
                score = abs(math.log(outcome.hazard_ratio))
                if best is None or score > best[0]:
                    best = (score, outcome.label, outcome.hazard_ratio)
        if best is not None:
            manifest["survival_best"] = {
                "label": best[1],
                "hazard_ratio": best[2],
            }

    # Equivalence testing
    if args.do_equivalence:
        LOG.info("Running pre-odor equivalence testing.")
        pre_window = parse_window(args.pre_window)
        if pre_window is None:
            raise ValueError("Pre-odor window required for equivalence testing.")
        eq_result = equivalence.run_equivalence(
            groups,
            time_s,
            alpha=args.alpha,
            epsilon=args.equiv_margin,
            epsilon_mult=args.equiv_margin_mult,
            pre_window=pre_window,
        )
        eq_csv, eq_plot = equivalence.save_outputs(eq_result, time_s, args.out)
        eq_mask = eq_result.decision > 0
        eq_onset = float(time_s[eq_mask][0]) if np.any(eq_mask) else None
        manifest["equivalence"] = {
            "csv": eq_csv,
            "plot": eq_plot,
            "earliest_equivalent_s": eq_onset,
        }

    # Yuen trimmed-mean test
    if args.do_yuen:
        LOG.info("Running Yuen trimmed-mean test (trim_pct=%.2f).", args.trim_pct)
        diff = utils.diff_matrix(groups)
        stats = np.full(diff.shape[1], np.nan, dtype=float)
        pvals = np.full_like(stats, np.nan)
        for idx in range(diff.shape[1]):
            t_stat, p = utils.yuen_p_value(diff[:, idx], args.trim_pct)
            stats[idx] = t_stat
            pvals[idx] = p
        qvals = utils.bh_fdr(pvals)
        yuen_df = pd.DataFrame({"time_s": time_s, "t_stat": stats, "p": pvals, "q": qvals})
        yuen_csv = os.path.join(args.out, "yuen.csv")
        yuen_df.to_csv(yuen_csv, index=False)
        yuen_plot = os.path.join(args.out, "yuen_plot.png")
        plotting.plot_yuen(time_s, stats, pvals, qvals, yuen_plot)
        manifest["yuen"] = {
            "csv": yuen_csv,
            "plot": yuen_plot,
            "earliest_onset_s": earliest_onset(time_s, qvals, args.alpha),
        }

    manifest_path = save_manifest(manifest, args.out)
    LOG.info("Analysis complete. Outputs written to %s", os.path.abspath(args.out))
    LOG.info("Manifest saved to %s", manifest_path)


if __name__ == "__main__":  # pragma: no cover
    main()

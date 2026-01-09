"""Kaplan–Meier and Cox survival analyses for latency thresholds."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

from . import plotting
from .utils import GROUP_A_LABEL, GROUP_B_LABEL, latency_to_threshold

LOG = logging.getLogger("stats.survival")


@dataclass
class SurvivalOutcome:
    label: str
    threshold: float
    km_plot: str
    km_csv: str
    cox_txt: str
    logrank_stat: float
    logrank_p: float
    hazard_ratio: Optional[float]
    hr_ci: Optional[Tuple[float, float]]
    cox_p: Optional[float]


def _sanitize_label(label: str) -> str:
    return label.replace(" ", "_").replace(".", "p").replace("%", "pct").replace(",", "_")


def resolve_thresholds(
    matrix: np.ndarray,
    thresholds: Optional[Sequence[float]],
    percentiles: Optional[Sequence[float]],
    adaptive_k: Optional[float],
    baseline_window: Optional[Tuple[int, int]],
) -> List[Tuple[str, float]]:
    values: List[Tuple[str, float]] = []
    if thresholds:
        for value in thresholds:
            values.append((f"thr_{value:.3g}", float(value)))
    if percentiles:
        for perc in percentiles:
            thr = float(np.nanpercentile(matrix, perc))
            values.append((f"perc_{perc:.1f}", thr))
    if adaptive_k is not None:
        if baseline_window is None:
            raise ValueError("Adaptive threshold requested but --baseline-window not provided.")
        start, end = baseline_window
        if start < 0 or end <= start or end > matrix.shape[1]:
            raise ValueError("Baseline window out of bounds for adaptive threshold.")
        baseline = matrix[:, start:end]
        mean = float(np.nanmean(baseline))
        std = float(np.nanstd(baseline))
        thr = mean + adaptive_k * std
        values.append((f"adaptive_k{adaptive_k:.2f}", thr))
    return values


def _collect_trials(groups) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    trials_a: List[np.ndarray] = []
    trials_b: List[np.ndarray] = []
    fly_a: List[str] = []
    fly_b: List[str] = []
    for group in groups:
        if group.group_a.trials.size:
            trials_a.append(group.group_a.trials)
            fly_a.extend([group.fly_id] * group.group_a.trials.shape[0])
        if group.group_b.trials.size:
            trials_b.append(group.group_b.trials)
            fly_b.extend([group.fly_id] * group.group_b.trials.shape[0])
    return (
        np.vstack(trials_a) if trials_a else np.empty((0, groups[0].group_a.trials.shape[1])),
        np.vstack(trials_b) if trials_b else np.empty((0, groups[0].group_b.trials.shape[1])),
        fly_a,
        fly_b,
    )


def run_survival_sweep(
    groups,
    matrix: np.ndarray,
    time_hz: float,
    thresholds: List[Tuple[str, float]],
    search_window: Optional[Tuple[int, int]],
    out_dir: str,
) -> List[SurvivalOutcome]:
    trials_a, trials_b, fly_a, fly_b = _collect_trials(groups)
    if trials_a.size == 0 or trials_b.size == 0:
        raise ValueError("Survival sweep requires trials in both groups.")
    outcomes: List[SurvivalOutcome] = []
    for label, threshold in thresholds:
        LOG.info("Running survival analysis for threshold %s (%.4f)", label, threshold)
        lat_a, evt_a = latency_to_threshold(trials_a, threshold, search_window)
        lat_b, evt_b = latency_to_threshold(trials_b, threshold, search_window)
        lat_a_sec = lat_a / float(time_hz)
        lat_b_sec = lat_b / float(time_hz)
        df = pd.DataFrame(
            {
                "duration": np.concatenate([lat_a_sec, lat_b_sec]),
                "event": np.concatenate([evt_a.astype(int), evt_b.astype(int)]),
                "group": np.concatenate(
                    [np.ones(lat_a_sec.size, dtype=float), np.zeros(lat_b_sec.size, dtype=float)]
                ),
                "fly": fly_a + fly_b,
            }
        )
        cox_path = os.path.join(out_dir, f"{_sanitize_label(label)}_cox_summary.txt")
        try:
            cox = CoxPHFitter()
            cox.fit(df, duration_col="duration", event_col="event", strata=["fly"], formula="group")
            hr = float(np.exp(cox.params_["group"]))
            conf_int = cox.confidence_intervals_.loc["group"].values
            hr_ci = (float(np.exp(conf_int[0])), float(np.exp(conf_int[1])))
            p_value = float(cox.summary.loc["group", "p"])  # type: ignore[index]
            with open(cox_path, "w", encoding="utf-8") as handle:
                handle.write(cox.summary.to_string())
        except Exception as exc:
            LOG.warning("Cox model failed for threshold %s: %s", label, exc)
            hr = math.nan
            hr_ci = (math.nan, math.nan)
            p_value = math.nan
            with open(cox_path, "w", encoding="utf-8") as handle:
                handle.write(f"Cox model failed: {exc}\n")
        try:
            lr = logrank_test(lat_a_sec, lat_b_sec, event_observed_A=evt_a, event_observed_B=evt_b)
            logrank_stat = float(lr.test_statistic)
            logrank_p = float(lr.p_value)
        except Exception as exc:
            LOG.warning("Log-rank test failed for threshold %s: %s", label, exc)
            logrank_stat = math.nan
            logrank_p = math.nan
        km_plot = os.path.join(out_dir, f"{_sanitize_label(label)}_km_plot.png")
        median_a, median_b = plotting.plot_km(lat_a_sec, evt_a.astype(int), lat_b_sec, evt_b.astype(int), km_plot)
        km_csv = os.path.join(out_dir, f"{_sanitize_label(label)}_km_summary.csv")
        pd.DataFrame(
            {
                "group": [GROUP_A_LABEL, GROUP_B_LABEL],
                "n_trials": [lat_a_sec.size, lat_b_sec.size],
                "median_latency_s": [median_a, median_b],
                "events": [int(evt_a.sum()), int(evt_b.sum())],
                "logrank_stat": [logrank_stat, logrank_stat],
                "logrank_p": [logrank_p, logrank_p],
                "cox_hr": [hr, hr],
                "cox_p": [p_value, p_value],
            }
        ).to_csv(km_csv, index=False)
        outcomes.append(
            SurvivalOutcome(
                label=label,
                threshold=threshold,
                km_plot=km_plot,
                km_csv=km_csv,
                cox_txt=cox_path,
                logrank_stat=logrank_stat,
                logrank_p=logrank_p,
                hazard_ratio=hr,
                hr_ci=hr_ci,
                cox_p=p_value,
            )
        )
    return outcomes


__all__ = ["resolve_thresholds", "run_survival_sweep", "SurvivalOutcome"]

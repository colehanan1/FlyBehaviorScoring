#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic text summarizer for trained (A) vs untrained (B) odor analyses."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LOG = logging.getLogger("stats.summarize")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    """Return a DataFrame if *path* exists and can be parsed, else ``None``."""

    if not path or not os.path.exists(path):
        LOG.debug("CSV not found: %s", path)
        return None
    try:
        df = pd.read_csv(path)
        LOG.debug("Loaded CSV %s shape=%s", path, df.shape)
        return df
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("Failed to read CSV %s: %s", path, exc)
        return None


def safe_read_text(path: str) -> Optional[str]:
    """Return file contents if available; tolerate missing files."""

    if not path or not os.path.exists(path):
        LOG.debug("Text not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
        LOG.debug("Loaded text %s (%d chars)", path, len(text))
        return text
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("Failed to read text %s: %s", path, exc)
        return None


def col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Resolve the first matching column name in ``df`` for the given aliases."""

    for name in candidates:
        if name in df.columns:
            return name
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def bh_windows(time_s: np.ndarray, q: np.ndarray, alpha: float) -> List[Dict[str, float]]:
    """Return contiguous windows where ``q < alpha``.

    Parameters
    ----------
    time_s:
        Time axis in seconds.
    q:
        Adjusted p-values aligned to ``time_s``.
    alpha:
        Significance threshold.
    """

    if time_s is None or q is None or len(time_s) != len(q) or len(q) == 0:
        return []
    mask = np.asarray(q) < alpha
    if not np.any(mask):
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    windows: List[Dict[str, float]] = []
    for start, end in zip(starts, ends):
        end_idx = end - 1
        start_s = float(time_s[start])
        end_s = float(time_s[end_idx])
        duration = float(max(0.0, end_s - start_s))
        windows.append(
            {
                "start_idx": int(start),
                "end_idx": int(end_idx),
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": duration,
            }
        )
    return windows


def find_first_sig_onset(time_s: np.ndarray, q: np.ndarray, alpha: float) -> Optional[float]:
    if time_s is None or q is None:
        return None
    mask = np.asarray(q) < alpha
    if not np.any(mask):
        return None
    idx = int(np.where(mask)[0][0])
    return float(np.asarray(time_s)[idx])


def fmt(value, *, default: str = "—", precision: int = 3) -> str:
    """Render values into fixed-width strings for report consumption."""

    if value is None:
        return default
    if isinstance(value, float):
        if not np.isfinite(value):
            return default
        return f"{value:.{precision}f}"
    return str(value)


# ---------------------------------------------------------------------------
# Dataclasses for parsed output
# ---------------------------------------------------------------------------


@dataclass
class TimeSeriesSummary:
    earliest_onset_s: Optional[float]
    n_windows: int
    total_duration_s: float
    direction: Optional[str]
    notes: str = ""


@dataclass
class ClusterPermSummary:
    clusters: List[Dict]
    n_significant: int
    notes: str = ""


@dataclass
class SurvivalSummary:
    threshold_spec: str
    median_a: Optional[float]
    median_b: Optional[float]
    responses_a: Optional[int]
    responses_b: Optional[int]
    n_a: Optional[int]
    n_b: Optional[int]
    logrank_p: Optional[float]
    hazard_ratio: Optional[float]
    hazard_ci: Optional[Tuple[float, float]]
    cox_p: Optional[float]


@dataclass
class GAMSummary:
    interaction_p: Optional[float]
    peak_diff: Optional[float]
    peak_time_s: Optional[float]


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------


def summarize_timewise(path: str, alpha: float) -> TimeSeriesSummary:
    df = safe_read_csv(path)
    if df is None:
        return TimeSeriesSummary(None, 0, 0.0, None, notes=f"Missing: {os.path.basename(path)}")
    tcol = col(df, ["time_s", "time", "t"])
    qcol = col(df, ["q", "q_value", "fdr"])
    pcol = col(df, ["p", "p_value", "pval"])
    effect_cols = ["effect_mean_AminusB", "effect_mean_A_minus_B", "effect_median_AminusB", "effect_median_A_minus_B"]
    ecol = col(df, effect_cols)
    if tcol is None or (qcol is None and pcol is None):
        LOG.warning("Incomplete columns for %s", path)
        return TimeSeriesSummary(None, 0, 0.0, None, notes=f"Incomplete columns in {os.path.basename(path)}")
    q_values = df[qcol].values if qcol else df[pcol].values
    windows = bh_windows(df[tcol].values, q_values, alpha)
    earliest = find_first_sig_onset(df[tcol].values, q_values, alpha)
    direction = None
    if windows and ecol:
        sig_mask = np.asarray(q_values) < alpha
        if np.any(sig_mask):
            effect = np.asarray(df[ecol].values, dtype=float)
            mean_effect = np.nanmean(effect[sig_mask])
            if np.isfinite(mean_effect):
                if mean_effect > 0:
                    direction = "A>B"
                elif mean_effect < 0:
                    direction = "B>A"
    total_duration = float(sum(win["duration_s"] for win in windows))
    return TimeSeriesSummary(earliest, len(windows), total_duration, direction)


def summarize_cluster_perm(timewise_path: str, clusters_path: str, alpha: float) -> ClusterPermSummary:
    d_time = safe_read_csv(timewise_path)
    d_clusters = safe_read_csv(clusters_path)
    if d_time is None or d_clusters is None:
        return ClusterPermSummary([], 0, notes="Cluster permutation outputs missing.")
    cid = col(d_time, ["cluster_id", "cluster"])
    pcol = col(d_time, ["p_cluster", "cluster_p"])
    if cid is None or pcol is None:
        return ClusterPermSummary([], 0, notes="Cluster permutation file lacks cluster identifiers.")
    sig_ids = sorted(set(d_time.loc[d_time[pcol] < alpha, cid].dropna().astype(int)))
    df = d_clusters.copy()
    if "p_cluster" not in df.columns and "cluster_p" in df.columns:
        df.rename(columns={"cluster_p": "p_cluster"}, inplace=True)
    if "p_cluster" in df.columns:
        df = df[df["p_cluster"] < alpha]
    df = df.sort_values([c for c in ["p_cluster", "start_s"] if c in df.columns])
    return ClusterPermSummary(df.to_dict(orient="records"), len(sig_ids))


def parse_logrank_p(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    match = re.search(r"log[- ]?rank.*?p[=\s:]+([0-9.]+(?:e-?\d+)?)", text, flags=re.I | re.S)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def parse_cox_summary(text: Optional[str]) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[float]]:
    if not text:
        return (None, None, None)
    hr = ci = p_val = None
    match_hr = re.search(r"(?:HR|hazard\s*ratio)\s*[:=]\s*([0-9.]+)", text, flags=re.I)
    if match_hr:
        try:
            hr = float(match_hr.group(1))
        except ValueError:
            hr = None
    match_ci = re.search(r"(?:CI|confidence\s*interval)[^0-9]*([0-9.]+)\s*[-–]\s*([0-9.]+)", text, flags=re.I)
    if match_ci:
        try:
            ci = (float(match_ci.group(1)), float(match_ci.group(2)))
        except ValueError:
            ci = None
    match_p = re.search(r"\bp\s*[:=]\s*([0-9.]+(?:e-?\d+)?)", text, flags=re.I)
    if match_p:
        try:
            p_val = float(match_p.group(1))
        except ValueError:
            p_val = None
    return hr, ci, p_val


def summarize_survival(km_csv: str, cox_txt: str, label: str) -> SurvivalSummary:
    df = safe_read_csv(km_csv)
    text = safe_read_text(cox_txt)
    hr, ci, cox_p = parse_cox_summary(text)
    logrank_p = parse_logrank_p(text)
    median_a = median_b = resp_a = resp_b = n_a = n_b = None
    if df is not None:
        gcol = col(df, ["group"])
        mcol = col(df, ["median_latency_s", "median", "median_s"])
        ncol = col(df, ["n_trials", "n", "count"])
        ecol = col(df, ["events", "n_events", "responses", "responded"])
        if gcol and mcol:
            mask_a = df[gcol].astype(str).str.contains("A", case=False, na=False)
            mask_b = df[gcol].astype(str).str.contains("B", case=False, na=False)
            if mask_a.any():
                median_a = float(df.loc[mask_a, mcol].values[0])
            if mask_b.any():
                median_b = float(df.loc[mask_b, mcol].values[0])
        if gcol and ncol:
            mask_a = df[gcol].astype(str).str.contains("A", case=False, na=False)
            mask_b = df[gcol].astype(str).str.contains("B", case=False, na=False)
            if mask_a.any():
                n_a = int(df.loc[mask_a, ncol].values[0])
            if mask_b.any():
                n_b = int(df.loc[mask_b, ncol].values[0])
        if gcol and ecol:
            mask_a = df[gcol].astype(str).str.contains("A", case=False, na=False)
            mask_b = df[gcol].astype(str).str.contains("B", case=False, na=False)
            if mask_a.any():
                resp_a = int(df.loc[mask_a, ecol].values[0])
            if mask_b.any():
                resp_b = int(df.loc[mask_b, ecol].values[0])
        if logrank_p is None and col(df, ["logrank_p", "logrank_p_value"]):
            try:
                logrank_p = float(df.loc[df.index[0], col(df, ["logrank_p", "logrank_p_value"])] )
            except Exception:  # pragma: no cover - fallback
                pass
    return SurvivalSummary(
        threshold_spec=label,
        median_a=median_a,
        median_b=median_b,
        responses_a=resp_a,
        responses_b=resp_b,
        n_a=n_a,
        n_b=n_b,
        logrank_p=logrank_p,
        hazard_ratio=hr,
        hazard_ci=ci,
        cox_p=cox_p,
    )


def summarize_gam(predicted_csv: str, summary_txt: str) -> GAMSummary:
    text = safe_read_text(summary_txt)
    interaction_p = None
    if text:
        match = re.search(r"Condition.*Time.*?p\s*[:=]\s*([0-9.]+(?:e-?\d+)?)", text, flags=re.I | re.S)
        if match:
            try:
                interaction_p = float(match.group(1))
            except ValueError:
                interaction_p = None
    df = safe_read_csv(predicted_csv)
    peak_diff = peak_time = None
    if df is not None:
        tcol = col(df, ["time_s", "time", "t"])
        dcol = col(df, ["diff", "pred_diff", "predicted_difference"])
        if tcol and dcol and len(df):
            diff_values = np.asarray(df[dcol].values, dtype=float)
            finite_mask = np.isfinite(diff_values)
            if np.any(finite_mask):
                idx = int(np.nanargmax(np.abs(diff_values[finite_mask])))
                finite_times = np.asarray(df[tcol].values, dtype=float)[finite_mask]
                peak_diff = float(np.abs(diff_values[finite_mask])[idx])
                peak_time = float(finite_times[idx])
    return GAMSummary(interaction_p=interaction_p, peak_diff=peak_diff, peak_time_s=peak_time)


def summarize_manifest(manifest_path: str) -> Dict[str, str]:
    manifest = safe_read_text(manifest_path)
    if manifest is None:
        return {}
    try:
        data = json.loads(manifest)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        LOG.warning("Failed to parse manifest %s", manifest_path)
    return {}


def render_template(template_path: str, context: Dict[str, str]) -> str:
    template = safe_read_text(template_path)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    try:
        return template.format(**context)
    except KeyError as exc:  # pragma: no cover - template safety
        LOG.warning("Template missing key: %s", exc)
        return template


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize stats outputs into a human-readable report.")
    parser.add_argument("--in", dest="input_dir", required=True, help="Directory containing stats outputs.")
    parser.add_argument("--out", dest="output_path", required=True, help="Path to write summary text.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for interpretation.")
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel), format="[%(levelname)s] %(message)s")

    input_dir = args.input_dir
    alpha = args.alpha
    LOG.info("Summarizing outputs in %s (alpha=%.3f)", input_dir, alpha)

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    wilcoxon = summarize_timewise(os.path.join(input_dir, "wilcoxon.csv"), alpha)
    paired_t = summarize_timewise(os.path.join(input_dir, "paired_t.csv"), alpha)
    randomization = summarize_timewise(os.path.join(input_dir, "randomization.csv"), alpha)
    yuen = summarize_timewise(os.path.join(input_dir, "yuen.csv"), alpha)
    mcnemar = summarize_timewise(os.path.join(input_dir, "mcnemar.csv"), alpha)

    cluster_perm = summarize_cluster_perm(
        os.path.join(input_dir, "cluster_perm_timewise.csv"),
        os.path.join(input_dir, "cluster_perm_clusters.csv"),
        alpha,
    )

    survival_summaries: List[SurvivalSummary] = []
    for prefix, label in [
        ("adaptive_k3p00", "adaptive baseline μ+3σ"),
        ("perc_90p0", "percentile 90%"),
        ("perc_95p0", "percentile 95%"),
    ]:
        km_csv = os.path.join(input_dir, f"{prefix}_km_summary.csv")
        cox_txt = os.path.join(input_dir, f"{prefix}_cox_summary.txt")
        if os.path.exists(km_csv) or os.path.exists(cox_txt):
            survival_summaries.append(summarize_survival(km_csv, cox_txt, label))

    gam = summarize_gam(
        os.path.join(input_dir, "gam_predicted_diff.csv"),
        os.path.join(input_dir, "gam_mixed_summary.txt"),
    )

    cluster_lines = []
    for cluster in cluster_perm.clusters:
        start = fmt(cluster.get("start_s"))
        end = fmt(cluster.get("end_s"))
        duration = fmt(cluster.get("duration_s"))
        sign = cluster.get("sign", "?")
        p_cluster = fmt(cluster.get("p_cluster")) if "p_cluster" in cluster else fmt(cluster.get("cluster_p"))
        cluster_lines.append(
            f"- {start}–{end} s (dur {duration} s), sign={sign}, p_cluster={p_cluster}"
        )
    cluster_block = "\n".join(cluster_lines) if cluster_lines else "None"

    survival_lines = []
    for item in survival_summaries:
        hr_text = fmt(item.hazard_ratio)
        if item.hazard_ci:
            hr_text = f"{hr_text} [{fmt(item.hazard_ci[0])}, {fmt(item.hazard_ci[1])}]"
        med_a = "∞" if (item.median_a is not None and np.isinf(item.median_a)) else fmt(item.median_a)
        med_b = "∞" if (item.median_b is not None and np.isinf(item.median_b)) else fmt(item.median_b)
        resp_a = f"{fmt(item.responses_a, default='?')}/{fmt(item.n_a, default='?')}"
        resp_b = f"{fmt(item.responses_b, default='?')}/{fmt(item.n_b, default='?')}"
        survival_lines.append(
            f"- {item.threshold_spec}: median A={med_a}, B={med_b}; responses A={resp_a}, B={resp_b}; "
            f"HR={hr_text}, Cox p={fmt(item.cox_p)}, log-rank p={fmt(item.logrank_p)}"
        )
    survival_block = "\n".join(survival_lines) if survival_lines else "None"

    context: Dict[str, str] = {
        "alpha": fmt(alpha),
        "cl_perm_n": str(cluster_perm.n_significant),
        "cl_perm_clusters": cluster_block,
        "wil_onset": fmt(wilcoxon.earliest_onset_s),
        "wil_ncl": str(wilcoxon.n_windows),
        "wil_totdur": fmt(wilcoxon.total_duration_s),
        "t_onset": fmt(paired_t.earliest_onset_s),
        "t_ncl": str(paired_t.n_windows),
        "t_totdur": fmt(paired_t.total_duration_s),
        "rz_onset": fmt(randomization.earliest_onset_s),
        "rz_ncl": str(randomization.n_windows),
        "rz_totdur": fmt(randomization.total_duration_s),
        "yuen_onset": fmt(yuen.earliest_onset_s),
        "yuen_ncl": str(yuen.n_windows),
        "yuen_totdur": fmt(yuen.total_duration_s),
        "mcn_onset": fmt(mcnemar.earliest_onset_s),
        "mcn_ncl": str(mcnemar.n_windows),
        "mcn_totdur": fmt(mcnemar.total_duration_s),
        "survival_block": survival_block,
        "gam_p": fmt(gam.interaction_p),
        "gam_peak": fmt(gam.peak_diff),
        "gam_peakt": fmt(gam.peak_time_s),
    }

    manifest = summarize_manifest(os.path.join(input_dir, "manifest.json"))
    if manifest:
        LOG.debug("Manifest entries: %s", list(manifest.keys()))

    template_path = os.path.join(os.path.dirname(__file__), "templates", "summary_template.txt")
    report = render_template(template_path, context)

    with open(args.output_path, "w", encoding="utf-8") as handle:
        handle.write(report)
    LOG.info("Wrote summary report to %s", args.output_path)


if __name__ == "__main__":
    main()


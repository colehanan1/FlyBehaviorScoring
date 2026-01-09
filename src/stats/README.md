# Fly Behavior Time-Series Statistics

The `stats` package provides a CLI-driven toolkit for running paired, fly-centric
comparisons on proboscis distance traces. Analyses operate directly on the
uncollapsed time-series, respecting the fly as the experimental unit.

## Methods Overview

- **Per-timepoint tests** – Paired t/Wilcoxon or Welch/Mann–Whitney when
  pairing is impossible. Outputs Benjamini–Hochberg–corrected p-series and
  effect curves.
- **McNemar (sign) test** – Counts across-fly direction reversals at each
  timepoint to test binary discordance.
- **Cluster-based permutation** – Time-adjacent clusters of supra-threshold
  test statistics are evaluated against a within-fly label-swap null.
- **Randomization test** – Time-wise sign-flip resampling of fly differences
  using median or mean statistics, with BH-FDR across time.
- **Mixed-effects GAM** – Linear mixed model with condition × spline(time)
  interactions and fly-level random intercepts to assess trajectory-wide
  deviations.
- **Survival sweeps** – Latency-to-threshold Kaplan–Meier curves, stratified
  Cox proportional hazards, and log-rank tests across multiple thresholds
  (numeric, percentile, or baseline-adaptive).
- **Equivalence (TOST)** – Paired two one-sided tests within a pre-odor
  interval using user-specified or baseline-scaled equivalence margins.
- **Yuen trimmed-mean** – Robust trimmed-mean t-statistics per timepoint with
  BH-FDR adjustment.

All plots and tables are written to disk; a manifest summarizes significant
onsets, cluster counts, and file paths.

## CLI Examples

Baseline outputs only:

```bash
python stats/run_all.py \
  --npy path/to/envelope_matrix_float16.npy \
  --meta path/to/code_maps.json \
  --out outputs/stats \
  --datasets testing \
  --target-trials 2,4,5 \
  --time-hz 40
```

Full pipeline with advanced analyses:

```bash
python stats/run_all.py \
  --npy path/to/envelope_matrix_float16.npy \
  --meta path/to/code_maps.json \
  --out outputs/stats \
  --datasets testing \
  --target-trials 2,4,5 \
  --time-hz 40 \
  --do-cluster-perm --cluster-stat t --n-perm 1000 \
  --do-randomization --rand-stat median --rand-n-perm 2000 \
  --do-gam --gam-knots 12 \
  --do-survival-sweep --km-percentiles 90,95 --km-adaptive-k 3 \
  --baseline-window 0:1200 --km-window 0:2400 \
  --do-equivalence --equiv-margin-mult 0.5 --pre-window 0:1200 \
  --do-yuen --trim-pct 0.2
```

## Interpretation Notes

- **Pairing** – Every analysis treats the fly as the unit of replication.
  Trials are averaged per fly (Group A vs Group B) before inferential tests.
- **Multiple comparisons** – Time-wise outputs (t/Wilcoxon, randomization,
  Yuen) provide BH-FDR-adjusted q-values. Clusters use permutation-derived
  family-wise correction.
- **Cluster tests** – Interpret cluster p-values at the cluster level; individual
  points inside a significant cluster are not independently controlled.
- **Survival models** – Cox models stratify by fly to respect repeated
  measures. Log-rank tests are reported with the available (unstratified)
  implementation and flagged if fitting fails.
- **Equivalence** – Supply a scientifically defensible margin. Using the
  baseline SD multiplier ties the margin to pre-odor variability.
- **Small sample caveats** – When fewer than two flies remain after filtering,
  analyses fall back to unpaired tests or warn and skip (e.g., McNemar,
  cluster permutation). Review logs for dropped flies and missing metadata.
- **Autocorrelation** – Time-series autocorrelation inflates pointwise test
  counts; prefer cluster-level inference or GAM/randomization summaries for
  trajectory-wide conclusions.

## Outputs

Each run produces CSV/PNG artifacts plus `manifest.json`, which records
significance onsets, cluster durations, survival sweep highlights, and paths to
all generated files. Examine DEBUG logs for per-fly trial counts, metadata
normalization steps, and any analytic fallbacks.

# Fly Behavior Scoring & ML Starter

A ready-to-run, Git-ready project to:
- Label fly behavior videos with a 0–5 human score via a simple GUI.
- Compute data-driven metrics (e.g., time above threshold, AUC, reaction latency, rise speed) and a data score.
- Persist meticulous, per-trial metadata to a single master CSV.
- Train a baseline PyTorch regressor to predict scores from metrics (outputs 0–5 after rounding).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Unsupervised trace-only clustering

Run the unsupervised models on trace data via:

```bash
python unsup/run_all.py \
  --npy path/to/envelope_matrix_float16.npy \
  --meta path/to/code_maps.json \
  --out outputs/unsup
```

Optional arguments:

- `--min-cluster-size`: minimum cluster size for the density-based model (default `5`).
- `--seed`: random seed used for PCA and clustering (default `0`).
- `--max-pcs`: cap on retained principal components (default `10`).
- `--min-clusters`/`--max-clusters`: bounds for the adaptive k-means/GMM cluster-count search (defaults `2`/`10`).
- `--datasets`: space-separated dataset names to retain (default `EB 3-octonol`).
- `--debug`: print verbose filtering/PCA/clustering diagnostics to stdout.

The k-means backend chooses the cluster count that maximizes the silhouette score within the provided bounds, and the Gaussian mixture backend picks the component count that minimizes the Bayesian Information Criterion (BIC). In addition to the PCA-driven models, the run now includes odor-aligned reaction profiling backends that focus on the odor-on window (frames 1,230–2,430) and the subsequent odor-off recovery period.

Models executed per run:

1. **`simple`** – PCA + k-means on the 80 % variance PCs (adaptive cluster count).
2. **`flexible`** – PCA + Gaussian mixture with BIC-based component selection.
3. **`noise_robust`** – PCA + HDBSCAN (DBSCAN fallback) with noise labeling.
4. **`reaction_motifs`** – odor-window feature extraction + NMF motifs + k-means clustering.
5. **`reaction_clusters`** – the same odor-window feature extraction clustered without motif decomposition.

Each invocation creates `outputs/unsup/YYYYMMDD_HHMMSS/` containing:

- `timepoint_importance.csv`: average absolute PCA loadings over the most informative PCs.
- `report_<model>.csv`: one-row metric summary per model (`simple`, `flexible`, `noise_robust`, `reaction_motifs`, `reaction_clusters`).
- `trial_clusters_<model>.csv`: filtered metadata with an added `cluster_label` column (noise marked `-1`).
- `pca_variance_<model>.png`: explained-variance bar chart with cumulative curve.
- `time_importance_<model>.png`: timepoint importance line plot.
- `embedding_<model>.png`: 2-D embedding scatter colored by cluster labels (noise appears as grey crosses).
- `motifs_reaction_motifs.csv` / `motifs_reaction_motifs.png`: odor-aligned temporal motifs recovered by the NMF-based model.

Input arrays/metadata are consumed at runtime and are not committed to the repository.

### 1) Label / score videos

```bash
python label_videos.py   --videos /path/to/videos   --data   /path/to/csvs   --output scoring_results.csv
```

- The GUI shows **video first** with a 0–5 selector.
- After you submit, it reveals metrics and **data-derived score** (weights configurable).
- All rows are appended to a single **master CSV** with fly/trial/video/user score/data score/combined score and raw metrics.

### 2) Train the baseline model

```bash
python train_model.py --data scoring_results.csv --epochs 100 --output-model fly_score_model.pth
```

- Predicts a continuous score that you should **round to 0–5** for reporting.
- GPU is used automatically when available (PyTorch + CUDA).

## Configure metric weights / threshold

Edit these at the top of `label_videos.py`:
- `THRESHOLD`: the reaction threshold in your signal units.
- `METRIC_WEIGHTS`: relative weights for metrics (e.g., make `time_fraction` or `auc` dominate).

## Summarise trained vs. untrained odor responses

Use the `stats/odor_response_contingency.py` utility to extract the four-cell
contingency table described in the project brief (``a`` through ``d``) from a
behavioural CSV export. The script focuses on the `during_hit` column and lets
you specify which trial numbers correspond to the trained and untrained odours.

```bash
python -m stats.odor_response_contingency \
  data/odor_trials.csv \
  "EB 3-octonol" \
  --trained-trials 2 4 5 \
  --untrained-trials 1 3 6 \
  --output outputs/odor_summary.png
```

- Provide the dataset name exactly as it appears in the `dataset` column.
- The trained trial list defaults to `2 4 5`; override it if your protocol
  differs.
- Pass whichever untrained trial numbers you want to include, omitting any you
  wish to exclude (e.g., skip 7–10 entirely as shown above).
- When `--output` is supplied, the script renders a PNG mirroring the 2×2
  contingency table; omit the flag to skip figure generation.

The command prints the counts for the four cases (`a` through `d`) along with
the total number of flies that match the configured trials.

## File associations

Videos and CSVs are matched by **base filename**. Example:
- `fly2736_testing_3.mp4` ↔ `fly2736_testing_3.csv`

The tool also tries to parse `fly_id` and `trial_id` from folder/filename digits.

## Note

This is intentionally minimal and extensible for PyCharm/Linux workflows. Extend `compute_metrics()` with more features and add them to `METRIC_WEIGHTS` to include them everywhere (CSV + training).

# FlyPCA

FlyPCA provides a reproducible, event-aligned lag-embedded PCA workflow for Drosophila proboscis-distance time series. The package smooths and baseline-normalizes traces, performs Hankel (time-delay) embedding, learns compact principal components, derives interpretable behavioral features, and clusters trials into reaction vs. non-reaction cohorts.

## Pipeline Overview

1. **Ingest** trial CSVs or manifests (trial_id, fly_id, distance, odor indices).
2. **Preprocess** each trial with Savitzky–Golay smoothing, optional low-pass filtering, and pre-odor z-scoring.
3. **Lag Embed & PCA** using Hankel matrices to preserve local temporal structure; fit PCA or IncrementalPCA.
4. **Project** trials into PC trajectories aligned to odor onset.
5. **Engineer Features** capturing temporal dynamics, velocity, Hilbert envelope, frequency bands, and PC-space summaries.
6. **Cluster & Evaluate** with GMM or HDBSCAN and compute silhouette, Calinski–Harabasz, AUROC, and AUPRC (leave-one-fly-out).
7. **Visualize & Report** scree plots, loadings, trajectories, cluster scatter, violin plots, and markdown reports.

## Quickstart

```bash
make venv
source .venv/bin/activate
make install
make test
```

Generate a synthetic demo dataset and full report:

```bash
make demo
```

## Running on Real Data

1. **Assemble a manifest or wide CSV** describing each trial.
   - *Stacked format*: one row per timepoint with columns `trial_id`, `fly_id`, `distance`, `odor_on_idx`, optional `odor_off_idx`, optional `time`, and optional `fps`.
   - *Wide format*: one row per trial where the time series samples occupy columns with a consistent prefix (e.g., `dir_val_0`, `dir_val_1`, …). Provide metadata columns for trial identity, fly identity, odor indices, and fps.
2. **Map column names in the config**. Copy `configs/default.yaml` and update the `io` section to match your data. Example for the wide file shown in the error transcript:

   ```yaml
   io:
     format: wide
     read_csv:
       low_memory: false
       dtype:
         trial_label: str
     wide:
       trial_id_column: trial_label
       trial_id_template: "{fly}_{trial_label}"
       fly_id_column: fly
       fps_column: fps
       odor_on_value: 1230
       odor_off_value: 2430
       time_columns:
         prefix: dir_val_
   ```

   Setting `dtype` ensures pandas does not emit mixed-type warnings. For stacked data, adjust `io.stacked.distance_column`, `io.stacked.time_column`, etc., instead.
3. **Verify indices**: `odor_on_idx` and `odor_off_idx` are frame indices (0-based). They must be within `[0, n_frames)` and `odor_on_idx < odor_off_idx`. Ensure the time column is strictly increasing if present; for wide data the loader generates time stamps using `fps`.
4. **Run the CLI pipeline**. The commands below fit the lag-embedded PCA model, project each trial, engineer features, cluster reactions, and generate a Markdown report with key plots.

```bash
flypca fit-lag-pca \
  --data data/manifest.csv \
  --config configs/default.yaml \
  --out artifacts/models/lagpca.joblib

flypca project \
  --model artifacts/models/lagpca.joblib \
  --data data/manifest.csv \
  --out artifacts/projections/

flypca features \
  --data data/manifest.csv \
  --config configs/default.yaml \
  --model artifacts/models/lagpca.joblib \
  --projections artifacts/projections/ \
  --out artifacts/features.parquet

flypca cluster \
  --features artifacts/features.parquet \
  --config configs/default.yaml \
  --projections-dir artifacts/projections/ \
  --method gmm \
  --out artifacts/cluster.csv \
  --labels-path data/labels.csv \
  --labels-column-name user_score_odor \
  --label-column user_score_odor

flypca report \
  --features artifacts/features.parquet \
  --clusters artifacts/cluster.csv \
  --model artifacts/models/lagpca.joblib \
  --projections artifacts/projections/ \
  --out-dir artifacts/
```

Outputs are written under `artifacts/` by default: the trained PCA model (`models/`), projected PC trajectories (`projections/`), engineered features (`features.parquet`), clustering assignments, summary figures (`figures/`), and a Markdown report describing variance explained, cluster metrics, and representative trajectories.

CLI entry points (Typer-based):

```bash
flypca fit-lag-pca --data data/manifest.csv --config configs/default.yaml --out artifacts/models/lagpca.joblib
flypca project --model artifacts/models/lagpca.joblib --data data/manifest.csv --out artifacts/projections/
flypca features --data data/manifest.csv --config configs/default.yaml --model artifacts/models/lagpca.joblib --projections artifacts/projections/ --out artifacts/features.parquet
flypca cluster --features artifacts/features.parquet --config configs/default.yaml --projections-dir artifacts/projections/ --method gmm --out artifacts/cluster.csv --label-column reaction

# cluster with label CSV
flypca cluster \
  --features artifacts/features.parquet \
  --config configs/default.yaml \
  --projections-dir artifacts/projections/ \
  --labels-path data/labels.csv \
  --labels-column-name user_score_odor \
  --out artifacts/cluster.csv
flypca report --features artifacts/features.parquet --clusters artifacts/cluster.csv --model artifacts/models/lagpca.joblib --projections artifacts/projections/ --out-dir artifacts/
```

### Clustering configuration

- `standardize`: z-score the feature/projection matrix before fitting the mixture model (enabled by default).
- `min_variance`: drop near-constant columns prior to clustering to prevent degeneracy.
- `component_range`: sweep a range of Gaussian mixture sizes (inclusive) and pick the lowest-BIC model with a valid silhouette.
- `covariance_types`: evaluate multiple covariance structures (`full`, `diag`, etc.) during the sweep.
- `use_projections`: `auto` by default; if projections are supplied they are incorporated automatically, otherwise the feature table alone is clustered. Set to `true` or `false` to force behaviour.
- `combine_with_features`: `auto` by default; when projections are used they are concatenated with engineered features unless explicitly disabled.
- `projection_components` / `projection_timepoints`: cap how many PCs and aligned samples are flattened from the NPZ files.

Label CSVs can be merged on-the-fly using `--labels-path` and `--labels-column-name`. The helper derives `trial_id` values by
applying the configured template (e.g. `{fly}_{trial_label}`) or, if absent, by combining `fly` and `trial_label` columns. The merged column is available for clustering diagnostics and supervised AUROC/AUPRC evaluation.

When `use_projections` is enabled the CLI expects `projections/manifest.csv` (written by `flypca project`) so trial IDs can be matched automatically.

Expected data layout for manifests:

```
manifest.csv:
path,trial_id,fly_id,odor_on_idx,odor_off_idx,fps
trial001.csv,tr1,flyA,80,120,40
...

trial001.csv:
frame,time,distance
0,0.00,1.23
...
```

## Testing & Quality

- Type-annotated, vectorized preprocessing and feature routines.
- Deterministic seeds; logging records parameter settings and array shapes.
- Pytest suite covers preprocessing, PCA embedding, feature extraction, and end-to-end synthetic performance (AUROC > 0.8).

## Interpreting PCs

- PC1 typically correlates with response amplitude and integrates the rising phase post-odor.
- PC2 captures latency and decay kinetics when present.
- Time-aligned PC trajectories and feature table outputs (parquet) enable downstream classifiers or visualization in standard tools.

## Make Targets

- `make venv`: create `.venv` using Python 3.11.
- `make install`: install flypca in editable mode with requirements.
- `make test`: run unit tests (`pytest -q`).
- `make demo`: synthesize data, run the full CLI pipeline, and emit artifacts (models, projections, features, clusters, figures, report).

Refer to `examples/01_synthetic_demo.ipynb` for a notebook walkthrough replicating the pipeline with code and inline commentary.

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

CLI entry points (Typer-based):

```bash
flypca fit-lag-pca --data data/manifest.csv --config configs/default.yaml --out artifacts/models/lagpca.joblib
flypca project --model artifacts/models/lagpca.joblib --data data/manifest.csv --out artifacts/projections/
flypca features --data data/manifest.csv --config configs/default.yaml --model artifacts/models/lagpca.joblib --projections artifacts/projections/ --out artifacts/features.parquet
flypca cluster --features artifacts/features.parquet --method gmm --out artifacts/cluster.csv --label-column reaction
flypca report --features artifacts/features.parquet --clusters artifacts/cluster.csv --model artifacts/models/lagpca.joblib --projections artifacts/projections/ --out-dir artifacts/
```

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

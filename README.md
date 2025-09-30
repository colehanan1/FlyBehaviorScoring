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

## File associations

Videos and CSVs are matched by **base filename**. Example:
- `fly2736_testing_3.mp4` ↔ `fly2736_testing_3.csv`

The tool also tries to parse `fly_id` and `trial_id` from folder/filename digits.

## Note

This is intentionally minimal and extensible for PyCharm/Linux workflows. Extend `compute_metrics()` with more features and add them to `METRIC_WEIGHTS` to include them everywhere (CSV + training).

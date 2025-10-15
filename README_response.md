# flybehavior_response

This package trains, evaluates, and visualizes supervised models that predict fly odor responses from proboscis traces and engineered features.

## Installation

```bash
pip install -e .
```

## Command Line Interface

After installation, the `flybehavior-response` command becomes available. Common arguments:

- `--data-csv`: Wide proboscis trace CSV.
- `--labels-csv`: Labels CSV with `user_score_odor` scores (0 = no response, 1-5 = increasing response strength).
- `--features`: Comma-separated engineered feature list (default: `AUC-During,TimeToPeak-During,Peak-Value`).
- `--include-auc-before`: Adds `AUC-Before` to the feature set.
- `--use-raw-pca` / `--no-use-raw-pca`: Toggle raw trace PCA (default enabled).
- `--n-pcs`: Number of PCA components (default 5).
- `--model`: `lda`, `logreg`, `mlp`, `both`, or `all` (default `all`).
- `--logreg-solver`: Logistic regression solver (`lbfgs`, `liblinear`, `saga`; default `lbfgs`).
- `--logreg-max-iter`: Iteration cap for logistic regression (default `1000`; increase if convergence warnings appear).
- `--cv`: Stratified folds for cross-validation (default 0 for none).
- `--artifacts-dir`: Root directory for outputs (default `./artifacts`).
- `--plots-dir`: Plot directory (default `./artifacts/plots`).
- `--seed`: Random seed (default 42).
- `--dry-run`: Validate pipeline without saving artifacts.
- `--verbose`: Enable DEBUG logging.

### Subcommands

| Command | Purpose |
| --- | --- |
| `prepare` | Validate inputs, report class balance and intensity distribution, write merged parquet. |
| `train` | Fit preprocessing + models, compute metrics, save joblib/config/metrics. |
| `eval` | Reload saved models and recompute metrics on merged data. |
| `viz` | Generate PC scatter, LDA score histogram, and ROC curve (if available). |
| `predict` | Score a merged CSV with a saved model and write predictions. |

### Examples

```bash
flybehavior-response prepare --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv

flybehavior-response train --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv --model all --n-pcs 5

flybehavior-response eval --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv

# explicitly evaluate a past run directory
flybehavior-response eval --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv --run-dir artifacts/2025-10-14T22-56-37Z

flybehavior-response viz --data-csv /home/ramanlab/Documents/cole/Data/Opto/Combined/all_envelope_rows_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv --plots-dir artifacts/plots

flybehavior-response predict --data-csv merged.csv --model-path artifacts/<run>/model_logreg.joblib \
  --output-csv artifacts/predictions.csv
```

## Training with the MLP classifier

- `--model all` trains LDA, logistic regression, and the new MLP classifier using a shared stratified 80/20 split and writes per-model confusion matrices into the run directory.
- Each training run now exports `predictions_<model>_{train,test}.csv` so you can audit which trials were classified correctly, along with their reaction probabilities and sample weights.
- `--model mlp` isolates the neural network if you want to iterate quickly without re-fitting the classical baselines.
- Existing scripts that still pass `--model both` continue to run LDA + logistic regression only; update them to `--model all` to include the MLP.
- Inspect `metrics.json` for `test` entries to verify held-out accuracy/F1 scores, and review `confusion_matrix_<model>.png` in the run directory for quick diagnostics.

## Preparing raw coordinate inputs

- Use the Typer subcommand to convert per-trial eye/proboscis traces into a modeling-ready CSV with metadata and optional `dir_val` distances:

  ```bash
  flybehavior-response prepare-raw \
    --data-csv /home/ramanlab/Documents/cole/Data/Opto/all_eye_prob_coords_per_trial.csv \
    --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
    --out /home/ramanlab/Documents/cole/Data/Opto/Combined/all_eye_prob_coords_prepared.csv \
    --fps 40 --odor-on-idx 1230 --odor-off-idx 2430 \
    --truncate-before 0 --truncate-after 0 \
    --series-prefixes "eye_x_f,eye_y_f,prob_x_f,prob_y_f" \
    --no-compute-dir-val
  ```
- If your acquisition exports trials as a 3-D NumPy array (trials × frames × 4 channels), save the matrix to `.npy` and provide a JSON metadata file describing each trial and the layout:

  ```bash
  flybehavior-response prepare-raw \
    --data-npy /home/ramanlab/Documents/cole/Data/Opto/all_eye_prob_coords_matrix.npy \
    --matrix-meta /home/ramanlab/Documents/cole/Data/Opto/all_eye_prob_coords_matrix.json \
    --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
    --out /home/ramanlab/Documents/cole/Data/Opto/Combined/all_eye_prob_coords_prepared.csv
  ```
  The metadata JSON must contain a `metadata` (or `trials`) array with per-row descriptors (`dataset`, `fly`, `fly_number`, `trial_type`, `trial_label` – legacy exports may name this `testing_trial` and will be auto-renamed), an optional `layout` field (`trial_time_channel` or `trial_channel_time`), and optional `channel_prefixes` that match the prefixes passed via `--series-prefixes`.
- The output keeps raw values with consistent 0-based frame indices per prefix, adds timing metadata, and can be fed directly to `flybehavior-response train --raw-series` (or an explicit `--series-prefixes eye_x_f,eye_y_f,prob_x_f,prob_y_f` if you customise the channel order).
- All subcommands (`prepare`, `train`, `eval`, `viz`, `predict`) accept `--raw-series` to prioritise the four eye/proboscis channels. When left unset, the loader still auto-detects the raw prefixes whenever `dir_val_` traces are absent, so legacy scripts continue to run unchanged.

### Running the modeling pipeline on raw coordinates

Once you have a wide table of raw coordinates, enable the raw channel handling on every CLI entry point with `--raw-series` (or supply an explicit `--series-prefixes` string if you re-ordered the channels):

```bash
# train all models on raw coordinates (engineered feature list is ignored automatically)
flybehavior-response train --raw-series \
  --data-csv /home/ramanlab/Documents/cole/Data/Opto/all_eye_prob_coords_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
  --model all --n-pcs 5

# evaluate an existing run against the same raw inputs
flybehavior-response eval --raw-series \
  --data-csv /home/ramanlab/Documents/cole/Data/Opto/all_eye_prob_coords_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
  --run-dir artifacts/<timestamp>

# regenerate confusion matrices and PCA/ROC plots for the raw-trained models
flybehavior-response viz --raw-series \
  --data-csv /home/ramanlab/Documents/cole/Data/Opto/all_eye_prob_coords_wide.csv \
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv \
  --run-dir artifacts/<timestamp>

# score new raw trials with a saved pipeline
flybehavior-response predict --raw-series \
  --data-csv /home/ramanlab/Documents/cole/Data/Opto/all_eye_prob_coords_wide.csv \
  --model-path artifacts/<timestamp>/model_logreg.joblib \
  --output-csv artifacts/<timestamp>/raw_predictions.csv
```

During training the loader automatically recognises that engineered features are absent and logs that it is proceeding in a trace-only configuration. Keep PCA enabled (`--use-raw-pca`, the default) to derive compact principal components from the four coordinate streams.

### Running without engineered features on legacy `dir_val_` traces

Older exports that only include `dir_val_###` columns (no engineered metrics) are now supported out of the box. Simply point the trainer at the data/label CSVs—no extra flags are required:

```bash
flybehavior-response train \
  --data-csv /path/to/dir_val_only_data.csv \
  --labels-csv /path/to/labels.csv \
  --model all
```

The loader detects that engineered features are missing, logs a trace-only message, and continues with PCA on the `dir_val_` traces. The same behaviour applies to `eval`, `viz`, and `predict`, so the entire pipeline operates normally on these legacy tables.

## Label weighting and troubleshooting

- Ensure trace columns follow contiguous 0-based numbering for each prefix (default `dir_val_`). Columns beyond `dir_val_3600` are trimmed automatically for legacy datasets.
- `user_score_odor` must contain non-negative integers where `0` denotes no response and higher integers (e.g., `1-5`) encode increasing reaction strength. Rows with missing labels are dropped automatically, while negative or fractional scores raise schema errors.
- Training uses proportional sample weights derived from label intensity so stronger reactions (e.g., `5`) contribute more than weaker ones (e.g., `1`). Review the logged weight summaries if model behaviour seems unexpected.
- Duplicate keys across CSVs (`fly`, `fly_number`, `trial_label`) raise errors to prevent ambiguous merges.
- Ratio features (`AUC-During-Before-Ratio`, `AUC-After-Before-Ratio`) are supported but produce warnings because they are unstable.
- Use `--dry-run` to confirm configuration before writing artifacts.
- The CLI automatically selects the newest run directory containing model artifacts. Override with `--run-dir` if you maintain
  multiple artifact trees (e.g., `artifacts/projections`).

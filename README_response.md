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
- `--model mlp` isolates the neural network if you want to iterate quickly without re-fitting the classical baselines.
- Existing scripts that still pass `--model both` continue to run LDA + logistic regression only; update them to `--model all` to include the MLP.
- Inspect `metrics.json` for `test` entries to verify held-out accuracy/F1 scores, and review `confusion_matrix_<model>.png` in the run directory for quick diagnostics.

## Label weighting and troubleshooting

- Ensure trace columns range from `dir_val_0` through `dir_val_3600`; any higher indices are removed automatically.
- `user_score_odor` must contain non-negative integers where `0` denotes no response and higher integers (e.g., `1-5`) encode increasing reaction strength. Rows with missing labels are dropped automatically, while negative or fractional scores raise schema errors.
- Training uses proportional sample weights derived from label intensity so stronger reactions (e.g., `5`) contribute more than weaker ones (e.g., `1`). Review the logged weight summaries if model behaviour seems unexpected.
- Duplicate keys across CSVs (`fly`, `fly_number`, `trial_label`) raise errors to prevent ambiguous merges.
- Ratio features (`AUC-During-Before-Ratio`, `AUC-After-Before-Ratio`) are supported but produce warnings because they are unstable.
- Use `--dry-run` to confirm configuration before writing artifacts.
- The CLI automatically selects the newest run directory containing model artifacts. Override with `--run-dir` if you maintain
  multiple artifact trees (e.g., `artifacts/projections`).

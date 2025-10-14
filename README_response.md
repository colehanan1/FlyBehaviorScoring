# flybehavior_response

This package trains, evaluates, and visualizes supervised models that predict fly odor responses from proboscis traces and engineered features.

## Installation

```bash
pip install -e .
```

## Command Line Interface

After installation, the `flybehavior-response` command becomes available. Common arguments:

- `--data-csv`: Wide proboscis trace CSV.
- `--labels-csv`: Labels CSV with `user_score_odor`.
- `--features`: Comma-separated engineered feature list (default: `AUC-During,TimeToPeak-During,Peak-Value`).
- `--include-auc-before`: Adds `AUC-Before` to the feature set.
- `--use-raw-pca` / `--no-use-raw-pca`: Toggle raw trace PCA (default enabled).
- `--n-pcs`: Number of PCA components (default 5).
- `--model`: `lda`, `logreg`, or `both` (default `both`).
- `--cv`: Stratified folds for cross-validation (default 0 for none).
- `--artifacts-dir`: Root directory for outputs (default `./artifacts`).
- `--plots-dir`: Plot directory (default `./artifacts/plots`).
- `--seed`: Random seed (default 42).
- `--dry-run`: Validate pipeline without saving artifacts.
- `--verbose`: Enable DEBUG logging.

### Subcommands

| Command | Purpose |
| --- | --- |
| `prepare` | Validate inputs, report class balance, write merged parquet. |
| `train` | Fit preprocessing + models, compute metrics, save joblib/config/metrics. |
| `eval` | Reload saved models and recompute metrics on merged data. |
| `viz` | Generate PC scatter, LDA score histogram, and ROC curve (if available). |
| `predict` | Score a merged CSV with a saved model and write predictions. |

### Examples

```bash
flybehavior-response prepare --data-csv all_envelope_rows_wide.csv \
  --labels-csv scoring_results_opto_new_BINARY.csv

flybehavior-response train --data-csv all_envelope_rows_wide.csv \
  --labels-csv scoring_results_opto_new_BINARY.csv --model both --n-pcs 5

flybehavior-response eval --data-csv all_envelope_rows_wide.csv \
  --labels-csv scoring_results_opto_new_BINARY.csv

flybehavior-response viz --data-csv all_envelope_rows_wide.csv \
  --labels-csv scoring_results_opto_new_BINARY.csv --plots-dir artifacts/plots

flybehavior-response predict --data-csv merged.csv --model-path artifacts/<run>/model_logreg.joblib \
  --output-csv artifacts/predictions.csv
```

## Troubleshooting

- Ensure trace columns range from `dir_val_0` through `dir_val_3600`; any higher indices are removed automatically.
- Labels must be binary (`0` or `1`) with no missing values; rows with missing labels are dropped during preparation.
- Duplicate keys across CSVs (`fly`, `fly_number`, `trial_label`) raise errors to prevent ambiguous merges.
- Ratio features (`AUC-During-Before-Ratio`, `AUC-After-Before-Ratio`) are supported but produce warnings because they are unstable.
- Use `--dry-run` to confirm configuration before writing artifacts.

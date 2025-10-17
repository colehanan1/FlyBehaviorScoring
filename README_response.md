# flybehavior_response

This package trains, evaluates, and visualizes supervised models that predict fly odor responses from proboscis traces and engineered features.

## Installation

```bash
pip install -e .
```

### Using this package from another repository

- **Pin it as a dependency.** In the consuming project (e.g. [`Ramanlab-Auto-Data-Analysis`](https://github.com/colehanan1/Ramanlab-Auto-Data-Analysis)), add the git URL to your dependency file so the environment always installs the latest revision of this project:

  ```text
  # requirements.txt inside Ramanlab-Auto-Data-Analysis
  flybehavior-response @ git+https://github.com/colehanan1/FlyBehaviorScoring.git
  ```

  Pip normalizes hyphens and underscores, so `flybehavior-response` is the canonical project name exported by `pyproject.toml`. Older guidance that used `flypca` or `flybehavior_response` will fail with a metadata mismatch error because the installer pulls a distribution named differently from the requested requirement. Update the dependency string as shown above.

  With `pip>=22`, this syntax works for `requirements.txt`, `pyproject.toml` (PEP 621 `dependencies`), and `setup.cfg`.

  To confirm the dependency resolves correctly, install from git in a clean environment and inspect the resulting metadata:

  ```bash
  python -m pip install "git+https://github.com/colehanan1/FlyBehaviorScoring.git#egg=flybehavior-response"
  python -m pip show flybehavior-response
  ```

  The `#egg=` fragment is optional for modern pip but keeps older tooling happy when parsing the distribution name from the URL.

- **Install together with the automation repo.** Once the dependency is listed, a regular `pip install -r requirements.txt` (or `pip install -e .` if the other repo itself is editable) pulls in this package exactly once—no manual reinstall inside each checkout is required.

- **Call the CLI from jobs or notebooks.** After installation, the `flybehavior-response` entry point is on `PATH`. Automation workflows can invoke it via shell scripts or Python:

  ```python
  import subprocess

  subprocess.run(
      [
          "flybehavior-response",
          "predict",
          "--data-csv",
          "/path/to/wide.csv",
          "--model-path",
          "/path/to/model_mlp.joblib",
          "--output-csv",
          "artifacts/predictions.csv",
      ],
      check=True,
  )
  ```

- **Import the building blocks directly.** When you need finer control than the CLI offers, import the core helpers:

  ```python
  from flybehavior_response.evaluate import load_pipeline

  pipeline = load_pipeline("/path/to/model_mlp.joblib")
  # df is a pandas DataFrame shaped like the merged training data
  predictions = pipeline.predict(df)
  ```

  The `flybehavior_response.io.load_and_merge` helper mirrors the CLI’s CSV merging logic so scheduled jobs can stay fully programmatic.

- **Match the NumPy major version with saved artifacts.** Models trained with NumPy 1.x store their random state differently from
  NumPy 2.x. Loading those joblib files inside an environment that already upgraded to NumPy 2.x raises:

  ```text
  ValueError: state is not a legacy MT19937 state
  ```

  Install `numpy<2.0` (already enforced by this package’s dependency pins) or rebuild the model artifact under the newer stack
  before invoking `flybehavior-response predict` inside automation repos.
  If you previously added a `sitecustomize.py` shim to coerce the MT19937 payload, remove it—the shim now runs even though NumPy
  is downgraded and corrupts the state with the following error:

  ```text
  TypeError: unhashable type: 'dict'
  ```

  Delete or update the shim so it gracefully handles dictionary payloads. With NumPy 1.x the extra hook is unnecessary, and the
  loader will succeed without further tweaks. If the shim keeps calling into NumPy, but returns a class object instead of the
  literal string `"MT19937"`, the loader fails with:

  ```text
  ValueError: <class 'numpy.random._mt19937.MT19937'> is not a known BitGenerator module.
  ```

  Update the shim so it returns `"MT19937"` when NumPy requests a bit generator by name, or guard the entire file behind a
  `numpy>=2` check. With NumPy 1.x the extra hook is unnecessary, and the loader will succeed without further tweaks. If other
  tools in the same environment still require the compatibility layer, replace the file with a guarded variant that short-circuits
  on NumPy < 2.0 and normalises dictionary payloads safely:

  ```python
  """Runtime compatibility shims for external tools invoked by the pipeline."""
  from __future__ import annotations

  import importlib
  from typing import Any

  import numpy as np


  def _normalise_mt19937_state(state: Any, target_name: str) -> Any:
      try:
          np_major = int(np.__version__.split(".")[0])
      except Exception:
          np_major = 0
      if np_major < 2:
          return state
      if isinstance(state, dict):
          payload = state.get("state") or state
          if isinstance(payload, dict) and {"key", "pos"}.issubset(payload):
              return {
                  "bit_generator": target_name,
                  "state": {
                      "key": np.asarray(payload["key"], dtype=np.uint32),
                      "pos": int(payload["pos"]),
                  },
              }
      return state


  def _install_numpy_joblib_shims() -> None:
      try:
          np_pickle = importlib.import_module("numpy.random._pickle")
      except ModuleNotFoundError:
          return
      original_ctor = getattr(np_pickle, "__bit_generator_ctor", None)
      if original_ctor is None:
          return

      class _CompatMT19937(np.random.MT19937):
          def __setstate__(self, state: Any) -> None:  # type: ignore[override]
              super().__setstate__(_normalise_mt19937_state(state, type(self).__name__))

      mapping = getattr(np_pickle, "BitGenerators", None)
      if isinstance(mapping, dict):
          mapping["MT19937"] = _CompatMT19937

      def _compat_ctor(bit_generator: Any = "MT19937") -> Any:
          return original_ctor("MT19937")

      np_pickle.__bit_generator_ctor = _compat_ctor


  _install_numpy_joblib_shims()
  ```

  This template preserves the original behaviour when NumPy 2.x is present, yet becomes a no-op under NumPy 1.x so your pipeline
  no longer crashes when loading FlyBehaviorScoring artifacts.

## Building and publishing the package

Follow these steps when you need a distributable artifact instead of an editable install or git reference:

1. Create a clean environment and install the build backend once:
   ```bash
   python -m pip install --upgrade pip build twine
   ```
2. Produce both wheel and source distributions:
   ```bash
   python -m build
   ```
   The artifacts land under `dist/` (for example, `dist/flybehavior-response-0.1.0-py3-none-any.whl`).
3. Upload to an index (test or production) with Twine:
   ```bash
   twine upload dist/*
   ```
   Replace the repository URL or credentials as needed (`--repository testpypi`).

Once published, downstream projects can depend on the released version instead of a git SHA:
```text
flybehavior-response==0.1.0
```

If you only need automation machines to consume the latest commit, prefer the git dependency shown earlier—publishing is optional.

### Publishing straight from Git

You do **not** have to cut a wheel to exercise the package from a private repo. Git-based installs work as long as the repository exposes a valid `pyproject.toml` (which this project does). Pick the option that matches your workflow:

1. **Pin the main branch head** for fast iteration:
   ```text
   flybehavior-response @ git+https://github.com/colehanan1/FlyBehaviorScoring.git
   ```

2. **Lock to a tag or commit** for reproducible automation:
   ```text
   flybehavior-response @ git+https://github.com/colehanan1/FlyBehaviorScoring.git@v0.1.0
   # or
   flybehavior-response @ git+https://github.com/colehanan1/FlyBehaviorScoring.git@<commit-sha>
   ```

3. **Reference a subdirectory** if you reorganize the repo later (pip needs the leading `src/` layout path):
   ```text
   flybehavior-response @ git+https://github.com/colehanan1/FlyBehaviorScoring.git#subdirectory=.
   ```

   The `src/` layout is already wired into `pyproject.toml`, so no extra flags are necessary today. Keep the `#subdirectory` fragment in mind if you move the project under a monorepo path.

Regardless of which selector you use, `pip show flybehavior-response` should list the install location under the environment’s site-packages directory. If it does not, check that your requirements file matches the casing and punctuation above and that you do not have an older `flypca` editable install overshadowing it on `sys.path`.


## Command Line Interface

After installation, the `flybehavior-response` command becomes available. Common arguments:

- `--data-csv`: Wide proboscis trace CSV.
- `--labels-csv`: Labels CSV with `user_score_odor` scores (0 = no response, 1-5 = increasing response strength).
- `--features`: Comma-separated engineered feature list (default: `AUC-During,TimeToPeak-During,Peak-Value`).
- `--include-auc-before`: Adds `AUC-Before` to the feature set.
- `--use-raw-pca` / `--no-use-raw-pca`: Toggle raw trace PCA (default enabled).
- `--n-pcs`: Number of PCA components (default 5).
- `--model`: `lda`, `logreg`, `mlp`, `mlp_adam`, `both`, or `all` (default `all`).
- `--logreg-solver`: Logistic regression solver (`lbfgs`, `liblinear`, `saga`; default `lbfgs`).
- `--logreg-max-iter`: Iteration cap for logistic regression (default `1000`; increase if convergence warnings appear).
- `--cv`: Stratified folds for cross-validation (default 0 for none).
- `--artifacts-dir`: Root directory for outputs (default `./artifacts`).
- `--plots-dir`: Plot directory (default `./artifacts/plots`).
- `--seed`: Random seed (default 42).
- `--dry-run`: Validate pipeline without saving artifacts.
- `--verbose`: Enable DEBUG logging.
- `--fly`, `--fly-number`, `--trial-label`/`--testing-trial` (predict only): Filter predictions to a single trial.

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

# score a specific fly/trial tuple in the original envelope export
flybehavior-response predict --data-csv /home/ramanlab/Documents/cole/Data/Opto/all_envelope_rows_wide.csv \
  --model-path artifacts/<run>/model_logreg.joblib --fly september_09_fly_3 --fly-number 3 --trial-label t2 \
  --output-csv artifacts/predictions_envelope_t2.csv
```

## Training with the neural network classifiers

- `--model all` trains LDA, logistic regression, and the new MLP classifier using a shared stratified 80/20 split and writes per-model confusion matrices into the run directory.
- Each training run now exports `predictions_<model>_{train,test}.csv` so you can audit which trials were classified correctly, along with their reaction probabilities and sample weights.
- `--model mlp` isolates the legacy single-layer network if you want to iterate quickly without re-fitting the classical baselines.
- The `mlp` option instantiates scikit-learn's `MLPClassifier` with a very wide hidden layer (`hidden_layer_sizes=20000`) to mirror the historical configuration shipped with the lab notebooks.
- `--model mlp_adam` enables a deeper Adam-optimized network (`hidden_layer_sizes=(512, 256, 64)`, ReLU activations, higher iteration budget) so you can scale capacity gradually rather than jumping straight to the legacy 20k-unit slab.
- Neither neural-network variant natively honors `sample_weight`. The training loop now expands the stratified training split proportionally to the provided weights before calling `.fit`, so every neural model still experiences the intended label-intensity emphasis.
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

The raw workflow is always two-step: generate a per-trial table with `prepare-raw`, then invoke `train`, `eval`, `viz`, and `predict` with `--raw-series` (or explicit `--series-prefixes`) so every command consumes the four eye/proboscis streams exactly as prepared.
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

### Scoring individual trials

- Use the new `predict` filters when you want to score a single envelope or raw trial without extracting it manually:

  ```bash
  flybehavior-response predict \
    --data-csv /home/ramanlab/Documents/cole/Data/Opto/all_envelope_rows_wide.csv \
    --model-path artifacts/<run>/model_logreg.joblib \
    --fly september_09_fly_3 --fly-number 3 --testing-trial t2 \
    --output-csv artifacts/<run>/prediction_september_09_fly_3_t2.csv
  ```

- The loader automatically treats a `testing_trial` column as the canonical `trial_label`, so legacy exports continue to work. Supply any subset of the filters (`--fly`, `--fly-number`, `--trial-label`/`--testing-trial`) to narrow the prediction set; when all three are present, exactly one trial is returned and written with its reaction probability.

## Label weighting and troubleshooting

- Ensure trace columns follow contiguous 0-based numbering for each prefix (default `dir_val_`). Columns beyond `dir_val_3600` are trimmed automatically for legacy datasets.
- `user_score_odor` must contain non-negative integers where `0` denotes no response and higher integers (e.g., `1-5`) encode increasing reaction strength. Rows with missing labels are dropped automatically, while negative or fractional scores raise schema errors.
- Training uses proportional sample weights derived from label intensity so stronger reactions (e.g., `5`) contribute more than weaker ones (e.g., `1`). Review the logged weight summaries if model behaviour seems unexpected.
- Duplicate keys across CSVs (`fly`, `fly_number`, `trial_label`) raise errors to prevent ambiguous merges.
- Ratio features (`AUC-During-Before-Ratio`, `AUC-After-Before-Ratio`) are supported but produce warnings because they are unstable.
- Use `--dry-run` to confirm configuration before writing artifacts.
- The CLI automatically selects the newest run directory containing model artifacts. Override with `--run-dir` if you maintain
  multiple artifact trees (e.g., `artifacts/projections`).

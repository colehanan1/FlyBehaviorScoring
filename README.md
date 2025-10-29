# flybehavior_response

This package trains, evaluates, and visualizes supervised models that predict fly odor responses from proboscis traces and engineered features.

## Installation

```bash
pip install -e .
```

## Optuna MLP Hyperparameter Tuning

The repository now ships with `optuna_mlp_tuning.py`, a production-grade search
script that co-optimises PCA dimensionality and the
`SampleWeightedMLPClassifier`. The workflow enforces fly-level leakage guards
via `GroupKFold`, up-weights high-intensity class-5 responders, and evaluates
macro-F1 as the primary score.

### Running the tuner

```bash
python optuna_mlp_tuning.py \
  --data-csv /path/to/all_envelope_rows_wide.csv \
  --labels-csv /path/to/labels.csv \
  --n-trials 100 \
  --timeout 7200 \
  --output-dir optuna_results
```

* Provide `--labels-csv` when using the canonical wide export split across data
  and label tables. If labels are embedded in the features CSV, ensure a
  `reaction_strength` column is present and omit `--labels-csv`.
* Add `--features "AUC-During,global_max,..."` to constrain the search to a
  curated subset of engineered scalars. The tuner validates every requested
  column, removes duplicates, and records the final selection alongside the
  saved hyperparameters.
* Whenever a feature subset leaves fewer usable columns than the requested PCA
  dimensionality, the tuner, deterministic baseline, best-parameter replay, and
  final retraining all clamp `n_components` to the available feature count while
  enforcing the minimum viable dimension. This guard stops PCA from erroring out
  on engineered-only configurations with very few predictors.
* Each trial prunes early when the interim macro-F1 under-performs the running
  Optuna median after two folds, keeping runtime within the two-hour budget.
* Sample weights default to 1.0 for non-responders and lower-intensity
  responses, with class-5 trials receiving a 5× multiplier during optimisation.
* The search space spans continuous integer ranges: PCA components anywhere from
  3 to 64, mini-batch sizes between 8 and 64, and hidden-layer widths from 96 up
  to 750 neurons per layer. Saved JSON payloads may therefore contain any
  integer within those bounds, and both the tuner and CLI will honour them.
* Provide `--best-params-json /path/to/best_params.json` to skip optimisation and
  retrain/evaluate using a previously exported Optuna configuration. The JSON
  may contain either the raw Optuna trial parameters (`architecture`, `h1`,
  `layer_config`, etc.) or the normalised output written by this script.

### Generated artefacts

The command writes all deliverables into `--output-dir` (defaults to
`optuna_results/`):

| File | Description |
| --- | --- |
| `optuna_study.db` | SQLite storage for resuming or auditing the study. |
| `optuna_trials.csv` | Tabular export of every trial with metrics and timings. |
| `optuna_history.html` | Interactive optimisation trace (Plotly). |
| `optuna_importances.html` | Hyperparameter importance plot emphasising PCA components. |
| `best_params.json` | Best configuration including architecture, optimiser settings, and the selected engineered features. |
| `best_mlp_model.joblib` | Retrained preprocessing + MLP pipeline for deployment. |
| `TUNING_REPORT.md` | Auto-generated summary comparing the tuned model with the baseline. |

The retrained pipeline includes the median imputer, scaler, PCA transform, and
the optimised neural network, allowing drop-in inference via
`joblib.load(output_dir / "best_mlp_model.joblib")`. Because `.joblib` files are
ignored by Git, they remain local run artefacts.

### Reusing a saved configuration

After a successful Optuna run, you can rebuild and retrain the best pipeline
without repeating the search:

```bash
python optuna_mlp_tuning.py \
  --data-csv /path/to/all_envelope_rows_wide.csv \
  --labels-csv /path/to/labels.csv \
  --best-params-json optuna_results/best_params.json \
  --output-dir optuna_results
```

The script normalises the JSON payload, resolves the hidden-layer topology, and
trains the `SampleWeightedMLPClassifier` end to end with the saved hyperparameters.
All downstream artefacts (model, report, and parameter snapshot) are refreshed
to reflect the supplied configuration. When a feature subset was enforced, the
`selected_features` array persisted in `best_params.json` is honoured so the
re-evaluation mirrors the original search space. Any PCA dimensionality in the
payload that exceeds the reduced feature set is automatically clamped to keep
the reconstruction numerically valid.

### Training the CLI with tuned hyperparameters

Once `best_params.json` is available, the primary CLI can consume it directly so
you can train every supported model—MLP included—without rerunning Optuna:

```bash
flybehavior-response train \
  --data-csv /path/to/all_envelope_rows_wide.csv \
  --labels-csv /path/to/labels.csv \
  --model all \
  --best-params-json optuna_results/best_params.json \
  --artifacts-dir artifacts
```

Providing `--best-params-json` automatically enables PCA on the raw traces,
overrides `--n-pcs` with the tuned `n_components`, and instantiates the
`SampleWeightedMLPClassifier` with the Optuna-selected architecture, learning
rate, regularisation, and batch size. The generated `config.json` embedded in
each run directory now records the consolidated Optuna payload so downstream
evaluation jobs can trace exactly which hyperparameters were used. When the
payload enumerates a `selected_features` subset, the training command enforces
that exact list—even if a different `--features` string is supplied—so
retraining stays faithful to the search space. Missing columns now trigger a
hard failure with the full list of available engineered features to help you fix
typos before any models are saved.

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

- **Stream raw geometry safely.** Large frame-level exports no longer require
  loading the entire CSV into memory. Use the updated ``prepare`` subcommand to
  stream in chunks, validate block continuity, and optionally persist a
  compressed parquet cache for subsequent runs:

  ```bash
  flybehavior-response \
      prepare \
      --data-csv stats/geometry_frames.csv \
      --labels-csv stats/labels.csv \
      --geom-columns "dataset,fly,fly_number,trial_type,trial_label,frame_idx,x,y" \
      --geom-chunk-size 20000 \
      --cache-parquet artifacts/geom_cache.parquet \
      --aggregate-geometry \
      --aggregate-stats mean,max \
      --aggregate-format parquet \
      --artifacts-dir artifacts
  ```

  The stream honours the original column order, emits per-chunk diagnostics, and
  enforces uniqueness of ``dataset``/``fly``/``fly_number``/``trial_type``/
  ``trial_label`` keys across the
  optional labels CSV. Aggregation is optional; when enabled it produces a
  per-trial summary file alongside the cache. Choose between a compressed
  parquet (default, requires ``pyarrow`` or ``fastparquet``) and a portable CSV
  by passing ``--aggregate-format parquet`` or ``--aggregate-format csv``
  respectively.
  The same pipeline is available programmatically via
  ``flybehavior_response.io.load_geom_frames`` and
  ``flybehavior_response.io.aggregate_trials`` for notebook workflows.

  Geometry exports that expose a different frame counter (for example a column
  named ``frame`` instead of the default ``frame_idx``) are resolved
  automatically. The loader now detects the alternate header, validates
  contiguity against that column, and keeps the block integrity checks active
  without any additional flags.

  Only trials present in the labels CSV are streamed. Rows without labels are
  dropped up front so aggregation and caching operate on fully annotated data.
  To debug unexpected omissions, rerun ``prepare`` with ``--keep-missing-labels``
  to surface a validation error listing the offending keys.

- **Train directly from geometry frames.** Provide ``--geometry-frames`` to the
  ``train``, ``eval``, and ``predict`` subcommands to stream per-frame CSVs or
  parquet exports on the fly. Combine ``--geom-granularity`` with the default
  ``trial`` mode to materialise aggregated per-trial features or switch to
  ``frame`` when frame-level rows are preferred. Aggregation honours the same
  ``--geom-stats`` options exposed by ``prepare``, while ``--geom-normalize``
  applies either ``zscore`` or ``minmax`` scaling before safely downcasting the
  values to ``float32`` so they align with the existing feature engineering
  pipeline. The ``train`` command now writes a ``split_manifest.csv`` alongside
  the trained models describing the fly-level ``GroupShuffleSplit`` assignment;
  pass ``--group-override none`` to disable leakage guards when cross-fly
  isolation is not required.
  When the geometry stream includes raw coordinate columns named
  ``eye_x``, ``eye_y``, ``prob_x``, and ``prob_y``, the loader assembles these
  into ``eye_x_f*``/``eye_y_f*``/``prob_x_f*``/``prob_y_f*`` trace series for
  every trial. These traces mirror the format produced by the legacy
  ``prepare_raw`` workflow, unlock PCA on raw motion signals without additional
  preprocessing, and remain aligned with the per-trial aggregation and leakage
  guards described above.

#### Restrict geometry features for modeling

If you prefer to train on a curated feature panel instead of the entire
aggregate table, pass ``--geom-feature-columns`` to ``train``, ``eval``, or
``predict``. Supply a comma-separated list directly or reference a
newline-delimited file by prefixing the path with ``@``:

```bash
flybehavior-response train \
  --geometry-frames /path/to/geom_frames.csv \
  --geometry-trials /path/to/geom_trial_summary.csv \
  --labels-csv /path/to/labels.csv \
  --geom-feature-columns @experiments/feature_subset.txt \
  --model mlp
```

The loader validates the selection (for example ``r_before_mean`` or
``metric_mean``) and raises a schema error when any requested column is absent
so mistakes surface immediately. The resolved subset is also written to
``config.json`` under ``geometry_feature_columns`` to keep the training
provenance auditable.

#### Merge precomputed per-trial geometry summaries

When a laboratory already maintains per-fly or per-trial statistics in a CSV, you can hand those features to the streaming loader with the new ``--geometry-trials`` flag. The file must contain one row per trial with the canonical identifier columns (``dataset``, ``fly``, ``fly_number``, ``trial_type``, ``trial_label``) plus the following engineered metrics so downstream models receive a consistent schema:

``W_est_fly``, ``H_est_fly``, ``diag_est_fly``, ``r_min_fly``, ``r_max_fly``, ``r_p01_fly``, ``r_p99_fly``, ``r_mean_fly``, ``r_std_fly``, ``n_frames``, ``r_mean_trial``, ``r_std_trial``, ``r_max_trial``, ``r95_trial``, ``dx_mean_abs``, ``dy_mean_abs``, ``r_pct_robust_fly_max``, ``r_pct_robust_fly_mean``, ``r_before_mean``, ``r_before_std``, ``r_during_mean``, ``r_during_std``, ``r_during_minus_before_mean``, ``cos_theta_during_mean``, ``sin_theta_during_mean``, ``direction_consistency``, ``frac_high_ext_during``, ``rise_speed``.

During ``load_geometry_dataset`` the summaries are merged with the streamed aggregates before normalisation and downcasting, so every new numeric column participates in the same scaling pipeline. If a column is present in both the streamed aggregates and the external summary, the loader keeps the streamed value and warns about mismatches so accidental drift is visible. ``--geometry-trials`` is only valid when ``--geometry-frames`` is provided at trial granularity.

Example training command:

```bash
flybehavior-response train \
  --geometry-frames /path/to/geom_frames.csv \
  --geometry-trials /path/to/geom_trial_summary.csv \
  --labels-csv /path/to/labels.csv \
  --model logreg
```

The run configuration now records the trial-summary path alongside the geometry frames so provenance remains auditable.
Provide a labels CSV containing the canonical ``user_score_odor`` column; values
greater than zero are automatically coerced to a binary responder target during
``train``/``eval``/``predict`` so no manual preprocessing step is required.
Rows missing labels are dropped from the geometry stream by default to keep the
aggregation consistent—rerun with ``--keep-missing-labels`` if you want to audit
which trials were skipped.

### Enriched per-frame geometry columns and responder training workflow

The geometry enrichment step now emits additional, behaviourally grounded
columns so downstream analyses no longer have to reconstruct stimulus epochs or
basic response summaries manually. Each frame row in the enriched CSV includes:

| Column | Definition | Why it matters |
| --- | --- | --- |
| ``is_before`` | Binary mask marking frames in the baseline (pre-odor) window. | Lets downstream code isolate baseline behaviour without re-deriving stimulus timing, which keeps reproducibility intact across labs. |
| ``is_during`` | Binary mask marking frames during odor stimulation. | Ensures frame-level filters target the causal window that determines the responder label. |
| ``is_after`` | Binary mask marking frames in the post-odor window. | Allows post-hoc inspection without contaminating training features that should focus on the odor epoch. |
| ``r_before_mean`` | Mean proboscis extension (percentage of the fly’s robust range) computed across baseline frames. | Captures resting proboscis position; elevated values indicate partial extension even before odor onset. |
| ``r_before_std`` | Standard deviation of proboscis extension during baseline. | Measures baseline “fidgeting.” High variance reveals spontaneous motion that can masquerade as responses. |
| ``r_during_mean`` | Mean extension percentage while odor is on. | Quantifies the sustained response amplitude during stimulation. |
| ``r_during_std`` | Standard deviation of extension during odor. | Summarises modulation depth; large swings reflect oscillatory probing, while small values indicate a rigid hold. |
| ``r_during_minus_before_mean`` | ``r_during_mean - r_before_mean``. | Expresses the odor-triggered change in the fly’s own units. Positive values are odor-locked proboscis extensions; zero or negative values show absence or suppression. |
| ``cos_theta_during_mean`` | Mean cosine of the proboscis direction vector (normalised ``dx``/``dy``) during odor. | Encodes whether the proboscis points forward, downward, or laterally—key for separating feeding-like probes from grooming. |
| ``sin_theta_during_mean`` | Mean sine of the proboscis direction vector during odor. | Complements ``cos_theta_during_mean`` so the full direction is available in head-centred coordinates. |
| ``direction_consistency`` | Length of the mean direction vector, computed as ``sqrt(cos_theta_during_mean**2 + sin_theta_during_mean**2)``. | Scores directional stability: values near 1.0 mean deliberate probes, while lower values flag chaotic motion unrelated to odor. |
| ``frac_high_ext_during`` | Fraction of odor-period frames where ``r_pct_robust_fly`` exceeds 75 % of that fly’s robust range. Range: [0, 1]. | Captures how long the proboscis stayed highly extended; separates quick flicks from sustained acceptance-like behaviour. |
| ``rise_speed`` | Initial slope of extension at odor onset: ``(mean extension in the first second of odor − r_before_mean) / 1 s`` expressed as percentage per second. | Measures how quickly the response ramps. Fast rises are characteristic of true stimulus-driven reactions. |

The geometry loader populates these columns automatically whenever the labels table supplies ``odor_on_idx`` and ``odor_off_idx`` values alongside the raw proboscis coordinates. The enrichment runs during ``load_geom_frames`` and all CLI entry points that consume geometry inputs, so downstream scripts and notebooks receive consistent epoch flags and responder summaries without additional preprocessing.
If the odor timing columns are absent, the loader still succeeds and emits the
responder summary columns, but their values fall back to ``NaN`` and the
``rise_speed`` metric remains undefined for those trials. Supplying the odor
indices is therefore strongly recommended whenever the experiment design makes
them available.

#### Build per-trial feature tables for supervised learning

These columns make the per-frame CSV directly usable for training binary
classifiers that decide whether a fly responded to the odor in a given trial.
Follow this procedure when preparing data for a multilayer perceptron (MLP) or
another lightweight model:

1. Obtain the human-annotated (or rule-derived) trial labels where
   ``Responder = 1`` denotes a clear odor response and ``Responder = 0`` denotes
   no response.
2. For each trial, collapse the per-frame enrichment into a single feature
   vector by extracting exactly these ten scalar summaries:
   ``[r_before_mean, r_before_std, r_during_mean, r_during_std,
   r_during_minus_before_mean, cos_theta_during_mean, sin_theta_during_mean,
   direction_consistency, frac_high_ext_during, rise_speed]``.
3. Assemble a training table with one row per trial and join the responder
   labels as the target column.
4. Train the MLP (or another binary classifier) on this 10-dimensional input to
   predict the responder label.

This feature set is intentionally compact, biologically interpretable, and
normalised per fly (all extension metrics operate on ``r_pct_robust_fly`` which
uses the fly’s own ``r_p01_fly``/``r_p99_fly`` range). It avoids dependence on
camera geometry or trial identifiers, and it limits the inputs to pre-odor and
during-odor information so the model answers the causal question: did the odor
move the proboscis away from baseline?

Avoid feeding raw per-frame series, file or fly identifiers, camera scaling
fields (``W_est_fly``, ``H_est_fly``), or any post-odor aggregates into the
first-round classifier. Those inputs inject nuisance variation, leak
non-causal structure, and encourage overfitting on small datasets.

Remember that the enriched CSV is an intermediate artefact designed for reuse
across pipelines. Build the actual training matrix by selecting one row per
trial and projecting down to the summary columns listed above before invoking
``flybehavior-response train`` or a custom scikit-learn script.

- **Regenerate the geometry cache without touching disk** by using ``--dry-run``
  together with ``--cache-parquet``; the CLI will validate inputs and report
  chunk-level statistics without writing artifacts. If the optional parquet
  engines are unavailable, switch to ``--aggregate-format csv`` for downstream
  smoke tests.

- **Validate the new pipeline locally.** Run the focused pytest targets to
  confirm schema handling, cache behaviour, and aggregation parity:

  ```bash
  PYTHONPATH=src pytest src/flybehavior_response/tests/test_response_io.py -k geom
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
  Every entry must match a column in the merged dataset. The trainer now aborts when a requested feature is missing instead of
  silently reverting to the full feature set, keeping curated subsets intact.
- `--raw-series`: Prioritize the default raw coordinate prefixes (eye/proboscis channels).
- `--no-raw`: Drop all trace columns so only engineered features feed the models.
- `--include-auc-before`: Adds `AUC-Before` to the feature set.
- `--use-raw-pca` / `--no-use-raw-pca`: Toggle raw trace PCA (default enabled).
- `--n-pcs`: Number of PCA components (default 5).
- `--model`: `lda`, `logreg`, `mlp`, `fp_optimized_mlp`, `both`, or `all` (default `all`).
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
  --labels-csv /home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new_MINIMAL.csv --model all --n-pcs 2

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

- `--model all` trains LDA, logistic regression, and both neural network configurations using a shared stratified split and writes per-model confusion matrices into the run directory.
- Each training run exports `predictions_<model>_{train,test}.csv` (and `validation` when applicable) so you can audit which trials were classified correctly, along with their reaction probabilities and sample weights.
- `--model mlp` isolates the legacy neural baseline: a scikit-learn `MLPClassifier` with a single hidden layer of 100 neurons between the feature input and the binary output unit.
- `--model fp_optimized_mlp` activates the new false-positive minimising architecture. It stacks two ReLU-activated hidden layers sized 256 and 128, uses Adam with a 0.001 learning rate, honours proportional intensity weights, and multiplies responder samples (`label==1`) by an additional class weight of 2.0. Training automatically performs a stratified 70/15/15 train/validation/test split, monitors validation performance with early stopping (`n_iter_no_change=10`), and logs precision plus false-positive rates across all splits.
- Inspect `metrics.json` for `test` (and `validation`) entries to verify held-out accuracy, precision, recall, F1, and false-positive rates. Review `confusion_matrix_<model>.png` in the run directory for quick diagnostics.
- Existing scripts that still pass `--model both` continue to run LDA + logistic regression only; update them to `--model all` to include the neural networks when desired.

Example run focused on minimising false positives:

```bash
flybehavior-response train \
  --data-csv stats/wide_features.csv \
  --labels-csv stats/labels.csv \
  --features "AUC-During,Peak-Value,global_max,local_min,local_max" \
  --series-prefixes "dir_val" \
  --model fp_optimized_mlp \
  --n-pcs 5 \
  --cv 5 \
  --artifacts-dir artifacts/fp_optimized
```

The run directory records the combined sample/class weights, validation metrics, and a confusion matrix that highlights the reduced false-positive rate.

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

Need to benchmark engineered features without the high-dimensional traces? Add `--no-raw` to the same subcommands. The loader drops every `dir_val_###`, `eye_x_f*`, `eye_y_f*`, `prob_x_f*`, and `prob_y_f*` column before training, stores that decision in `config.json`, and automatically disables PCA on the now-missing traces. Downstream `eval`, `viz`, and `predict` runs inherit the configuration, so omitting `--no-raw` later still reproduces the engineered-only workflow unless you explicitly override the series selection. The same flag works when you stream geometry frames with `--geometry-frames`: the trial aggregator now skips raw trace assembly so you can train purely on engineered responder features while keeping the persisted configuration in sync with `eval`, `viz`, and `predict` commands.

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
- Duplicate keys across CSVs (`dataset`, `fly`, `fly_number`, `trial_type`, `trial_label`) raise errors to prevent ambiguous merges.
- Ratio features (`AUC-During-Before-Ratio`, `AUC-After-Before-Ratio`) are supported but produce warnings because they are unstable.
- The CLI recognises the following engineered scalar columns out of the box: `AUC-Before`, `AUC-During`, `AUC-After`, `AUC-During-Before-Ratio`, `AUC-After-Before-Ratio`, `TimeToPeak-During`, `Peak-Value`, `global_min`, `global_max`, `local_min`, `local_max`, `local_min_during`, `local_max_during`, `local_max_over_global_min`, `local_max_during_over_global_min`, `local_max_during_odor`, and `local_max_during_odor_over_global_min`. Any subset passed via `--features` (or baked into `best_params.json`) is validated against this list so feature-only runs fail fast when a requested column is absent.
- Use `--dry-run` to confirm configuration before writing artifacts.
- The CLI automatically selects the newest run directory containing model artifacts. Override with `--run-dir` if you maintain
  multiple artifact trees (e.g., `artifacts/projections`).

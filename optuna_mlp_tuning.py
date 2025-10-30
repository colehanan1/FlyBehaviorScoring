"""Optuna-driven hyperparameter search for the sample-weighted MLP pipeline.

This module orchestrates an end-to-end optimisation workflow that jointly tunes
PCA dimensionality and multi-layer perceptron architecture while preserving
fly-level group splits. The script is designed for command-line execution and
integrates tightly with the existing :mod:`flybehavior_response` package.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CLASS_WEIGHT_FP_OPTIMIZED: Mapping[int, float] = {0: 1.0, 1: 2.0}

from flybehavior_response.io import (  # noqa: E402
    LABEL_COLUMN,
    LABEL_INTENSITY_COLUMN,
    MERGE_KEYS,
    NON_REACTIVE_FLAG_COLUMN,
    load_dataset,
)
from flybehavior_response.logging_utils import get_logger  # noqa: E402
from flybehavior_response.sample_weighted_mlp import (  # noqa: E402
    SampleWeightedMLPClassifier,
)
from flybehavior_response.modeling import (  # noqa: E402
    ALLOWED_BATCH_SIZES,
    ALLOWED_LAYER_WIDTHS,
    ARCHITECTURE_SINGLE,
    ARCHITECTURE_TWO_LAYER,
    MODEL_FP_OPTIMIZED_MLP,
    MODEL_MLP,
    normalise_mlp_params,
    resolve_hidden_layer_sizes_from_params,
)


@dataclass(slots=True)
class DatasetBundle:
    """Container for training data and associated metadata."""

    features: pd.DataFrame
    labels: pd.Series
    groups: pd.Series
    sample_weights: pd.Series
    label_intensity: pd.Series


@dataclass(slots=True)
class EvaluationResult:
    """Cross-validation metrics captured during model evaluation."""

    fold_macro_f1: List[float]
    fold_balanced_accuracy: List[float]
    fold_positive_f1: List[float]
    fold_negative_f1: List[float]
    fit_durations: List[float]

    @property
    def mean_macro_f1(self) -> float:
        return float(np.mean(self.fold_macro_f1))

    @property
    def std_macro_f1(self) -> float:
        return float(np.std(self.fold_macro_f1, ddof=1)) if len(self.fold_macro_f1) > 1 else 0.0

    @property
    def mean_balanced_accuracy(self) -> float:
        return float(np.mean(self.fold_balanced_accuracy))

    @property
    def std_balanced_accuracy(self) -> float:
        return (
            float(np.std(self.fold_balanced_accuracy, ddof=1))
            if len(self.fold_balanced_accuracy) > 1
            else 0.0
        )

    @property
    def mean_positive_f1(self) -> float:
        return float(np.mean(self.fold_positive_f1))

    @property
    def mean_negative_f1(self) -> float:
        return float(np.mean(self.fold_negative_f1))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Optuna hyperparameter optimisation for the sample-weighted MLP "
            "classifier on fly proboscis response data."
        )
    )
    parser.add_argument(
        "--data-csv",
        type=Path,
        required=True,
        help="Path to the trial-level feature matrix CSV exported from preprocessing.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to the labels CSV. When omitted, the data CSV must "
            "already include the user_score_odor label columns."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("optuna_results"),
        help="Directory where optimisation artefacts will be stored.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="mlp_tuning",
        help="Optuna study name for resuming runs.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Maximum number of Optuna trials to evaluate.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Optional global timeout (seconds). Use 0 to disable the limit.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help=(
            "Comma-separated list of engineered feature columns to retain. When omitted, "
            "all available scalar features are used."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_MLP,
        choices=[MODEL_MLP, MODEL_FP_OPTIMIZED_MLP],
        help=(
            "MLP variant to tune. Use 'fp_optimized_mlp' to apply the responder-focused "
            "class weighting during optimisation."
        ),
    )
    parser.add_argument(
        "--best-params-json",
        type=Path,
        default=None,
        help=(
            "Path to a JSON file containing a previously discovered Optuna parameter set. "
            "When provided, the study is skipped and the supplied configuration is evaluated "
            "and retrained."
        ),
    )
    parser.add_argument(
        "--baseline-hidden",
        type=int,
        default=128,
        help=(
            "Hidden width for the baseline comparison. For fp_optimized_mlp the value "
            "sets the first layer and a smaller second layer is derived automatically."
        ),
    )
    parser.add_argument(
        "--baseline-components",
        type=int,
        default=40,
        help="Number of PCA components used in the baseline model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> logging.Logger:
    """Initialise the project logger with the requested verbosity."""

    logger = get_logger("optuna_mlp_tuning", verbose=verbose)
    logger.propagate = False
    return logger


def _normalise_feature_list(raw: Optional[str]) -> Optional[Tuple[str, ...]]:
    """Parse a comma-separated feature string into a normalised tuple."""

    if raw is None:
        return None

    tokens = [token.strip() for token in raw.split(",")]
    filtered = [token for token in tokens if token]

    if not filtered:
        raise ValueError("--features was provided but no valid feature names were detected.")

    # Preserve order while removing duplicates
    seen = {}
    for token in filtered:
        seen.setdefault(token, None)
    return tuple(seen.keys())


def _generate_component_candidates(feature_dim: int) -> Tuple[int, ...]:
    """Return the permissible PCA component counts for the given feature space."""

    if feature_dim <= 0:
        raise ValueError("Feature dimensionality must be positive to derive PCA candidates.")

    upper = min(64, feature_dim)
    if upper < 3:
        return tuple(range(1, upper + 1))
    return tuple(range(3, upper + 1))


def _architecture_candidates_for_model(model_type: str) -> Tuple[str, ...]:
    """Return the permissible architecture tokens for the requested model variant."""

    if model_type == MODEL_FP_OPTIMIZED_MLP:
        return (ARCHITECTURE_TWO_LAYER,)
    return (ARCHITECTURE_SINGLE, ARCHITECTURE_TWO_LAYER)


def _prepare_dataframe_from_dataset(
    bundle: DatasetBundle,
    selected_features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return the feature matrix stripped of identifier and label columns."""

    drop_columns = [*MERGE_KEYS, LABEL_COLUMN, LABEL_INTENSITY_COLUMN, "reaction_strength"]
    available_to_drop = [col for col in drop_columns if col in bundle.features.columns]
    feature_frame = bundle.features.drop(columns=available_to_drop, errors="ignore")

    if selected_features is not None:
        missing = [name for name in selected_features if name not in feature_frame.columns]
        if missing:
            raise ValueError(
                "Requested engineered features are missing from the dataset: "
                f"{missing}"
            )
        feature_frame = feature_frame.loc[:, list(dict.fromkeys(selected_features))]
    return feature_frame


def load_training_data(
    *,
    data_csv: Path,
    labels_csv: Path | None,
    logger: logging.Logger,
) -> DatasetBundle:
    """Load and harmonise the training dataset.

    When ``labels_csv`` is supplied, the helper relies on
    :func:`flybehavior_response.io.load_dataset` for robust validation. If the
    dataset already contains labels (for example, ``reaction_strength``), the
    function constructs the expected binary labels and intensity columns in-place.
    """

    if labels_csv is not None:
        dataset = load_dataset(
            data_csv=data_csv,
            labels_csv=labels_csv,
            logger_name="optuna_mlp_tuning",
            include_traces=True,
        )
        frame = dataset.frame.copy()
        labels = frame[LABEL_COLUMN].astype(int)
        intensity = frame[LABEL_INTENSITY_COLUMN].astype(int)
    else:
        logger.info(
            "Loading combined feature+label CSV without a separate labels file: %s",
            data_csv,
        )
        frame = pd.read_csv(data_csv)
        missing_keys = [key for key in MERGE_KEYS if key not in frame.columns]
        if missing_keys:
            raise ValueError(
                "Combined CSV is missing identifier columns required for grouped CV: "
                f"{missing_keys}"
            )
        if "reaction_strength" not in frame.columns:
            raise ValueError(
                "Combined CSV must include a 'reaction_strength' column encoding 0-5 intensity scores."
            )
        intensity = pd.to_numeric(frame["reaction_strength"], errors="raise").astype(int)
        if (intensity < 0).any():
            raise ValueError("reaction_strength cannot contain negative values")
        labels = (intensity > 0).astype(int)
        frame[LABEL_INTENSITY_COLUMN] = intensity
        frame[LABEL_COLUMN] = labels
        frame[NON_REACTIVE_FLAG_COLUMN] = (labels == 0).astype(int)

    logger.info("Binary label distribution: %s", labels.value_counts().sort_index().to_dict())
    logger.info(
        "Label intensity distribution: %s",
        intensity.value_counts().sort_index().to_dict(),
    )

    weights = pd.Series(1.0, index=frame.index, dtype=float)
    weights.loc[intensity == 5] = 5.0

    if (weights <= 0).any():
        raise ValueError("Sample weights must be strictly positive across all rows.")

    logger.info(
        "Sample weight summary | min=%.2f | mean=%.2f | max=%.2f",
        float(weights.min()),
        float(weights.mean()),
        float(weights.max()),
    )

    group_series = frame[["dataset", "fly", "fly_number"]].astype(str).agg("_".join, axis=1)

    logger.info(
        "Loaded dataset with %d rows, %d features, and %d unique flies.",
        frame.shape[0],
        frame.drop(columns=[*MERGE_KEYS, LABEL_COLUMN, LABEL_INTENSITY_COLUMN], errors="ignore").shape[1],
        group_series.nunique(),
    )

    return DatasetBundle(
        features=frame,
        labels=labels,
        groups=group_series,
        sample_weights=weights,
        label_intensity=intensity,
    )


def build_pipeline(
    *,
    n_components: int,
    hidden_layer_sizes: Tuple[int, ...],
    alpha: float,
    batch_size: int,
    learning_rate_init: float,
) -> Pipeline:
    """Construct the preprocessing and classifier pipeline."""

    mlp = SampleWeightedMLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
            ("mlp", mlp),
        ]
    )
    return pipeline


def build_pipeline_from_params(params: Mapping[str, object]) -> Pipeline:
    """Convenience helper to construct a pipeline from a parameter mapping."""

    hidden_layers = resolve_hidden_layer_sizes_from_params(params)
    learning_rate_key = "learning_rate_init" if "learning_rate_init" in params else "learning_rate"
    return build_pipeline(
        n_components=int(params["n_components"]),
        hidden_layer_sizes=hidden_layers,
        alpha=float(params["alpha"]),
        batch_size=int(params["batch_size"]),
        learning_rate_init=float(params[learning_rate_key]),
    )


def _combine_sample_and_class_weights(
    labels: pd.Series,
    sample_weights: pd.Series,
    class_weight: Mapping[int, float],
) -> np.ndarray:
    """Multiply proportional sample weights by class penalties."""

    combined = sample_weights.to_numpy(dtype=float, copy=True)
    label_array = labels.to_numpy()
    for class_label, multiplier in class_weight.items():
        combined[label_array == int(class_label)] *= float(multiplier)
    return combined


def _derive_two_layer_baseline(hidden: int) -> Tuple[int, int]:
    """Derive a deterministic two-layer baseline shape from the requested width."""

    if hidden not in ALLOWED_LAYER_WIDTHS:
        raise ValueError(
            "Baseline hidden size must belong to the supported width set. Received "
            f"{hidden}."
        )

    ordered_widths = sorted(ALLOWED_LAYER_WIDTHS)
    index = ordered_widths.index(hidden)
    if index == 0:
        return hidden, hidden
    return hidden, ordered_widths[index - 1]


def _verify_group_separation(groups: pd.Series, train_idx: Iterable[int], val_idx: Iterable[int]) -> None:
    """Raise if any fly identifiers leak between train and validation folds."""

    train_groups = set(groups.iloc[train_idx])
    val_groups = set(groups.iloc[val_idx])
    if train_groups & val_groups:
        raise RuntimeError("Group leakage detected between train and validation folds.")


def evaluate_pipeline(
    *,
    pipeline: Pipeline,
    bundle: DatasetBundle,
    logger: logging.Logger,
    trial: Trial | None = None,
    feature_frame: pd.DataFrame | None = None,
    selected_features: Optional[Sequence[str]] = None,
    class_weight: Mapping[int, float] | None = None,
) -> EvaluationResult:
    """Execute grouped cross-validation and report metrics."""

    if feature_frame is None:
        feature_frame = _prepare_dataframe_from_dataset(bundle, selected_features)
    cv = GroupKFold(n_splits=5)

    fold_macro_f1: List[float] = []
    fold_balanced: List[float] = []
    fold_positive_f1: List[float] = []
    fold_negative_f1: List[float] = []
    fit_durations: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(feature_frame, bundle.labels, groups=bundle.groups)):
        _verify_group_separation(bundle.groups, train_idx, val_idx)
        train_groups = bundle.groups.iloc[train_idx].unique()
        val_groups = bundle.groups.iloc[val_idx].unique()
        logger.info(
            "Fold %d/%d | train flies=%d | val flies=%d",
            fold_idx + 1,
            cv.get_n_splits(),
            len(train_groups),
            len(val_groups),
        )

        X_train = feature_frame.iloc[train_idx]
        X_val = feature_frame.iloc[val_idx]
        y_train = bundle.labels.iloc[train_idx]
        y_val = bundle.labels.iloc[val_idx]
        sw_train = bundle.sample_weights.iloc[train_idx]
        if class_weight is not None:
            sample_weight_array = _combine_sample_and_class_weights(
                y_train,
                sw_train,
                class_weight,
            )
        else:
            sample_weight_array = sw_train.to_numpy()

        start_time = time.perf_counter()
        try:
            pipeline.fit(X_train, y_train, mlp__sample_weight=sample_weight_array)
        except Exception as exc:
            if trial is not None:
                logger.error("Trial %s failed during fit on fold %d: %s", trial.number, fold_idx + 1, exc)
            else:
                logger.error("Baseline evaluation failed during fit on fold %d: %s", fold_idx + 1, exc)
            raise
        duration = time.perf_counter() - start_time
        fit_durations.append(duration)

        y_pred = pipeline.predict(X_val)

        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        balanced = balanced_accuracy_score(y_val, y_pred)
        per_class = f1_score(y_val, y_pred, average=None, labels=[0, 1], zero_division=0)
        fold_macro_f1.append(float(macro_f1))
        fold_balanced.append(float(balanced))
        fold_negative_f1.append(float(per_class[0]))
        fold_positive_f1.append(float(per_class[1]))

        logger.info(
            "Fold %d metrics | Macro-F1=%.4f | Balanced=%.4f | F1[0]=%.4f | F1[1]=%.4f | fit_time=%.2fs",
            fold_idx + 1,
            macro_f1,
            balanced,
            per_class[0],
            per_class[1],
            duration,
        )

        if trial is not None:
            trial.report(float(macro_f1), step=fold_idx)
            if trial.should_prune():
                logger.warning(
                    "Trial %s pruned at fold %d with interim Macro-F1=%.4f",
                    trial.number,
                    fold_idx + 1,
                    macro_f1,
                )
                raise optuna.TrialPruned()

    return EvaluationResult(
        fold_macro_f1=fold_macro_f1,
        fold_balanced_accuracy=fold_balanced,
        fold_positive_f1=fold_positive_f1,
        fold_negative_f1=fold_negative_f1,
        fit_durations=fit_durations,
    )


def objective_factory(
    *,
    bundle: DatasetBundle,
    logger: logging.Logger,
    feature_frame: pd.DataFrame,
    selected_features: Optional[Sequence[str]],
    class_weight: Mapping[int, float] | None,
    model_type: str,
) -> optuna.ObjectiveFuncType:
    """Create the Optuna objective closure bound to the dataset and logger.

    Parameters
    ----------
    feature_frame:
        Pre-sliced feature matrix that respects the caller's engineered feature
        selection. Sharing a single DataFrame avoids redundant column drops for
        every trial.
    selected_features:
        Optional tuple of engineered features retained in ``feature_frame``.
    """

    feature_dim = feature_frame.shape[1]
    if feature_dim == 0:
        raise ValueError(
            "No feature columns available for optimisation. Ensure --features yields at least one column."
        )

    component_candidates = _generate_component_candidates(feature_dim)

    architecture_candidates = _architecture_candidates_for_model(model_type)

    def objective(trial: Trial) -> float:
        n_components = int(
            trial.suggest_categorical("n_components", component_candidates)
        )
        alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
        batch_size = int(trial.suggest_categorical("batch_size", ALLOWED_BATCH_SIZES))
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)

        architecture = trial.suggest_categorical("architecture", architecture_candidates)
        if architecture == ARCHITECTURE_SINGLE:
            hidden_size = int(trial.suggest_categorical("h1", ALLOWED_LAYER_WIDTHS))
            hidden_layer_sizes = (hidden_size,)
        else:
            h1 = int(trial.suggest_categorical("h1", ALLOWED_LAYER_WIDTHS))
            h2 = int(trial.suggest_categorical("h2", ALLOWED_LAYER_WIDTHS))
            hidden_layer_sizes = (h1, h2)

        pipeline = build_pipeline(
            n_components=int(n_components),
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=float(alpha),
            batch_size=int(batch_size),
            learning_rate_init=float(learning_rate_init),
        )

        logger.info(
            "Trial %d configuration | architecture=%s | components=%d | alpha=%.3e | batch=%d | lr=%.3e | hidden=%s",
            trial.number,
            architecture,
            int(n_components),
            alpha,
            int(batch_size),
            learning_rate_init,
            hidden_layer_sizes,
        )

        trial.set_user_attr("architecture", architecture)
        trial.set_user_attr("model_type", model_type)

        try:
            result = evaluate_pipeline(
                pipeline=pipeline,
                bundle=bundle,
                logger=logger,
                trial=trial,
                feature_frame=feature_frame,
                selected_features=selected_features,
                class_weight=class_weight,
            )
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.error("Trial %d encountered an error and will be pruned: %s", trial.number, exc)
            raise optuna.TrialPruned() from exc

        trial.set_user_attr("fold_macro_f1", result.fold_macro_f1)
        trial.set_user_attr("fold_balanced_accuracy", result.fold_balanced_accuracy)
        trial.set_user_attr("fold_positive_f1", result.fold_positive_f1)
        trial.set_user_attr("fold_negative_f1", result.fold_negative_f1)
        trial.set_user_attr("fit_durations", result.fit_durations)

        mean_macro_f1 = result.mean_macro_f1
        logger.info(
            "Trial %d summary | Macro-F1 mean=%.4f ± %.4f | Balanced mean=%.4f",
            trial.number,
            mean_macro_f1,
            result.std_macro_f1,
            result.mean_balanced_accuracy,
        )
        return mean_macro_f1

    return objective


def _baseline_params(
    hidden: int,
    components: int,
    *,
    model_type: str,
    class_weight: Mapping[int, float] | None,
) -> dict:
    params = {
        "n_components": components,
        "alpha": 1e-4,
        "batch_size": 32,
        "learning_rate_init": 1e-3,
        "model_type": model_type,
    }
    if model_type == MODEL_FP_OPTIMIZED_MLP:
        first, second = _derive_two_layer_baseline(hidden)
        params.update(
            {
                "hidden_layer_sizes": (first, second),
                "architecture": ARCHITECTURE_TWO_LAYER,
                "layer_config": f"{first}_{second}",
                "h1": first,
                "h2": second,
            }
        )
    else:
        params.update(
            {
                "hidden_layer_sizes": (hidden,),
                "architecture": ARCHITECTURE_SINGLE,
                "h1": hidden,
            }
        )
    if class_weight is not None:
        params["class_weight"] = {int(k): float(v) for k, v in class_weight.items()}
    return params


def _apply_component_constraints(
    params: Mapping[str, object],
    *,
    feature_dim: int,
    logger: logging.Logger,
    context: str,
) -> dict:
    """Ensure PCA components respect the available feature dimensionality."""

    component_candidates = _generate_component_candidates(feature_dim)
    min_components = min(component_candidates)
    max_components = max(component_candidates)

    requested = int(params["n_components"])
    if requested < min_components:
        raise ValueError(
            f"{context} PCA components {requested} fall below the minimum allowable "
            f"{min_components} for {feature_dim} feature columns."
        )

    adjusted = int(min(requested, max_components))
    if adjusted != requested:
        logger.warning(
            "%s PCA components %d exceed the available feature dimension (%d); using %d instead.",
            context,
            requested,
            feature_dim,
            adjusted,
        )

    constrained = dict(params)
    constrained["n_components"] = adjusted
    return constrained


def run_baseline(
    *,
    bundle: DatasetBundle,
    hidden: int,
    components: int,
    logger: logging.Logger,
    feature_frame: pd.DataFrame,
    selected_features: Optional[Sequence[str]],
    class_weight: Mapping[int, float] | None,
) -> Tuple[dict, EvaluationResult]:
    """Evaluate a deterministic baseline configuration for reporting.

    The precomputed ``feature_frame`` ensures that baseline scoring honours the
    same engineered feature subset supplied via the command line.
    """

    if int(hidden) not in ALLOWED_LAYER_WIDTHS:
        raise ValueError(
            "Baseline hidden size must be chosen from the supported widths "
            f"{ALLOWED_LAYER_WIDTHS}. Received {hidden}."
        )

    params = _baseline_params(
        hidden,
        components,
        model_type=model_type,
        class_weight=class_weight,
    )
    params = _apply_component_constraints(
        params,
        feature_dim=feature_frame.shape[1],
        logger=logger,
        context="Baseline",
    )
    pipeline = build_pipeline_from_params(params)
    logger.info(
        "Evaluating baseline model | architecture=%s | components=%d | hidden_layers=%s",
        params["architecture"],
        params["n_components"],
        params["hidden_layer_sizes"],
    )
    result = evaluate_pipeline(
        pipeline=pipeline,
        bundle=bundle,
        logger=logger,
        trial=None,
        feature_frame=feature_frame,
        selected_features=selected_features,
        class_weight=class_weight,
    )
    return params, result


def ensure_output_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_visualisations(study: optuna.Study, output_dir: Path, logger: logging.Logger) -> None:
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
    except Exception as exc:  # pragma: no cover - optional dependency guard
        logger.warning("Optuna visualisation modules are unavailable: %s", exc)
        return

    history_path = output_dir / "optuna_history.html"
    importances_path = output_dir / "optuna_importances.html"

    try:
        plot_optimization_history(study).write_html(str(history_path))
        plot_param_importances(study).write_html(str(importances_path))
        logger.info("Saved Optuna visualisations to %s", output_dir)
    except Exception as exc:  # pragma: no cover - plotly runtime guard
        logger.warning("Failed to generate Optuna visualisations: %s", exc)


def save_report(
    *,
    output_dir: Path,
    logger: logging.Logger,
    baseline_params: dict,
    baseline_result: EvaluationResult,
    best_params: dict,
    best_result: EvaluationResult,
    study: optuna.Study | None,
    selected_features: Optional[Sequence[str]],
) -> None:
    report_path = output_dir / "TUNING_REPORT.md"

    if study is not None:
        completed_trials = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        pruned_trials = len([t for t in study.trials if t.state == TrialState.PRUNED])
        summary = (
            "Optuna explored {:d} completed trials ({} pruned) using a median-pruned "
            "TPE sampler. The best configuration achieved a macro-F1 of {:.4f} "
            "across five fly-grouped folds, outperforming the deterministic baseline "
            "macro-F1 of {:.4f}."
        ).format(
            completed_trials,
            pruned_trials,
            best_result.mean_macro_f1,
            baseline_result.mean_macro_f1,
        )
    else:
        summary = (
            "Optimisation was skipped in favour of a supplied parameter set. The provided "
            "configuration achieved a macro-F1 of {:.4f} across five fly-grouped folds, "
            "outperforming the deterministic baseline macro-F1 of {:.4f}."
        ).format(best_result.mean_macro_f1, baseline_result.mean_macro_f1)

    best_params_serialisable = dict(best_params)
    best_params_serialisable["hidden_layer_sizes"] = list(best_params_serialisable["hidden_layer_sizes"])
    best_params_serialisable["selected_features"] = (
        list(selected_features) if selected_features is not None else None
    )
    baseline_params_serialisable = dict(baseline_params)
    baseline_params_serialisable["hidden_layer_sizes"] = list(baseline_params_serialisable["hidden_layer_sizes"])

    if study is not None:
        artefact_section = (
            "## Optuna Study Artefacts\n\n"
            "* Study database: `optuna_study.db`\n"
            "* Trials table: `optuna_trials.csv`\n"
            "* Optimisation history: `optuna_history.html`\n"
            "* Hyperparameter importances: `optuna_importances.html`\n"
        )
        importance_section = (
            "## Hyperparameter Importance Insights\n\n"
            "The Optuna importance analysis ranks parameter influence on macro-F1 using the\n"
            "functional ANOVA approach. Inspect `optuna_importances.html` for an interactive\n"
            "view that highlights the relative contribution of PCA dimensionality alongside\n"
            "network architecture choices.\n"
        )
    else:
        artefact_section = (
            "## Optuna Study Artefacts\n\n"
            "Optuna was not executed during this run, so study artefacts were not generated.\n"
        )
        importance_section = (
            "## Hyperparameter Importance Insights\n\n"
            "Hyperparameter importance plots are unavailable because optimisation was bypassed.\n"
            "Re-run Optuna to obtain interactive diagnostics.\n"
        )

    model_variant = best_params.get("model_type", MODEL_MLP)
    if best_params.get("class_weight"):
        weight_mapping = best_params["class_weight"]
        class_weight_summary = (
            "Responder-focused class weighting was enabled with the mapping "
            f"{weight_mapping}."
        )
    else:
        class_weight_summary = (
            "No additional class weighting was applied beyond the intensity-derived sample weights."
        )

    if selected_features is None:
        feature_clause = (
            "All engineered scalar features present in the dataset were included in the search."
        )
    else:
        feature_clause = (
            "The optimisation was restricted to the following engineered features: "
            + ", ".join(selected_features)
        )

    feature_summary = (
        f"The optimisation targeted the `{model_variant}` pipeline. "
        f"{class_weight_summary} {feature_clause}"
    )

    contents = f"""# Optuna Tuning Report

## Executive Summary

{summary}

## Feature Configuration

{feature_summary}

## Best Hyperparameters

```json
{json.dumps(best_params_serialisable, indent=2)}
```

* Mean Macro-F1: {best_result.mean_macro_f1:.4f}
* Macro-F1 Std: {best_result.std_macro_f1:.4f}
* Mean Balanced Accuracy: {best_result.mean_balanced_accuracy:.4f}
* Positive-Class F1: {best_result.mean_positive_f1:.4f}
* Negative-Class F1: {best_result.mean_negative_f1:.4f}

## Baseline Comparison

Baseline parameters:

```json
{json.dumps(baseline_params_serialisable, indent=2)}
```

* Mean Macro-F1: {baseline_result.mean_macro_f1:.4f}
* Macro-F1 Std: {baseline_result.std_macro_f1:.4f}
* Mean Balanced Accuracy: {baseline_result.mean_balanced_accuracy:.4f}
* Positive-Class F1: {baseline_result.mean_positive_f1:.4f}
* Negative-Class F1: {baseline_result.mean_negative_f1:.4f}

{artefact_section}

{importance_section}

## Recommendations for Production

1. Retrain the provided pipeline on the full dataset whenever new trials become
   available to keep the PCA transformation aligned with the latest sampling.
2. Persist the Optuna SQLite storage (`optuna_study.db`) to allow warm-started
   tuning sessions as the behavioural corpus grows.
3. Monitor macro-F1 and balanced accuracy on a hold-out cohort of flies to
   validate that the sample-weight strategy continues to prioritise high-intensity
   responders without overfitting specific experiments.
"""

    report_path.write_text(contents)
    logger.info("Wrote tuning report to %s", report_path)


def retrain_final_model(
    *,
    bundle: DatasetBundle,
    best_params: dict,
    output_dir: Path,
    logger: logging.Logger,
    feature_frame: pd.DataFrame,
    class_weight: Mapping[int, float] | None,
    model_type: str,
) -> Path:
    """Retrain the best-scoring pipeline on the full dataset.

    The provided ``feature_frame`` guarantees that retraining honours the same
    engineered feature subset that produced the winning score.
    """

    constrained_params = _apply_component_constraints(
        best_params,
        feature_dim=feature_frame.shape[1],
        logger=logger,
        context="Final retraining",
    )
    pipeline = build_pipeline_from_params(constrained_params)

    logger.info("Retraining best pipeline on the full dataset (%d samples)", len(feature_frame))
    if class_weight is not None:
        final_sample_weight = _combine_sample_and_class_weights(
            bundle.labels,
            bundle.sample_weights,
            class_weight,
        )
    else:
        final_sample_weight = bundle.sample_weights.to_numpy()

    pipeline.fit(
        feature_frame,
        bundle.labels,
        mlp__sample_weight=final_sample_weight,
    )

    suffix = model_type
    model_filename = f"best_{suffix}_model.joblib"
    model_path = output_dir / model_filename
    joblib.dump(pipeline, model_path)
    logger.info("Saved retrained pipeline to %s", model_path)
    return model_path

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logger = configure_logging(args.verbose)

    random.seed(42)
    np.random.seed(42)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    ensure_output_directory(args.output_dir)

    logger.info("Loading dataset for optimisation")
    bundle = load_training_data(
        data_csv=args.data_csv,
        labels_csv=args.labels_csv,
        logger=logger,
    )

    selected_features = _normalise_feature_list(args.features)
    feature_frame = _prepare_dataframe_from_dataset(bundle, selected_features)
    if selected_features is None:
        logger.info(
            "Optimisation will consider all %d engineered features.",
            feature_frame.shape[1],
        )
    else:
        logger.info(
            "Restricting optimisation to %d user-specified features: %s",
            len(selected_features),
            ", ".join(selected_features),
        )

    model_type = args.model
    if model_type == MODEL_FP_OPTIMIZED_MLP:
        class_weight: Mapping[int, float] | None = CLASS_WEIGHT_FP_OPTIMIZED
        logger.info(
            "Applying responder-focused class weighting for fp_optimized_mlp: %s",
            class_weight,
        )
    else:
        class_weight = None

    baseline_params, baseline_result = run_baseline(
        bundle=bundle,
        hidden=args.baseline_hidden,
        components=args.baseline_components,
        logger=logger,
        feature_frame=feature_frame,
        selected_features=selected_features,
        class_weight=class_weight,
        model_type=model_type,
    )

    study: optuna.Study | None = None
    if args.best_params_json is not None:
        logger.info(
            "Bypassing optimisation; loading parameter set from %s", args.best_params_json
        )
        best_params_raw = json.loads(args.best_params_json.read_text())
        best_params = normalise_mlp_params(best_params_raw)
        if "model_type" not in best_params:
            best_params["model_type"] = model_type
        if class_weight is not None:
            best_params["class_weight"] = {
                int(k): float(v) for k, v in class_weight.items()
            }
        best_params = _apply_component_constraints(
            best_params,
            feature_dim=feature_frame.shape[1],
            logger=logger,
            context="Best-parameter replay",
        )
        best_pipeline = build_pipeline_from_params(best_params)
        best_result = evaluate_pipeline(
            pipeline=best_pipeline,
            bundle=bundle,
            logger=logger,
            feature_frame=feature_frame,
            selected_features=selected_features,
            class_weight=class_weight,
        )
    else:
        storage_path = args.output_dir / "optuna_study.db"
        storage = optuna.storages.RDBStorage(url=f"sqlite:///{storage_path}")
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2, interval_steps=1),
        )

        objective = objective_factory(
            bundle=bundle,
            logger=logger,
            feature_frame=feature_frame,
            selected_features=selected_features,
            class_weight=class_weight,
            model_type=model_type,
        )
        timeout = None if args.timeout == 0 else args.timeout
        study.optimize(objective, n_trials=args.n_trials, timeout=timeout)

        logger.info("Optimisation finished | best value=%.4f", study.best_value)
        logger.info("Best trial parameters: %s", study.best_trial.params)

        best_params = normalise_mlp_params(study.best_trial.params)
        best_params["model_type"] = model_type
        if class_weight is not None:
            best_params["class_weight"] = {
                int(k): float(v) for k, v in class_weight.items()
            }
        best_params = _apply_component_constraints(
            best_params,
            feature_dim=feature_frame.shape[1],
            logger=logger,
            context="Optuna best trial",
        )
        best_pipeline = build_pipeline_from_params(best_params)
        best_result = evaluate_pipeline(
            pipeline=best_pipeline,
            bundle=bundle,
            logger=logger,
            feature_frame=feature_frame,
            selected_features=selected_features,
            class_weight=class_weight,
        )

    retrain_final_model(
        bundle=bundle,
        best_params=best_params,
        output_dir=args.output_dir,
        logger=logger,
        feature_frame=feature_frame,
        class_weight=class_weight,
        model_type=model_type,
    )

    if study is not None:
        trials_df = study.trials_dataframe()
        trials_path = args.output_dir / "optuna_trials.csv"
        trials_df.to_csv(trials_path, index=False)
        logger.info("Exported Optuna trials to %s", trials_path)

    best_params_path = args.output_dir / "best_params.json"
    serialisable_params = dict(best_params)
    serialisable_params["hidden_layer_sizes"] = list(serialisable_params["hidden_layer_sizes"])
    if selected_features is not None:
        serialisable_params["selected_features"] = list(selected_features)
    else:
        serialisable_params["selected_features"] = None
    best_params_path.write_text(json.dumps(serialisable_params, indent=2))
    logger.info("Saved best parameters to %s", best_params_path)

    if study is not None:
        save_visualisations(study, args.output_dir, logger)

    save_report(
        output_dir=args.output_dir,
        logger=logger,
        baseline_params=baseline_params,
        baseline_result=baseline_result,
        best_params=best_params,
        best_result=best_result,
        study=study,
        selected_features=selected_features,
    )


if __name__ == "__main__":
    main()


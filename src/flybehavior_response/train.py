"""Training routines for flybehavior_response."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import PipelineConfig, compute_class_balance, hash_file, make_run_artifacts
from .evaluate import evaluate_models, perform_cross_validation, save_metrics
from .features import build_column_transformer, validate_features
from .io import (
    LABEL_COLUMN,
    LABEL_INTENSITY_COLUMN,
    MERGE_KEYS,
    NON_REACTIVE_FLAG_COLUMN,
    load_dataset,
)
from .logging_utils import get_logger
from .modeling import (
    MODEL_FP_OPTIMIZED_MLP,
    MODEL_HGB,
    MODEL_LDA,
    MODEL_LOGREG,
    MODEL_MLP,
    MODEL_RF,
    MODEL_XGB,
    build_model_pipeline,
    supported_models,
)
from .weights import expand_samples_by_weight


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _stratified_train_val_test_split(
    *,
    y: pd.Series,
    seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stratified indices for an explicit train/validation/test split.

    The helper keeps class proportions stable across all partitions by chaining
    two stratified ``train_test_split`` calls that share a deterministic
    ``random_state``. The requested fractions must sum to one.
    """

    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("Train/validation/test fractions must sum to 1.0")
    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )
    relative_val_fraction = val_size / (train_size + val_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_fraction,
        stratify=y.iloc[train_val_idx],
        random_state=seed,
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def _combine_sample_and_class_weights(
    labels: pd.Series,
    sample_weights: pd.Series,
    class_weight: Dict[int, float],
) -> np.ndarray:
    """Multiply proportional sample weights by class-based penalties.

    Parameters
    ----------
    labels:
        Series of binary labels aligned with ``sample_weights``.
    sample_weights:
        Base weights derived from label intensity.
    class_weight:
        Mapping from class label to multiplicative weight (e.g. ``{0: 1.0, 1: 2.0}``).

    Returns
    -------
    np.ndarray
        Combined weights ready to forward to ``MLPClassifier.fit``.
    """

    combined = sample_weights.to_numpy(dtype=float, copy=True)
    label_array = labels.to_numpy()
    for class_label, multiplier in class_weight.items():
        combined[label_array == class_label] *= float(multiplier)
    return combined


def train_models(
    *,
    data_csv: Path | None,
    labels_csv: Path,
    features: Sequence[str],
    include_traces: bool,
    use_raw_pca: bool,
    n_pcs: int,
    models: Sequence[str],
    artifacts_dir: Path,
    cv: int,
    seed: int,
    verbose: bool,
    dry_run: bool = False,
    logreg_solver: str = "lbfgs",
    logreg_max_iter: int = 1000,
    logreg_class_weight: Mapping[int, float] | str | None = None,
    rf_n_estimators: int = 100,
    rf_max_depth: int | None = None,
    rf_class_weight: Mapping[int, float] | str | None = None,
    trace_prefixes: Sequence[str] | None = None,
    geometry_source: Path | None = None,
    geom_chunk_size: int = 100_000,
    geom_columns: Sequence[str] | None = None,
    geom_cache_parquet: Path | None = None,
    geom_use_cache: bool = False,
    geom_frame_column: str = "frame_idx",
    geom_stats: Sequence[str] | None = None,
    geom_granularity: str = "trial",
    geom_normalization: str = "none",
    geom_drop_missing_labels: bool = True,
    geom_downcast: bool = True,
    geom_trial_summary: Path | None = None,
    geom_feature_columns: Sequence[str] | None = None,
    group_column: str = "fly",
    group_override: str | None = None,
    test_size: float = 0.2,
    mlp_params: Mapping[str, object] | None = None,
    classification_mode: str = 'binary',
) -> Dict[str, Dict[str, object]]:
    logger = get_logger(__name__, verbose=verbose)
    _set_seeds(seed)

    dataset = load_dataset(
        data_csv=data_csv,
        labels_csv=labels_csv,
        logger_name=__name__,
        trace_prefixes=trace_prefixes,
        include_traces=include_traces,
        geometry_source=geometry_source,
        geom_chunk_size=geom_chunk_size,
        geom_columns=geom_columns,
        geom_cache_parquet=geom_cache_parquet,
        geom_use_cache=geom_use_cache,
        geom_frame_column=geom_frame_column,
        geom_stats=geom_stats,
        geom_granularity=geom_granularity,
        geom_normalization=geom_normalization,
        geom_drop_missing_labels=geom_drop_missing_labels,
        geom_downcast=geom_downcast,
        geom_trial_summary=geom_trial_summary,
        geom_feature_columns=geom_feature_columns,
        classification_mode=classification_mode,
    )
    resolved_prefixes = list(dataset.trace_prefixes)
    logger.debug("Trace prefixes resolved to: %s", resolved_prefixes)

    available_features = list(dataset.feature_columns)
    if (
        NON_REACTIVE_FLAG_COLUMN in dataset.frame.columns
        and NON_REACTIVE_FLAG_COLUMN not in available_features
    ):
        available_features.append(NON_REACTIVE_FLAG_COLUMN)

    if available_features:
        try:
            selected_features = validate_features(
                features, available_features, logger_name=__name__
            )
        except ValueError as exc:
            available_set = set(available_features)
            requested_available = [
                feat for feat in features if feat in available_set
            ]
            if requested_available:
                available = ", ".join(available_features)
                logger.error(
                    "Requested engineered features %s are not fully available. Dataset columns: %s.",
                    sorted(features),
                    available,
                )
                raise
            logger.warning(
                "Requested engineered features %s are unavailable for this dataset; proceeding without engineered scalars.",
                sorted(features),
            )
            selected_features = []
        logger.debug("Selected features: %s", selected_features)
    else:
        if features:
            logger.warning(
                "Ignoring requested engineered features %s because the dataset does not include "
                "any of the expected columns.",
                sorted(features),
            )
        selected_features = []
        logger.debug("Dataset does not contain engineered features; proceeding with trace-only preprocessing.")

    if use_raw_pca and not dataset.trace_columns:
        logger.info("Disabling raw PCA because no trace columns are available in the dataset.")
        use_raw_pca = False

    if not selected_features and not dataset.trace_columns and dataset.feature_columns:
        selected_features = list(dataset.feature_columns)
        logger.info(
            "No trace columns available; using %d geometry feature columns for preprocessing.",
            len(selected_features),
        )

    if dataset.sample_weights is None or dataset.label_intensity is None:
        raise ValueError("Training requires labelled data with computed sample weights.")

    sample_weights = dataset.sample_weights
    logger.info(
        "Applying proportional sample weights derived from label intensities (min=%.2f mean=%.2f max=%.2f)",
        float(sample_weights.min()),
        float(sample_weights.mean()),
        float(sample_weights.max()),
    )

    if mlp_params is not None:
        optuna_features = mlp_params.get("selected_features")
        if optuna_features:
            resolved_optuna_features = validate_features(
                optuna_features, dataset.feature_columns, logger_name=__name__
            )
            if selected_features and selected_features != resolved_optuna_features:
                logger.warning(
                    "Overriding CLI feature selection %s with Optuna-selected subset %s.",
                    selected_features,
                    resolved_optuna_features,
                )
            selected_features = resolved_optuna_features
            logger.info(
                "Applying Optuna-selected engineered feature subset: %s",
                ", ".join(selected_features),
            )

    preprocessor = build_column_transformer(
        trace_columns=dataset.trace_columns,
        feature_columns=dataset.feature_columns,
        selected_features=selected_features,
        use_raw_pca=use_raw_pca,
        n_pcs=n_pcs,
        seed=seed,
    )

    requested_models = list(models)
    if not requested_models:
        requested_models = list(supported_models())

    if "both" in requested_models or "all" in requested_models:
        logger.warning(
            "Received legacy model keyword in train_models; please provide explicit model list."
        )
        merged = []
        for name in requested_models:
            if name == "both":
                merged.extend([MODEL_LDA, MODEL_LOGREG])
            elif name == "all":
                merged.extend(supported_models())
            else:
                merged.append(name)
        requested_models = list(dict.fromkeys(merged))

    invalid = [name for name in requested_models if name not in supported_models()]
    if invalid:
        raise ValueError(f"Unsupported models requested: {invalid}")

    artifacts = make_run_artifacts(artifacts_dir) if not dry_run else None
    if artifacts:
        logger.info("Writing artifacts to %s", artifacts.run_dir)

    trace_indices = []
    for col in dataset.trace_columns:
        digits = "".join(ch for ch in col if ch.isdigit())
        if digits:
            trace_indices.append(int(digits))
    if trace_indices:
        trace_range = (min(trace_indices), max(trace_indices))
    else:
        trace_range = (0, 0)

    intensity_counts = {
        str(int(k)): int(v)
        for k, v in dataset.label_intensity.value_counts().sort_index().items()
    }
    weight_summary = {
        "min": float(sample_weights.min()),
        "mean": float(sample_weights.mean()),
        "max": float(sample_weights.max()),
    }

    logger.info("Trace series prefixes: %s", resolved_prefixes)
    if mlp_params is not None:
        if not use_raw_pca:
            logger.info(
                "Enabling PCA preprocessing to honour Optuna n_components=%d.",
                int(mlp_params["n_components"]),
            )
            use_raw_pca = True
        if n_pcs != int(mlp_params["n_components"]):
            logger.info(
                "Overriding n_pcs=%d with Optuna-derived n_components=%d.",
                n_pcs,
                int(mlp_params["n_components"]),
            )
            n_pcs = int(mlp_params["n_components"])
        logger.info(
            "Applying Optuna-derived MLP hyperparameters: hidden=%s | alpha=%.3g | batch_size=%d | lr=%.3g | n_components=%d",
            mlp_params.get("hidden_layer_sizes"),
            float(mlp_params["alpha"]),
            int(mlp_params["batch_size"]),
            float(mlp_params["learning_rate_init"]),
            int(mlp_params["n_components"]),
        )

    data_path = geometry_source if geometry_source is not None else data_csv
    if data_path is None:
        raise ValueError("Either data_csv or geometry_source must be provided for training.")

    file_hashes = {
        "data_csv": hash_file(data_path),
        "labels_csv": hash_file(labels_csv),
    }
    if geom_trial_summary is not None:
        file_hashes["geometry_trial_summary_csv"] = hash_file(geom_trial_summary)

    resolved_group_column = group_column
    if group_override is not None:
        override = group_override.strip().lower()
        if override in {"", "none"}:
            resolved_group_column = None
        else:
            resolved_group_column = group_override

    groups_series: pd.Series | None = None
    if resolved_group_column and resolved_group_column in dataset.frame.columns:
        groups_series = dataset.frame[resolved_group_column].astype(str).fillna("__missing__")
    elif resolved_group_column:
        logger.warning(
            "Group column '%s' not found; proceeding without group-aware splitting.",
            resolved_group_column,
        )
        resolved_group_column = None

    drop_columns = [LABEL_COLUMN]
    if LABEL_INTENSITY_COLUMN in dataset.frame.columns:
        drop_columns.append(LABEL_INTENSITY_COLUMN)
    X = dataset.frame.drop(columns=drop_columns, errors="ignore")
    y = dataset.frame[LABEL_COLUMN].astype(int)

    use_fp_optimised_split = MODEL_FP_OPTIMIZED_MLP in requested_models
    class_weight_dict: Dict[int, float] | None = None
    val_idx: np.ndarray | None = None

    effective_test_size = 0.15 if use_fp_optimised_split else test_size

    if groups_series is not None:
        # =====================================================================
        # HYBRID SPLITTING STRATEGY (Default)
        # Balance class distribution across train/test while maintaining
        # group integrity (no fly leakage). Alternates group assignment by
        # response rate to ensure balanced splits.
        # =====================================================================

        # Calculate response rate per group
        group_stats = dataset.frame.groupby(resolved_group_column).agg({
            LABEL_COLUMN: ['count', 'mean']
        })
        group_stats.columns = ['n_trials', 'response_rate']

        # Sort groups by response rate
        group_stats_sorted = group_stats.sort_values('response_rate')
        groups_sorted = group_stats_sorted.index.tolist()

        # Alternate assignment to balance response rates
        # Every Nth group → test, rest → train (where N = 1/test_size)
        n_groups = len(groups_sorted)
        test_interval = int(1.0 / effective_test_size)

        test_groups = []
        train_groups = []

        for i, group in enumerate(groups_sorted):
            if i % test_interval == 0:
                test_groups.append(group)
            else:
                train_groups.append(group)

        # Create index masks
        train_mask = dataset.frame[resolved_group_column].isin(train_groups)
        test_mask = dataset.frame[resolved_group_column].isin(test_groups)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        # Verify no group leakage
        train_groups_set = set(train_groups)
        test_groups_set = set(test_groups)
        overlap = train_groups_set & test_groups_set
        if overlap:
            raise RuntimeError(
                "Group leakage detected across splits for groups: %s" % sorted(overlap)
            )

        # Log split statistics
        train_response_rate = y.iloc[train_idx].mean()
        test_response_rate = y.iloc[test_idx].mean()
        balance_gap = abs(train_response_rate - test_response_rate)

        logger.info(
            "Hybrid split: %d train trials (%d groups, %.1f%% responders), "
            "%d test trials (%d groups, %.1f%% responders), balance gap: %.1f%%",
            len(train_idx), len(train_groups), train_response_rate * 100,
            len(test_idx), len(test_groups), test_response_rate * 100,
            balance_gap * 100
        )

        if use_fp_optimised_split:
            logger.warning(
                "Group-aware split in use; fp_optimized_mlp validation fold may not remain class stratified."
            )
            train_idx = np.asarray(train_idx)
            test_idx = np.asarray(test_idx)

            # Apply hybrid split for validation fold too
            train_groups_series = dataset.frame.iloc[train_idx][resolved_group_column]
            train_y = y.iloc[train_idx]

            train_group_stats = dataset.frame.iloc[train_idx].groupby(resolved_group_column).agg({
                LABEL_COLUMN: ['count', 'mean']
            })
            train_group_stats.columns = ['n_trials', 'response_rate']
            train_group_stats_sorted = train_group_stats.sort_values('response_rate')
            train_groups_sorted = train_group_stats_sorted.index.tolist()

            val_interval = int((1 - effective_test_size) / effective_test_size)
            val_groups = [g for i, g in enumerate(train_groups_sorted) if i % val_interval == 0]
            actual_train_groups = [g for g in train_groups_sorted if g not in val_groups]

            val_mask_relative = train_groups_series.isin(val_groups)
            actual_train_mask_relative = train_groups_series.isin(actual_train_groups)

            val_idx = train_idx[np.where(val_mask_relative)[0]]
            train_idx = train_idx[np.where(actual_train_mask_relative)[0]]
    else:
        if use_fp_optimised_split:
            train_idx, val_idx, test_idx = _stratified_train_val_test_split(
                y=y,
                seed=seed,
                train_size=0.70,
                val_size=0.15,
                test_size=0.15,
            )
        else:
            train_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size=test_size,
                stratify=y,
                random_state=seed,
            )

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    sw_train = sample_weights.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    sw_test = sample_weights.iloc[test_idx]

    if use_fp_optimised_split and val_idx is None:
        raise RuntimeError(
            "fp_optimized_mlp requested but validation split could not be produced."
        )

    if val_idx is not None:
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        sw_val = sample_weights.iloc[val_idx]
        logger.info(
            "Split data into training (%d samples, reaction rate=%.2f), validation (%d samples, reaction rate=%.2f) and test (%d samples, reaction rate=%.2f)",
            len(y_train),
            float(y_train.mean()) if len(y_train) else 0.0,
            len(y_val),
            float(y_val.mean()) if len(y_val) else 0.0,
            len(y_test),
            float(y_test.mean()) if len(y_test) else 0.0,
        )
    else:
        X_val = None
        y_val = None
        sw_val = None
        logger.info(
            "Split data into training (%d samples, reaction rate=%.2f) and test (%d samples, reaction rate=%.2f)",
            len(y_train),
            float(y_train.mean()) if len(y_train) else 0.0,
            len(y_test),
            float(y_test.mean()) if len(y_test) else 0.0,
        )

    # Check for custom class weights
    if mlp_params is not None and "class_weight" in mlp_params:
        class_weight_dict = mlp_params["class_weight"]
        if use_fp_optimised_split:
            logger.info(
                "fp_optimized_mlp will apply CUSTOM class weights %s on top of proportional sample weights.",
                class_weight_dict,
            )
        else:
            logger.info(
                "MLP models will apply custom class weights %s on top of proportional sample weights.",
                class_weight_dict,
            )
    elif use_fp_optimised_split:
        class_weight_dict = {0: 1.0, 1: 1.25}
        logger.info(
            "fp_optimized_mlp will apply default class weights %s on top of proportional sample weights.",
            class_weight_dict,
        )

    # Log logistic regression class weights if provided
    if logreg_class_weight is not None:
        logger.info(
            "Logistic regression will apply class weights %s to reduce false negatives.",
            logreg_class_weight,
        )

    # Log random forest class weights if provided
    if rf_class_weight is not None:
        logger.info(
            "Random Forest will apply class weights %s to reduce false negatives.",
            rf_class_weight,
        )

    metrics: Dict[str, Dict[str, object]] = {}

    mlp_params_for_config = None
    if mlp_params is not None:
        mlp_params_for_config = dict(mlp_params)
        hidden_layers = mlp_params_for_config.get("hidden_layer_sizes")
        if isinstance(hidden_layers, tuple):
            mlp_params_for_config["hidden_layer_sizes"] = list(hidden_layers)
        selected_subset = mlp_params_for_config.get("selected_features")
        if isinstance(selected_subset, tuple):
            mlp_params_for_config["selected_features"] = list(selected_subset)

    config = PipelineConfig(
        features=list(selected_features),
        n_pcs=n_pcs,
        use_raw_pca=use_raw_pca,
        seed=seed,
        models=list(requested_models),
        trace_column_range=trace_range,
        data_csv=str(data_path),
        labels_csv=str(labels_csv),
        file_hashes=file_hashes,
        class_balance=compute_class_balance(dataset.frame[LABEL_COLUMN].astype(int).tolist()),
        logreg_solver=logreg_solver,
        logreg_max_iter=logreg_max_iter,
        label_intensity_counts=intensity_counts,
        label_weight_summary=weight_summary,
        label_weight_strategy="proportional_intensity",
        trace_series_prefixes=resolved_prefixes,
        use_trace_series=bool(dataset.trace_columns),
        data_format="geometry" if geometry_source is not None else "tabular",
        dataset_granularity=dataset.granularity,
        group_column=resolved_group_column,
        geometry_aggregations=list(geom_stats or []),
        geometry_normalization=dataset.normalization,
        geometry_trial_summary=str(geom_trial_summary) if geom_trial_summary is not None else None,
        geometry_feature_columns=list(geom_feature_columns or []),
        mlp_params=mlp_params_for_config,
    )

    def _write_split_predictions(
        *,
        split_name: str,
        model_name: str,
        pipeline,
        X_split,
        y_split,
        weights_split,
    ) -> None:
        if dry_run:
            return
        assert artifacts is not None
        predictions = pipeline.predict(X_split)
        proba = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X_split)
            if proba is not None and proba.ndim == 2 and proba.shape[1] >= 2:
                proba = proba[:, 1]
            else:
                proba = None
        original_rows = dataset.frame.loc[y_split.index].copy()
        if "model" in original_rows.columns:
            original_rows = original_rows.drop(columns=["model"])
        if "split" in original_rows.columns:
            original_rows = original_rows.drop(columns=["split"])
        original_rows.insert(0, "model", model_name)
        original_rows.insert(1, "split", split_name)
        original_rows["predicted_label"] = predictions.astype(int)
        original_rows["correct"] = (
            original_rows[LABEL_COLUMN].astype(int) == original_rows["predicted_label"]
        )
        if proba is not None:
            original_rows["prob_reaction"] = proba.astype(float)
        weights_aligned = weights_split.reindex(y_split.index)
        original_rows["sample_weight"] = weights_aligned.to_numpy(dtype=float)
        predictions_path = artifacts.run_dir / f"predictions_{model_name}_{split_name}.csv"
        original_rows.to_csv(predictions_path, index=False)
        logger.info("Saved %s predictions to %s", split_name, predictions_path)

    if artifacts is not None:
        manifest_columns = [col for col in MERGE_KEYS if col in dataset.frame.columns]
        if manifest_columns:
            manifest = dataset.frame.loc[:, manifest_columns].copy()
        else:
            manifest = pd.DataFrame(index=dataset.frame.index)
        manifest["split"] = "train"
        manifest.loc[X_test.index, "split"] = "test"
        if X_val is not None:
            manifest.loc[X_val.index, "split"] = "validation"
        if resolved_group_column:
            manifest[resolved_group_column] = dataset.frame[resolved_group_column]
        manifest["granularity"] = dataset.granularity
        manifest_path = artifacts.run_dir / "split_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        logger.info("Saved split manifest to %s", manifest_path)

    for model_name in requested_models:
        logger.info("Training model: %s", model_name)
        pipeline = build_model_pipeline(
            preprocessor,
            model_type=model_name,
            seed=seed,
            logreg_solver=logreg_solver,
            logreg_max_iter=logreg_max_iter,
            logreg_class_weight=logreg_class_weight if model_name == MODEL_LOGREG else None,
            rf_n_estimators=rf_n_estimators,
            rf_max_depth=rf_max_depth,
            rf_class_weight=rf_class_weight if model_name == MODEL_RF else None,
            mlp_params=mlp_params if model_name == MODEL_MLP else None,
        )
        if model_name == MODEL_LDA:
            X_fit, y_fit = expand_samples_by_weight(X_train, y_train, sw_train)
            pipeline.fit(X_fit, y_fit)
        else:
            fit_kwargs = {}
            if model_name == MODEL_LOGREG:
                fit_kwargs["model__sample_weight"] = sw_train.to_numpy()
            elif model_name == MODEL_RF:
                fit_kwargs["model__sample_weight"] = sw_train.to_numpy()
            elif model_name == MODEL_HGB:
                fit_kwargs["model__sample_weight"] = sw_train.to_numpy()
            elif model_name == MODEL_XGB:
                fit_kwargs["model__sample_weight"] = sw_train.to_numpy()
            elif model_name in {MODEL_MLP, MODEL_FP_OPTIMIZED_MLP}:
                # Apply class weights if provided, otherwise just use sample weights
                if class_weight_dict is not None:
                    combined_weights = _combine_sample_and_class_weights(
                        y_train,
                        sw_train,
                        class_weight_dict,
                    )
                    fit_kwargs["model__sample_weight"] = combined_weights
                else:
                    fit_kwargs["model__sample_weight"] = sw_train.to_numpy()
            pipeline.fit(
                X_train,
                y_train,
                **fit_kwargs,
            )

        train_metrics = evaluate_models(
            {model_name: pipeline},
            X_train,
            y_train,
            sample_weight=sw_train,
        )[model_name]
        metrics[model_name] = train_metrics

        if X_val is not None and y_val is not None:
            val_metrics = evaluate_models(
                {model_name: pipeline},
                X_val,
                y_val,
                sample_weight=sw_val,
            )[model_name]
            metrics[model_name]["validation"] = val_metrics

        test_metrics = evaluate_models(
            {model_name: pipeline},
            X_test,
            y_test,
            sample_weight=sw_test,
        )[model_name]
        metrics[model_name]["test"] = test_metrics

        logger.info(
            "Model %s accuracy -> train: %.3f | test: %.3f",
            model_name,
            train_metrics["accuracy"],
            test_metrics["accuracy"],
        )
        # Log F1 score (binary if available, otherwise macro)
        if train_metrics["f1_binary"] is not None:
            logger.info(
                "Model %s F1 (binary) -> train: %.3f | test: %.3f",
                model_name,
                train_metrics["f1_binary"],
                test_metrics["f1_binary"],
            )
        else:
            logger.info(
                "Model %s F1 (macro) -> train: %.3f | test: %.3f",
                model_name,
                train_metrics["f1_macro"],
                test_metrics["f1_macro"],
            )
        if model_name == MODEL_FP_OPTIMIZED_MLP and X_val is not None and y_val is not None:
            logger.info(
                "Model %s precision -> train: %.3f | validation: %.3f | test: %.3f",
                model_name,
                train_metrics["precision"],
                metrics[model_name]["validation"]["precision"],
                test_metrics["precision"],
            )
            logger.info(
                "Model %s false positive rate -> train: %.3f | validation: %.3f | test: %.3f",
                model_name,
                train_metrics["false_positive_rate"],
                metrics[model_name]["validation"]["false_positive_rate"],
                test_metrics["false_positive_rate"],
            )
        model_step = pipeline.named_steps["model"]
        if hasattr(model_step, "n_iter_"):
            n_iter = model_step.n_iter_
            if isinstance(n_iter, np.ndarray):
                iter_count = int(n_iter.max())
            elif isinstance(n_iter, (list, tuple)):
                iter_count = int(max(n_iter))
            else:
                iter_count = int(n_iter)
            logger.info(
                "Model %s iterations: %d (max_iter=%d)",
                model_name,
                iter_count,
                getattr(model_step, "max_iter", -1),
            )
            if getattr(model_step, "max_iter", None) is not None and iter_count >= getattr(model_step, "max_iter"):
                logger.warning(
                    "Model %s reached max_iter without clear convergence; consider --logreg-max-iter",
                    model_name,
                )
        if cv >= 2:
            logger.info("Running %d-fold CV for %s", cv, model_name)
            cv_groups = None
            if groups_series is not None:
                cv_groups = groups_series.iloc[train_idx]
            cv_metrics = perform_cross_validation(
                X_train,
                y_train,
                model_type=model_name,
                preprocessor=preprocessor,
                cv=cv,
                seed=seed,
                sample_weights=sw_train,
                class_weight=class_weight_dict if model_name in (MODEL_MLP, MODEL_FP_OPTIMIZED_MLP) else None,
                groups=cv_groups,
                mlp_params=mlp_params if model_name in (MODEL_MLP, MODEL_FP_OPTIMIZED_MLP) else None,
            )
            metrics[model_name]["cross_validation"] = cv_metrics

        if not dry_run:
            assert artifacts is not None
            model_path = artifacts.run_dir / f"model_{model_name}.joblib"
            dump(pipeline, model_path)
            logger.info("Saved model to %s", model_path)

            _write_split_predictions(
                split_name="train",
                model_name=model_name,
                pipeline=pipeline,
                X_split=X_train,
                y_split=y_train,
                weights_split=sw_train,
            )
            _write_split_predictions(
                split_name="test",
                model_name=model_name,
                pipeline=pipeline,
                X_split=X_test,
                y_split=y_test,
                weights_split=sw_test,
            )
            if X_val is not None and y_val is not None:
                _write_split_predictions(
                    split_name="validation",
                    model_name=model_name,
                    pipeline=pipeline,
                    X_split=X_val,
                    y_split=y_val,
                    weights_split=sw_val,
                )

            cm = np.array(test_metrics["confusion_matrix"]["raw"], dtype=int)

            # Generate display labels based on number of classes
            n_classes = cm.shape[0]
            if n_classes == 2:
                display_labels = ["No Reaction", "Reaction"]
            else:
                display_labels = [f"Class {i}" for i in range(n_classes)]

            fig, ax = plt.subplots(figsize=(5, 5))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=display_labels,
            )
            disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
            ax.set_title(f"{model_name.upper()} Confusion Matrix (Test)")
            fig.tight_layout()
            cm_path = artifacts.run_dir / f"confusion_matrix_{model_name}.png"
            fig.savefig(cm_path, dpi=300)
            plt.close(fig)
            logger.info("Saved confusion matrix plot to %s", cm_path)

    metrics_payload = {"models": metrics}
    if not dry_run:
        assert artifacts is not None
        config.to_json(artifacts.config_path)
        logger.info("Saved config to %s", artifacts.config_path)
        save_metrics(metrics_payload, artifacts.metrics_path)
        logger.info("Saved metrics to %s", artifacts.metrics_path)

    return metrics_payload

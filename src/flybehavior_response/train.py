"""Training routines for flybehavior_response."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import PipelineConfig, compute_class_balance, hash_file, make_run_artifacts
from .evaluate import evaluate_models, perform_cross_validation, save_metrics
from .features import build_column_transformer, validate_features
from .io import LABEL_COLUMN, LABEL_INTENSITY_COLUMN, MERGE_KEYS, load_dataset
from .logging_utils import get_logger
from .modeling import (
    MODEL_LDA,
    MODEL_LOGREG,
    MODEL_MLP,
    build_model_pipeline,
    supported_models,
)
from .weights import expand_samples_by_weight


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_models(
    *,
    data_csv: Path | None,
    labels_csv: Path,
    features: Sequence[str],
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
    group_column: str = "fly",
    group_override: str | None = None,
    test_size: float = 0.2,
) -> Dict[str, Dict[str, object]]:
    logger = get_logger(__name__, verbose=verbose)
    _set_seeds(seed)

    dataset = load_dataset(
        data_csv=data_csv,
        labels_csv=labels_csv,
        logger_name=__name__,
        trace_prefixes=trace_prefixes,
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
    )
    resolved_prefixes = list(dataset.trace_prefixes)
    logger.debug("Trace prefixes resolved to: %s", resolved_prefixes)

    if dataset.feature_columns:
        try:
            selected_features = validate_features(
                features, dataset.feature_columns, logger_name=__name__
            )
        except ValueError:
            logger.warning(
                "Requested features %s not found; falling back to dataset-provided features %s.",
                sorted(features),
                dataset.feature_columns,
            )
            selected_features = list(dataset.feature_columns)
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

    if dataset.sample_weights is None or dataset.label_intensity is None:
        raise ValueError("Training requires labelled data with computed sample weights.")

    sample_weights = dataset.sample_weights
    logger.info(
        "Applying proportional sample weights derived from label intensities (min=%.2f mean=%.2f max=%.2f)",
        float(sample_weights.min()),
        float(sample_weights.mean()),
        float(sample_weights.max()),
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

    data_path = geometry_source if geometry_source is not None else data_csv
    if data_path is None:
        raise ValueError("Either data_csv or geometry_source must be provided for training.")

    file_hashes = {
        "data_csv": hash_file(data_path),
        "labels_csv": hash_file(labels_csv),
    }

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

    if groups_series is not None:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(X, y, groups_series))
        train_groups = set(groups_series.iloc[train_idx])
        test_groups = set(groups_series.iloc[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(
                "Group leakage detected across splits for groups: %s" % sorted(overlap)
            )
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=test_size,
            stratify=y,
            random_state=seed,
        )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    sw_train = sample_weights.iloc[train_idx]
    sw_test = sample_weights.iloc[test_idx]

    logger.info(
        "Split data into training (%d samples, reaction rate=%.2f) and test (%d samples, reaction rate=%.2f)",
        len(y_train),
        float(y_train.mean()) if len(y_train) else 0.0,
        len(y_test),
        float(y_test.mean()) if len(y_test) else 0.0,
    )

    metrics: Dict[str, Dict[str, object]] = {}

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
        data_format="geometry" if geometry_source is not None else "tabular",
        dataset_granularity=dataset.granularity,
        group_column=resolved_group_column,
        geometry_aggregations=list(geom_stats or []),
        geometry_normalization=dataset.normalization,
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
        )
        if model_name == MODEL_LDA:
            X_fit, y_fit = expand_samples_by_weight(X_train, y_train, sw_train)
            pipeline.fit(X_fit, y_fit)
        else:
            fit_kwargs = {}
            if model_name in {MODEL_LOGREG, MODEL_MLP}:
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
        test_metrics = evaluate_models(
            {model_name: pipeline},
            X_test,
            y_test,
            sample_weight=sw_test,
        )[model_name]
        metrics[model_name] = train_metrics
        metrics[model_name]["test"] = test_metrics

        logger.info(
            "Model %s accuracy -> train: %.3f | test: %.3f",
            model_name,
            train_metrics["accuracy"],
            test_metrics["accuracy"],
        )
        logger.info(
            "Model %s F1 (binary) -> train: %.3f | test: %.3f",
            model_name,
            train_metrics["f1_binary"],
            test_metrics["f1_binary"],
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
                groups=cv_groups,
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

            cm = np.array(test_metrics["confusion_matrix"]["raw"], dtype=int)
            fig, ax = plt.subplots(figsize=(5, 5))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["No Reaction", "Reaction"],
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

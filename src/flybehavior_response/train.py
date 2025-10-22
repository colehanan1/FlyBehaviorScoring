"""Training routines for flybehavior_response."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from joblib import load
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold

from .config import PipelineConfig, compute_class_balance, hash_file, make_run_artifacts
from .evaluate import evaluate_models, perform_cross_validation, save_metrics
from .features import build_column_transformer, validate_features
from .io import LABEL_COLUMN, LABEL_INTENSITY_COLUMN, load_and_merge
from .logging_utils import get_logger
from .modeling import (
    MODEL_LDA,
    MODEL_LOGREG,
    MODEL_MLP,
    build_model_pipeline,
    supported_models,
)
from .metrics import detect_fly_column
from .synthetic_fly import (
    SyntheticConfig,
    SyntheticFlyGenerator,
    save_synthetic_artifacts,
)
from .weights import expand_samples_by_weight


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def train_models(
    *,
    data_csv: Path,
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
    synthetic_config: SyntheticConfig | None = None,
) -> Dict[str, Dict[str, object]]:
    logger = get_logger(__name__, verbose=verbose)
    _set_seeds(seed)

    if synthetic_config is None:
        synthetic_config = SyntheticConfig()

    dataset = load_and_merge(
        data_csv,
        labels_csv,
        logger_name=__name__,
        trace_prefixes=trace_prefixes,
    )
    resolved_prefixes = list(dataset.trace_prefixes)
    logger.debug("Trace prefixes resolved to: %s", resolved_prefixes)
    if dataset.feature_columns:
        selected_features = validate_features(
            features, dataset.feature_columns, logger_name=__name__
        )
        logger.debug("Selected features: %s", selected_features)
    else:
        if features:
            logger.warning(
                "Ignoring requested engineered features %s because the dataset does not "
                "include any of the expected columns.",
                sorted(features),
            )
        selected_features = []
        logger.debug("Dataset does not contain engineered features; proceeding with trace-only preprocessing.")
    logger.debug("Trace columns count: %d", len(dataset.trace_columns))

    fly_column = detect_fly_column(dataset.frame)

    provenance_defaults = {
        "is_synthetic": 0,
        "synthetic_fly_id": "",
        "synthetic_trial_id": dataset.frame.get("trial_label", dataset.frame.index.astype(str)).astype(str),
        "parent_fly_id": dataset.frame[fly_column].astype(str),
        "parent_trial_ids": dataset.frame.get("trial_label", dataset.frame.index.astype(str)).astype(str),
        "aug_op": "",
        "seed": np.nan,
        "conflict_flag": False,
        "pred_prob": np.nan,
        "decision": "real",
        "final_label": dataset.frame[LABEL_COLUMN].astype(int),
    }
    for column, default in provenance_defaults.items():
        if column not in dataset.frame.columns:
            dataset.frame[column] = default

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

    intensity_counts = {str(int(k)): int(v) for k, v in dataset.label_intensity.value_counts().sort_index().items()}
    weight_summary = {
        "min": float(sample_weights.min()),
        "mean": float(sample_weights.mean()),
        "max": float(sample_weights.max()),
    }

    logger.info("Trace series prefixes: %s", resolved_prefixes)

    config = PipelineConfig(
        features=list(selected_features),
        n_pcs=n_pcs,
        use_raw_pca=use_raw_pca,
        seed=seed,
        models=list(requested_models),
        trace_column_range=trace_range,
        data_csv=str(data_csv),
        labels_csv=str(labels_csv),
        file_hashes={
            "data_csv": hash_file(data_csv),
            "labels_csv": hash_file(labels_csv),
        },
        class_balance=compute_class_balance(dataset.frame[LABEL_COLUMN].astype(int).tolist()),
        logreg_solver=logreg_solver,
        logreg_max_iter=logreg_max_iter,
        label_intensity_counts=intensity_counts,
        label_weight_summary=weight_summary,
        label_weight_strategy="proportional_intensity",
        trace_series_prefixes=resolved_prefixes,
    )

    drop_columns = [LABEL_COLUMN]
    if LABEL_INTENSITY_COLUMN in dataset.frame.columns:
        drop_columns.append(LABEL_INTENSITY_COLUMN)
    X = dataset.frame.drop(columns=drop_columns)
    y = dataset.frame[LABEL_COLUMN].astype(int)

    groups = dataset.frame[fly_column].astype(str)
    unique_groups = groups.drop_duplicates()
    if unique_groups.empty:
        raise ValueError("GroupKFold requires at least one fly identifier.")
    n_splits = min(5, len(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two unique flies required for grouped split.")

    shuffled_indices = np.arange(len(dataset.frame))
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled_indices)
    X_shuffled = X.iloc[shuffled_indices]
    y_shuffled = y.iloc[shuffled_indices]
    groups_shuffled = groups.iloc[shuffled_indices]

    gkf = GroupKFold(n_splits=n_splits)
    train_idx_shuff, test_idx_shuff = next(gkf.split(X_shuffled, y_shuffled, groups=groups_shuffled))
    train_indices = shuffled_indices[train_idx_shuff]
    test_indices = shuffled_indices[test_idx_shuff]

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    sw_train = sample_weights.iloc[train_indices]
    sw_test = sample_weights.iloc[test_indices]

    feature_columns = X_train.columns

    logger.info(
        "GroupKFold split (n_splits=%d) produced %d training flies and %d test flies.",
        n_splits,
        groups.iloc[train_indices].nunique(),
        groups.iloc[test_indices].nunique(),
    )

    logger.info(
        "Split data into training (%d samples, reaction rate=%.2f) and test (%d samples, reaction rate=%.2f)",
        len(y_train),
        float(y_train.mean()),
        len(y_test),
        float(y_test.mean()),
    )

    synthetic_store: List[pd.DataFrame] = []

    if synthetic_config.use_synthetics:
        logger.info("Generating synthetic flies (ratio=%.2f)", synthetic_config.synthetic_fly_ratio)
        generator = SyntheticFlyGenerator(config=synthetic_config, logger=logger)
        preview_pipeline = None
        if synthetic_config.preview_score_checkpoint is not None:
            try:
                preview_pipeline = load(synthetic_config.preview_score_checkpoint)
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning("Failed to load preview checkpoint %s: %s", synthetic_config.preview_score_checkpoint, exc)
        training_frame = dataset.frame.iloc[train_indices]
        synthetic_result = generator.generate(
            training_frame,
            trace_columns=dataset.trace_columns,
            fly_column=fly_column,
            preview_pipeline=preview_pipeline,
        )

        conflict_total = int(synthetic_result.manifest.get("conflict_flag", pd.Series(dtype=bool)).sum()) if not synthetic_result.manifest.empty else 0
        logger.info("Synthetic conflicts flagged: %d", conflict_total)

        if not dry_run:
            csv_path, manifest_path = save_synthetic_artifacts(
                synthetic_result,
                config=synthetic_config,
            )
            logger.info("Saved synthetic CSV to %s and manifest to %s", csv_path, manifest_path)
            if synthetic_result.manifest.empty:
                logger.info("Synthetic manifest is empty; no trials were generated.")
            if synthetic_result.preview_image is not None:
                logger.info("Saved synthetic preview grid to %s", synthetic_result.preview_image)

        if synthetic_result.kept_indices:
            kept_df = synthetic_result.dataframe.iloc[synthetic_result.kept_indices].copy()
            kept_df.index = kept_df["synthetic_trial_id"].astype(str)
            kept_df[LABEL_COLUMN] = kept_df["final_label"].astype(int)
            if LABEL_INTENSITY_COLUMN in kept_df.columns:
                intensities = kept_df[LABEL_INTENSITY_COLUMN].astype(float)
                intensities.loc[kept_df[LABEL_COLUMN] == 0] = 0.0
                intensities.loc[(kept_df[LABEL_COLUMN] == 1) & (intensities <= 0)] = 1.0
                kept_df[LABEL_INTENSITY_COLUMN] = intensities
            else:
                kept_df[LABEL_INTENSITY_COLUMN] = kept_df[LABEL_COLUMN].astype(float)

            kept_aligned = kept_df.reindex(columns=dataset.frame.columns, fill_value=np.nan)
            synthetic_store.append(kept_aligned)

            synth_features = kept_aligned.drop(columns=drop_columns, errors="ignore")
            if not synth_features.empty:
                non_empty_feature_cols = synth_features.columns[
                    synth_features.notna().any(axis=0)
                ]
                synth_features = synth_features.loc[:, non_empty_feature_cols]
            synth_labels = kept_aligned[LABEL_COLUMN].astype(int)
            synth_weights = pd.Series(1.0, index=kept_aligned.index, dtype=float)
            positive_mask = kept_aligned[LABEL_INTENSITY_COLUMN] > 0
            synth_weights.loc[positive_mask] = kept_aligned.loc[
                positive_mask, LABEL_INTENSITY_COLUMN
            ].astype(float)

            X_train = pd.concat([X_train, synth_features], axis=0)
            X_train = X_train.reindex(columns=feature_columns)
            y_train = pd.concat([y_train, synth_labels], axis=0)
            sw_train = pd.concat([sw_train, synth_weights], axis=0)

            kept_counts = synth_labels.value_counts().to_dict()
            logger.info(
                "Synthetic trials proposed=%d kept=%d | kept class distribution: %s",
                len(synthetic_result.dataframe),
                len(kept_df),
                kept_counts,
            )
        else:
            logger.info("No synthetic trials were retained after gating.")

    metrics: Dict[str, Dict[str, object]] = {}

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
        augmented_frame = dataset.frame
        if synthetic_store:
            synthetic_aug = pd.concat(synthetic_store, axis=0)
            synthetic_aug = synthetic_aug.reindex(columns=dataset.frame.columns)
            # Drop columns that are entirely NA to avoid pandas concat dtype warnings while
            # preserving the real dataset schema. The dropped columns will still exist after
            # concatenation because they are present in ``dataset.frame``.
            non_empty_cols = synthetic_aug.columns[synthetic_aug.notna().any(axis=0)]
            synthetic_aug = synthetic_aug.loc[:, non_empty_cols]
            augmented_frame = pd.concat([dataset.frame, synthetic_aug], axis=0)
            augmented_frame = augmented_frame.reindex(columns=dataset.frame.columns)
        try:
            original_rows = augmented_frame.loc[y_split.index].copy()
        except KeyError as exc:
            missing = [idx for idx in y_split.index if idx not in augmented_frame.index]
            raise KeyError(
                f"Missing rows for prediction export: {missing[:5]} (total missing={len(missing)})"
            ) from exc
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
            try:
                cv_metrics = perform_cross_validation(
                    X_train,
                    y_train,
                    model_type=model_name,
                    preprocessor=preprocessor,
                    cv=cv,
                    seed=seed,
                    sample_weights=sw_train,
                )
            except ValueError as exc:
                logger.warning("Skipping cross-validation for %s due to: %s", model_name, exc)
                metrics[model_name]["cross_validation"] = None
            else:
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

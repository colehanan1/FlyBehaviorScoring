"""Training routines for flybehavior_response."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

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
) -> Dict[str, Dict[str, object]]:
    logger = get_logger(__name__, verbose=verbose)
    _set_seeds(seed)

    dataset = load_and_merge(data_csv, labels_csv, logger_name=__name__)
    selected_features = validate_features(features, dataset.feature_columns, logger_name=__name__)
    logger.debug("Selected features: %s", selected_features)
    logger.debug("Trace columns count: %d", len(dataset.trace_columns))

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

    trace_indices = [int(col.split("_")[-1]) for col in dataset.trace_columns]
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
    )

    drop_columns = [LABEL_COLUMN]
    if LABEL_INTENSITY_COLUMN in dataset.frame.columns:
        drop_columns.append(LABEL_INTENSITY_COLUMN)
    X = dataset.frame.drop(columns=drop_columns)
    y = dataset.frame[LABEL_COLUMN].astype(int)

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X,
        y,
        sample_weights,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )

    logger.info(
        "Split data into training (%d samples, reaction rate=%.2f) and test (%d samples, reaction rate=%.2f)",
        len(y_train),
        float(y_train.mean()),
        len(y_test),
        float(y_test.mean()),
    )

    metrics: Dict[str, Dict[str, object]] = {}

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
            cv_metrics = perform_cross_validation(
                X_train,
                y_train,
                model_type=model_name,
                preprocessor=preprocessor,
                cv=cv,
                seed=seed,
                sample_weights=sw_train,
            )
            metrics[model_name]["cross_validation"] = cv_metrics

        if not dry_run:
            assert artifacts is not None
            model_path = artifacts.run_dir / f"model_{model_name}.joblib"
            dump(pipeline, model_path)
            logger.info("Saved model to %s", model_path)

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

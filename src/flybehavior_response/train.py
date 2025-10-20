"""Training routines for flybehavior_response."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from .synthetic import (
    SyntheticConfig,
    make_synthetics,
    preview_and_gate,
    score_synthetics,
)
from .weights import expand_samples_by_weight


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _sorted_trace_columns(
    trace_columns: Sequence[str],
    trace_prefixes: Sequence[str],
) -> List[List[str]]:
    ordered_groups: List[List[str]] = []
    used: set[str] = set()
    prefixes = list(trace_prefixes) if trace_prefixes else []
    if not prefixes:
        prefixes = []
        if trace_columns:
            common_prefix = "".join(ch for ch in trace_columns[0] if not ch.isdigit())
            prefixes.append(common_prefix)
    for prefix in prefixes:
        group = [col for col in trace_columns if col.startswith(prefix)]
        if not group:
            continue
        group.sort(key=lambda name: int("".join(filter(str.isdigit, name)) or 0))
        ordered_groups.append(group)
        used.update(group)
    remaining = [col for col in trace_columns if col not in used]
    if remaining:
        remaining.sort(key=lambda name: int("".join(filter(str.isdigit, name)) or 0))
        ordered_groups.append(remaining)
    return ordered_groups


def _build_trace_tensor(
    df: pd.DataFrame,
    trace_columns: Sequence[str],
    trace_prefixes: Sequence[str],
) -> tuple[np.ndarray, List[List[str]]]:
    if not trace_columns:
        raise ValueError("Trace columns are required for synthetic generation.")
    column_groups = _sorted_trace_columns(trace_columns, trace_prefixes)
    lengths = {len(group) for group in column_groups if group}
    if len(lengths) != 1:
        raise ValueError("Trace columns must share a consistent time dimension.")
    if not lengths:
        raise ValueError("Unable to infer trace column grouping.")
    time_steps = lengths.pop()
    order = [col for group in column_groups for col in group]
    values = df.loc[:, order].to_numpy(dtype=np.float32)
    tensor = values.reshape(len(df), len(column_groups), time_steps)
    tensor = np.transpose(tensor, (0, 2, 1))
    return tensor, column_groups


def _series_to_row(series: np.ndarray, column_groups: Sequence[Sequence[str]]) -> dict[str, float]:
    row: dict[str, float] = {}
    for channel_idx, columns in enumerate(column_groups):
        values = series[:, channel_idx]
        for column, value in zip(columns, values):
            row[column] = float(value)
    return row


def _compose_identifier(row: pd.Series) -> str:
    dataset_name = str(row.get("dataset", "dataset"))
    fly = str(row.get("fly", "fly"))
    fly_number = row.get("fly_number")
    fly_number_text = f"|flynum={fly_number}" if pd.notna(fly_number) else ""
    trial = str(row.get("trial_label", "trial"))
    return f"{dataset_name}|fly={fly}{fly_number_text}|trial={trial}"


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
    use_synthetics: bool = False,
    synthetic_ratio: float = 0.5,
    synthetic_ops: Sequence[str] | None = None,
    mixup_alpha: float = 0.2,
    preview_synthetics: int = 12,
    preview_score_checkpoint: Path | None = None,
    auto_filter_threshold: float = 0.0,
    save_synthetics_dir: Path | None = None,
) -> Dict[str, Dict[str, object]]:
    logger = get_logger(__name__, verbose=verbose)
    _set_seeds(seed)

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

    if use_synthetics:
        ops_to_use = tuple(synthetic_ops) if synthetic_ops is not None else SyntheticConfig().synthetic_ops
        save_dir = Path(save_synthetics_dir) if save_synthetics_dir is not None else Path("./synthetics")
        syn_config = SyntheticConfig(
            synthetic_ratio=synthetic_ratio,
            synthetic_ops=ops_to_use,
            mixup_alpha=mixup_alpha,
            preview_synthetics=preview_synthetics,
            preview_score_checkpoint=preview_score_checkpoint,
            auto_filter_threshold=auto_filter_threshold,
            save_synthetics_dir=save_dir,
            seed=seed,
        )
        logger.info(
            "Synthetic generation enabled: ratio=%.3f ops=%s preview=%d",
            synthetic_ratio,
            ",".join(ops_to_use),
            preview_synthetics,
        )
        trace_tensor, column_groups = _build_trace_tensor(X_train, dataset.trace_columns, dataset.trace_prefixes)
        train_meta = dataset.frame.loc[X_train.index]
        ids_train = [_compose_identifier(row) for _, row in train_meta.iterrows()]
        class_names = sorted({int(val) for val in y_train.to_numpy()})
        X_syn_ts, y_syn, syn_parents, syn_ops, syn_seeds = make_synthetics(
            trace_tensor,
            y_train.to_numpy(),
            ids_train,
            class_names,
            syn_config,
        )
        if len(X_syn_ts):
            non_trace_columns = [col for col in X_train.columns if col not in dataset.trace_columns]
            id_to_index = {syn_id: idx for syn_id, idx in zip(ids_train, X_train.index)}
            synthetic_index = [f"synthetic_{i:05d}" for i in range(len(X_syn_ts))]
            synthetic_rows: List[dict[str, float]] = []
            synthetic_weights: List[float] = []
            for idx, series in enumerate(X_syn_ts):
                parent_id = syn_parents[idx][0]
                parent_index = id_to_index[parent_id]
                parent_row = X_train.loc[parent_index]
                row_dict = {col: float(parent_row[col]) for col in non_trace_columns}
                row_dict.update(_series_to_row(series, column_groups))
                synthetic_rows.append(row_dict)
                if parent_index in sw_train.index:
                    synthetic_weights.append(float(sw_train.loc[parent_index]))
                else:
                    synthetic_weights.append(1.0)
            syn_df = pd.DataFrame(synthetic_rows, index=synthetic_index)
            syn_df = syn_df.reindex(columns=X_train.columns)
            syn_weights = pd.Series(synthetic_weights, index=synthetic_index, dtype=float)
            syn_weights.name = sw_train.name
            syn_labels = pd.Series(y_syn, index=synthetic_index, dtype=int)
            syn_labels.name = y_train.name
            try:
                probs = score_synthetics(syn_config.preview_score_checkpoint, syn_df)
            except FileNotFoundError:
                raise
            except Exception as exc:  # pragma: no cover - resilience
                logger.exception("Failed to score synthetics; proceeding without probabilities: %s", exc)
                probs = None
            mask_keep, final_labels, _ = preview_and_gate(
                syn_ids=synthetic_index,
                X_syn=X_syn_ts,
                y_syn=y_syn,
                parents=syn_parents,
                ops=syn_ops,
                seeds=syn_seeds,
                class_names=class_names,
                config=syn_config,
                probs=probs,
            )
            y_syn_final = np.asarray(final_labels, dtype=int)
            syn_labels[:] = y_syn_final
            kept_idx = np.where(mask_keep)[0]
            dropped = int(len(X_syn_ts) - len(kept_idx))
            relabel_zero = int(np.sum((mask_keep) & (y_syn_final == 0) & (y_syn != 0)))
            relabel_one = int(np.sum((mask_keep) & (y_syn_final == 1) & (y_syn != 1)))
            logger.info(
                "Synthetic gating results -> proposed=%d kept=%d dropped=%d relabel0=%d relabel1=%d",
                len(X_syn_ts),
                len(kept_idx),
                dropped,
                relabel_zero,
                relabel_one,
            )
            if kept_idx.size:
                kept_df = syn_df.iloc[kept_idx]
                kept_labels = syn_labels.iloc[kept_idx]
                kept_weights = syn_weights.iloc[kept_idx]
                approved_npz = syn_config.save_synthetics_dir / "X_syn_approved.npz"
                approved_npy = syn_config.save_synthetics_dir / "y_syn_approved.npy"
                np.savez(approved_npz, X=X_syn_ts[kept_idx].astype(np.float32))
                np.save(approved_npy, kept_labels.to_numpy(dtype=int))
                logger.info("Saved approved synthetic traces to %s", approved_npz.resolve())
                logger.info("Saved approved synthetic labels to %s", approved_npy.resolve())
                X_train = pd.concat([X_train, kept_df])
                y_train = pd.concat([y_train, kept_labels])
                sw_train = pd.concat([sw_train, kept_weights])
            else:
                logger.info("No synthetic samples approved after gating.")
        else:
            logger.info("No synthetic samples generated; proceeding without augmentation.")

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

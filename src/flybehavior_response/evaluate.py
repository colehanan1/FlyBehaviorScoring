"""Evaluation utilities for flybehavior_response."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .modeling import MODEL_LDA, MODEL_LOGREG, build_model_pipeline
from .weights import expand_samples_by_weight


def _serialize_confusion(matrix: np.ndarray) -> List[List[float]]:
    return matrix.tolist()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    proba: np.ndarray | None,
    model_type: str,
    sample_weight: np.ndarray | None = None,
) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    metrics["f1_binary"] = float(f1_score(y_true, y_pred, average="binary"))
    raw_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    norm_cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize="true")
    norm_cm = np.nan_to_num(norm_cm)
    metrics["confusion_matrix"] = {
        "raw": _serialize_confusion(raw_cm),
        "normalized": _serialize_confusion(norm_cm),
    }
    if model_type == MODEL_LOGREG and proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
    else:
        metrics["roc_auc"] = None

    if sample_weight is not None:
        weighted_metrics: Dict[str, object] = {}
        weighted_metrics["accuracy"] = float(
            accuracy_score(y_true, y_pred, sample_weight=sample_weight)
        )
        weighted_metrics["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", sample_weight=sample_weight)
        )
        weighted_metrics["f1_binary"] = float(
            f1_score(y_true, y_pred, average="binary", sample_weight=sample_weight)
        )
        weighted_raw = confusion_matrix(
            y_true, y_pred, labels=[0, 1], sample_weight=sample_weight
        )
        weighted_norm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
            normalize="true",
            sample_weight=sample_weight,
        )
        weighted_norm = np.nan_to_num(weighted_norm)
        weighted_metrics["confusion_matrix"] = {
            "raw": _serialize_confusion(weighted_raw),
            "normalized": _serialize_confusion(weighted_norm),
        }
        if model_type == MODEL_LOGREG and proba is not None:
            weighted_metrics["roc_auc"] = float(
                roc_auc_score(y_true, proba[:, 1], sample_weight=sample_weight)
            )
        else:
            weighted_metrics["roc_auc"] = None
        metrics["weighted"] = weighted_metrics
    return metrics


def evaluate_pipeline(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str,
    *,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> Dict[str, object]:
    y_pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    weight_array = (
        sample_weight.to_numpy() if hasattr(sample_weight, "to_numpy") else sample_weight
    )
    return compute_metrics(
        y_true=y.to_numpy(),
        y_pred=y_pred,
        proba=proba,
        model_type=model_type,
        sample_weight=weight_array,
    )


def perform_cross_validation(
    data: pd.DataFrame,
    labels: pd.Series,
    *,
    model_type: str,
    preprocessor,
    cv: int,
    seed: int,
    sample_weights: pd.Series | None = None,
) -> Dict[str, float | List[List[float]] | None | Dict[str, List[List[float]]]]:
    if cv <= 1:
        raise ValueError("Cross-validation requires cv >= 2")
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    aggregate_raw = np.zeros((2, 2), dtype=float)
    metrics_accum: Dict[str, List[float]] = {"accuracy": [], "f1_macro": [], "f1_binary": [], "roc_auc": []}
    weighted_accum: Dict[str, List[float]] | None = None
    aggregate_weighted_raw: np.ndarray | None = None
    if sample_weights is not None:
        weighted_accum = {"accuracy": [], "f1_macro": [], "f1_binary": [], "roc_auc": []}
        aggregate_weighted_raw = np.zeros((2, 2), dtype=float)
    for train_idx, test_idx in splitter.split(data, labels):
        model = build_model_pipeline(preprocessor, model_type=model_type, seed=seed)
        if sample_weights is not None and model_type == MODEL_LDA:
            train_data, train_labels = expand_samples_by_weight(
                data.iloc[train_idx], labels.iloc[train_idx], sample_weights.iloc[train_idx]
            )
            model.fit(train_data, train_labels)
        else:
            fit_kwargs = {}
            if sample_weights is not None:
                fit_kwargs["model__sample_weight"] = sample_weights.iloc[train_idx].to_numpy()
            model.fit(data.iloc[train_idx], labels.iloc[train_idx], **fit_kwargs)
        fold_sample_weight = None
        if sample_weights is not None:
            fold_sample_weight = sample_weights.iloc[test_idx]
        fold_metrics = evaluate_pipeline(
            model,
            data.iloc[test_idx],
            labels.iloc[test_idx],
            model_type,
            sample_weight=fold_sample_weight,
        )
        aggregate_raw += np.array(fold_metrics["confusion_matrix"]["raw"], dtype=float)
        for key in ["accuracy", "f1_macro", "f1_binary"]:
            metrics_accum[key].append(float(fold_metrics[key]))
        if model_type == MODEL_LOGREG and fold_metrics.get("roc_auc") is not None:
            metrics_accum["roc_auc"].append(float(fold_metrics["roc_auc"]))
        if weighted_accum is not None and "weighted" in fold_metrics:
            weighted = fold_metrics["weighted"]
            aggregate_weighted_raw += np.array(weighted["confusion_matrix"]["raw"], dtype=float)
            for key in ["accuracy", "f1_macro", "f1_binary"]:
                weighted_accum[key].append(float(weighted[key]))
            if model_type == MODEL_LOGREG and weighted.get("roc_auc") is not None:
                weighted_accum["roc_auc"].append(float(weighted["roc_auc"]))
    averaged = {key: float(np.mean(values)) if values else None for key, values in metrics_accum.items()}
    normalized = np.divide(
        aggregate_raw,
        aggregate_raw.sum(axis=1, keepdims=True),
        out=np.zeros_like(aggregate_raw),
        where=aggregate_raw.sum(axis=1, keepdims=True) != 0,
    )
    averaged["confusion_matrix"] = {
        "raw": _serialize_confusion(aggregate_raw),
        "normalized": _serialize_confusion(normalized),
    }
    if weighted_accum is not None and aggregate_weighted_raw is not None:
        weighted_avg = {
            key: float(np.mean(values)) if values else None for key, values in weighted_accum.items()
        }
        weighted_norm = np.divide(
            aggregate_weighted_raw,
            aggregate_weighted_raw.sum(axis=1, keepdims=True),
            out=np.zeros_like(aggregate_weighted_raw),
            where=aggregate_weighted_raw.sum(axis=1, keepdims=True) != 0,
        )
        weighted_avg["confusion_matrix"] = {
            "raw": _serialize_confusion(aggregate_weighted_raw),
            "normalized": _serialize_confusion(weighted_norm),
        }
        averaged["weighted"] = weighted_avg
    return averaged


def load_pipeline(path: Path):
    return load(path)


def save_metrics(metrics: Mapping[str, object], path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def evaluate_models(
    models: Mapping[str, object],
    data: pd.DataFrame,
    labels: pd.Series,
    *,
    sample_weight: pd.Series | np.ndarray | None = None,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for name, model in models.items():
        results[name] = evaluate_pipeline(
            model,
            data,
            labels,
            model_type=name,
            sample_weight=sample_weight,
        )
    return results

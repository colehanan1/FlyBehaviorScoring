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

from .modeling import MODEL_LOGREG, build_model_pipeline


def _serialize_confusion(matrix: np.ndarray) -> List[List[float]]:
    return matrix.tolist()


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    proba: np.ndarray | None,
    model_type: str,
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
    return metrics


def evaluate_pipeline(model, X: pd.DataFrame, y: pd.Series, model_type: str) -> Dict[str, object]:
    y_pred = model.predict(X)
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    return compute_metrics(y_true=y.to_numpy(), y_pred=y_pred, proba=proba, model_type=model_type)


def perform_cross_validation(
    data: pd.DataFrame,
    labels: pd.Series,
    *,
    model_type: str,
    preprocessor,
    cv: int,
    seed: int,
) -> Dict[str, float | List[List[float]] | None | Dict[str, List[List[float]]]]:
    if cv <= 1:
        raise ValueError("Cross-validation requires cv >= 2")
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    aggregate_raw = np.zeros((2, 2), dtype=float)
    metrics_accum: Dict[str, List[float]] = {"accuracy": [], "f1_macro": [], "f1_binary": [], "roc_auc": []}
    for train_idx, test_idx in splitter.split(data, labels):
        model = build_model_pipeline(preprocessor, model_type=model_type, seed=seed)
        model.fit(data.iloc[train_idx], labels.iloc[train_idx])
        fold_metrics = evaluate_pipeline(model, data.iloc[test_idx], labels.iloc[test_idx], model_type)
        aggregate_raw += np.array(fold_metrics["confusion_matrix"]["raw"], dtype=float)
        for key in ["accuracy", "f1_macro", "f1_binary"]:
            metrics_accum[key].append(float(fold_metrics[key]))
        if model_type == MODEL_LOGREG and fold_metrics.get("roc_auc") is not None:
            metrics_accum["roc_auc"].append(float(fold_metrics["roc_auc"]))
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
    return averaged


def load_pipeline(path: Path):
    return load(path)


def save_metrics(metrics: Mapping[str, object], path: Path) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def evaluate_models(models: Mapping[str, object], data: pd.DataFrame, labels: pd.Series) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for name, model in models.items():
        results[name] = evaluate_pipeline(model, data, labels, model_type=name)
    return results

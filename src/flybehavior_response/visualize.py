"""Visualization utilities for flybehavior_response."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .io import LABEL_COLUMN, LABEL_INTENSITY_COLUMN, load_and_merge
from .logging_utils import get_logger
from .modeling import MODEL_LDA, MODEL_LOGREG


def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    mean_diff = group_a.mean() - group_b.mean()
    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)
    pooled = np.sqrt(((len(group_a) - 1) * var_a + (len(group_b) - 1) * var_b) / (len(group_a) + len(group_b) - 2))
    return float(mean_diff / pooled) if pooled != 0 else 0.0


def plot_pc_scatter(data: pd.DataFrame, trace_columns, labels: pd.Series, path: Path, seed: int) -> Path:
    scaler = StandardScaler()
    traces = scaler.fit_transform(data[trace_columns])
    pca = PCA(n_components=2, random_state=seed)
    pcs = pca.fit_transform(traces)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap="viridis", alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Trace PCA Scatter")
    ev = pca.explained_variance_ratio_ * 100
    ax.annotate(f"Explained variance: PC1 {ev[0]:.1f}%, PC2 {ev[1]:.1f}%", xy=(0.05, 0.95), xycoords="axes fraction", va="top")
    legend = ax.legend(*scatter.legend_elements(), title="Label")
    ax.add_artist(legend)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_lda_scores(pipeline, data: pd.DataFrame, labels: pd.Series, path: Path) -> Path:
    preprocess = pipeline.named_steps["preprocess"]
    lda = pipeline.named_steps["model"]
    transformed = preprocess.transform(data)
    scores = lda.transform(transformed).ravel()
    fig, ax = plt.subplots(figsize=(8, 4))
    classes = sorted(labels.unique())
    for cls in classes:
        cls_scores = scores[labels == cls]
        ax.hist(cls_scores, bins=20, alpha=0.6, label=f"Class {cls}")
    ax.set_xlabel("LDA Score")
    ax.set_ylabel("Frequency")
    ax.set_title("LDA Score Distribution")
    means = [scores[labels == cls].mean() for cls in classes]
    if len(classes) == 2:
        d = _cohens_d(scores[labels == classes[0]], scores[labels == classes[1]])
        ax.annotate(f"Means: {means[0]:.2f}, {means[1]:.2f} | d={d:.2f}", xy=(0.5, 0.95), xycoords="axes fraction", ha="center", va="top")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_roc_curve(pipeline, data: pd.DataFrame, labels: pd.Series, path: Path) -> Path:
    from sklearn.metrics import roc_curve, auc

    proba = pipeline.predict_proba(data)[:, 1]
    fpr, tpr, _ = roc_curve(labels, proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Logistic Regression ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_validation_curve(
    *,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    param_range: Sequence,
    param_name: str,
    model_name: str,
    scoring_label: str = "Accuracy",
    path: Path,
    log_scale: bool = False,
) -> Path:
    """Plot training vs CV score as a function of a model complexity parameter."""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Build x-axis values and labels
    x_labels = [str(v) if v is not None else "None" for v in param_range]
    x_vals = np.arange(len(param_range))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, train_mean, "o-", color="#1f77b4", label="Training Score")
    ax.fill_between(x_vals, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#1f77b4")
    ax.plot(x_vals, test_mean, "o-", color="#ff7f0e", label="Cross-Validation Score")
    ax.fill_between(x_vals, test_mean - test_std, test_mean + test_std, alpha=0.15, color="#ff7f0e")

    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel(param_name)
    ax.set_ylabel(scoring_label)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{model_name.upper()} \u2014 Validation Curve ({param_name})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_regularization_path(
    *,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    param_range: np.ndarray,
    param_name: str,
    model_name: str,
    scoring_label: str = "Accuracy",
    path: Path,
    invert_log: bool = False,
) -> Path:
    """Plot training vs CV score as a function of regularization on ln(lambda) scale."""
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # For parameters like C (inverse regularization), invert so increasing x = stronger reg
    if invert_log:
        x_vals = -np.log(param_range)
        x_label = f"ln(1/{param_name})"
    else:
        x_vals = np.log(param_range)
        x_label = f"ln({param_name})"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, train_mean, "o-", color="#1f77b4", markersize=3, label="Training Score")
    ax.fill_between(x_vals, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#1f77b4")
    ax.plot(x_vals, test_mean, "o-", color="#ff7f0e", markersize=3, label="Cross-Validation Score")
    ax.fill_between(x_vals, test_mean - test_std, test_mean + test_std, alpha=0.15, color="#ff7f0e")

    ax.set_xlabel(x_label)
    ax.set_ylabel(scoring_label)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{model_name.upper()} \u2014 Regularization Path ({param_name})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def generate_visuals(
    *,
    data_csv: Path,
    labels_csv: Path,
    run_dir: Path,
    seed: int,
    output_dir: Path,
    verbose: bool,
    trace_prefixes: Sequence[str] | None = None,
    include_traces: bool = True,
) -> Dict[str, Path]:
    logger = get_logger(__name__, verbose=verbose)
    dataset = load_and_merge(
        data_csv,
        labels_csv,
        logger_name=__name__,
        trace_prefixes=trace_prefixes,
        include_trace_columns=include_traces,
    )
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}. Run training before visualization.")
    config = PipelineConfig.from_json(config_path)
    plots: Dict[str, Path] = {}
    labels = dataset.frame[LABEL_COLUMN].astype(int)
    if dataset.trace_columns:
        plots["pc_scatter"] = plot_pc_scatter(
            dataset.frame,
            dataset.trace_columns,
            labels,
            output_dir / "pc_scatter.png",
            seed,
        )
    else:
        logger.info("Skipping PC scatter plot because raw trace columns are unavailable.")

    lda_path = run_dir / "model_lda.joblib"
    if lda_path.exists():
        from joblib import load

        lda_model = load(lda_path)
        drop_cols = [LABEL_COLUMN]
        if LABEL_INTENSITY_COLUMN in dataset.frame.columns:
            drop_cols.append(LABEL_INTENSITY_COLUMN)
        plots["lda_scores"] = plot_lda_scores(
            lda_model,
            dataset.frame.drop(columns=drop_cols),
            labels,
            output_dir / "lda_scores.png",
        )
    else:
        logger.info("LDA model not found at %s; skipping LDA score plot.", lda_path)

    logreg_path = run_dir / "model_logreg.joblib"
    if logreg_path.exists():
        from joblib import load

        logreg_model = load(logreg_path)
        drop_cols = [LABEL_COLUMN]
        if LABEL_INTENSITY_COLUMN in dataset.frame.columns:
            drop_cols.append(LABEL_INTENSITY_COLUMN)
        plots["roc_curve"] = plot_roc_curve(
            logreg_model,
            dataset.frame.drop(columns=drop_cols),
            labels,
            output_dir / "roc.png",
        )
    else:
        logger.info("Logistic Regression model not found at %s; skipping ROC plot.", logreg_path)

    return plots

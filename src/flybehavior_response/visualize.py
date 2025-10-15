"""Visualization utilities for flybehavior_response."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

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


def generate_visuals(
    *,
    data_csv: Path,
    labels_csv: Path,
    run_dir: Path,
    seed: int,
    output_dir: Path,
    verbose: bool,
) -> Dict[str, Path]:
    logger = get_logger(__name__, verbose=verbose)
    dataset = load_and_merge(data_csv, labels_csv, logger_name=__name__)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}. Run training before visualization.")
    config = PipelineConfig.from_json(config_path)
    plots: Dict[str, Path] = {}
    labels = dataset.frame[LABEL_COLUMN].astype(int)
    plots["pc_scatter"] = plot_pc_scatter(dataset.frame, dataset.trace_columns, labels, output_dir / "pc_scatter.png", seed)

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

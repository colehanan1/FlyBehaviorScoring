"""Model training, evaluation, and artifact generation for fly behavior."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler

from .features_from_envelope import FOLD_PCA_COMPONENTS, RANDOM_SEED, prepare_pca_matrix

logger = logging.getLogger(__name__)

CLUSTER_FEATURES = [
    "odor_auc_above_thr",
    "odor_pct_above_thr",
    "latency_to_thr",
    "rise_slope",
]

ODOR_ON_COLUMN = 1245
ODOR_OFF_COLUMN = 2445


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_annotated_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    totals = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        perc = np.divide(cm, totals, where=totals != 0) * 100
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            pct = perc[i, j]
            label = f"{value}\n({pct:.1f}%)" if not np.isnan(pct) else f"{value}\n(-)"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=12,
            )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _categorize_feature(name: str) -> str:
    if name.startswith(
        (
            "baseline_",
            "odor_",
            "latency_",
            "rise_",
            "odor_window_len_s",
        )
    ):
        return "engineered"
    if name.startswith("cluster_"):
        return "cluster_one_hot"
    if name.startswith("pc_"):
        return "cluster_pc"
    if name.startswith("fold_pca_"):
        return "odor_pca"
    return "other"


def _log_feature_breakdown(feature_names: Sequence[str], context: str) -> None:
    if not feature_names:
        logger.warning("%s produced an empty feature set", context)
        return
    buckets: Dict[str, int] = {}
    for name in feature_names:
        bucket = _categorize_feature(name)
        buckets[bucket] = buckets.get(bucket, 0) + 1
    ordering = [
        "engineered",
        "odor_pca",
        "cluster_one_hot",
        "cluster_pc",
        "other",
    ]
    summary = ", ".join(
        f"{bucket}={buckets[bucket]}" for bucket in ordering if bucket in buckets
    )
    logger.info("%s contains %d features (%s)", context, len(feature_names), summary)


def _prepare_engineered_frame(features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    drop_cols = {
        "dataset",
        "fly",
        "trial_type",
        "trial_label",
        "trial_key",
        "fps",
        "y_bin",
        "y_reg",
        "odor_window_vector",
    }
    drop_cols.update({c for c in features.columns if c.startswith("global_pca_")})
    drop_cols.update({c for c in features.columns if c.startswith("dir_val_")})
    candidate = features.drop(columns=[c for c in drop_cols if c in features.columns])
    numeric_cols = candidate.select_dtypes(include=[np.number]).columns.tolist()
    _log_feature_breakdown(numeric_cols, "Base feature frame")
    X = candidate[numeric_cols].copy()
    return X, numeric_cols


def _extract_raw_trace_matrix(features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    raw_cols = [f"dir_val_{idx}" for idx in range(ODOR_ON_COLUMN, ODOR_OFF_COLUMN + 1)]
    available = [col for col in raw_cols if col in features.columns]
    missing = sorted(set(raw_cols) - set(available))
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            "Raw trace columns missing for odor window; first missing: %s" % sample
        )
    logger.info(
        "Raw trace feature window contains %d columns (dir_val_%d..dir_val_%d)",
        len(available),
        ODOR_ON_COLUMN,
        ODOR_OFF_COLUMN,
    )
    X = features[available].astype(np.float32)
    return X, available


def _fill_latency_inplace(
    X: pd.DataFrame, latency_source: pd.Series, window_len: pd.Series
) -> None:
    if "latency_to_thr" not in X.columns:
        return
    latency_vals = latency_source.reindex(X.index).to_numpy().astype(float)
    inf_mask = np.isinf(latency_vals)
    if inf_mask.any():
        replacement = window_len.reindex(X.index).to_numpy().astype(float) + 1.0
        latency_vals[inf_mask] = replacement[inf_mask]
    X.loc[:, "latency_to_thr"] = latency_vals


def _stack_vectors(vectors: Sequence[np.ndarray], target_len: int) -> np.ndarray:
    return prepare_pca_matrix(vectors, target_len)


def _fit_fold_pca(train_vectors: Sequence[np.ndarray], n_components: int) -> Tuple[PCA, int]:
    lengths = [len(v) for v in train_vectors if len(v) > 0]
    if not lengths:
        raise ValueError("No non-empty odor vectors available for PCA")
    target_len = min(lengths)
    matrix = _stack_vectors(train_vectors, target_len)
    n_components = min(n_components, matrix.shape[0], matrix.shape[1])
    if n_components <= 0:
        raise ValueError("Not enough samples/features for PCA")
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(matrix)
    return pca, target_len


def _transform_pca(pca: PCA, vectors: Sequence[np.ndarray], target_len: int) -> np.ndarray:
    matrix = _stack_vectors(vectors, target_len)
    return pca.transform(matrix)


def _impute_with_median(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()
    for frame in (X_train, X_test):
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    medians = X_train.median()
    X_train.fillna(medians, inplace=True)
    X_test.fillna(medians, inplace=True)
    return X_train, X_test


def run_cross_validation(
    features: pd.DataFrame,
    y_bin: pd.Series,
    y_reg: pd.Series,
    output_dir: Path,
) -> Dict[str, object]:
    _ensure_output_dir(output_dir)
    X_engineered, engineered_names = _prepare_engineered_frame(features)
    X_raw, raw_names = _extract_raw_trace_matrix(features)
    groups = features["fly"].astype(str)
    logo = LeaveOneGroupOut()

    n_samples = len(features)
    oof_bin = np.zeros(n_samples, dtype=int)
    oof_reg = np.full(n_samples, np.nan)
    oof_inf = np.zeros(n_samples, dtype=float)
    oof_proba = np.zeros(n_samples, dtype=float)
    fold_assign = np.empty(n_samples, dtype=object)

    pca_components_per_fold: Dict[str, List[str]] = {}
    feature_names_extended: List[str] = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X_raw, y_bin, groups)):
        train_features = features.iloc[train_idx]
        test_features = features.iloc[test_idx]
        train_vectors = train_features["odor_window_vector"].tolist()
        test_vectors = test_features["odor_window_vector"].tolist()

        try:
            pca, target_len = _fit_fold_pca(train_vectors, FOLD_PCA_COMPONENTS)
            train_pca = _transform_pca(pca, train_vectors, target_len)
            test_pca = _transform_pca(pca, test_vectors, target_len)
            pca_cols = [f"fold_pca_{i+1}" for i in range(train_pca.shape[1])]
        except ValueError as exc:
            logger.warning("Skipping PCA in fold %d due to: %s", fold_idx, exc)
            train_pca = np.zeros((len(train_idx), 0), dtype=np.float32)
            test_pca = np.zeros((len(test_idx), 0), dtype=np.float32)
            pca_cols = []

        X_train_engineered = X_engineered.iloc[train_idx].copy()
        X_test_engineered = X_engineered.iloc[test_idx].copy()
        _fill_latency_inplace(
            X_train_engineered,
            features.iloc[train_idx]["latency_to_thr"],
            features.iloc[train_idx]["odor_window_len_s"],
        )
        _fill_latency_inplace(
            X_test_engineered,
            features.iloc[test_idx]["latency_to_thr"],
            features.iloc[test_idx]["odor_window_len_s"],
        )

        if train_pca.shape[1] > 0:
            X_train_engineered = pd.concat(
                [X_train_engineered.reset_index(drop=True), pd.DataFrame(train_pca, columns=pca_cols)],
                axis=1,
            )
            X_test_engineered = pd.concat(
                [X_test_engineered.reset_index(drop=True), pd.DataFrame(test_pca, columns=pca_cols)],
                axis=1,
            )
        else:
            X_train_engineered = X_train_engineered.reset_index(drop=True)
            X_test_engineered = X_test_engineered.reset_index(drop=True)

        X_train_engineered, X_test_engineered = _impute_with_median(
            X_train_engineered, X_test_engineered
        )
        scaler_engineered = StandardScaler()
        X_train_engineered_scaled = scaler_engineered.fit_transform(X_train_engineered)
        X_test_engineered_scaled = scaler_engineered.transform(X_test_engineered)

        X_train_raw = X_raw.iloc[train_idx].copy()
        X_test_raw = X_raw.iloc[test_idx].copy()
        X_train_raw, X_test_raw = _impute_with_median(X_train_raw, X_test_raw)
        scaler_raw = StandardScaler()
        X_train_raw_scaled = scaler_raw.fit_transform(X_train_raw)
        X_test_raw_scaled = scaler_raw.transform(X_test_raw)

        y_train_bin = y_bin.iloc[train_idx]
        y_test_bin = y_bin.iloc[test_idx]
        logistic = LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, random_state=RANDOM_SEED)
        logistic.fit(X_train_raw_scaled, y_train_bin)
        prob_test = logistic.predict_proba(X_test_raw_scaled)[:, 1]
        pred_test = (prob_test >= 0.5).astype(int)

        y_test_reg = y_reg.iloc[test_idx].to_numpy()
        y_reg_pred = np.full_like(y_test_reg, np.nan, dtype=float)
        y_train_reg = y_reg.iloc[train_idx]
        reg_mask_train = ~y_train_reg.isna()
        reg_mask_test = ~np.isnan(y_test_reg)
        if reg_mask_train.sum() >= 2:
            linear = LinearRegression()
            linear.fit(
                X_train_engineered_scaled[reg_mask_train],
                y_train_reg[reg_mask_train],
            )
            if reg_mask_test.any():
                y_reg_pred[reg_mask_test] = linear.predict(
                    X_test_engineered_scaled[reg_mask_test]
                )
        else:
            logger.warning("Insufficient reacted samples for regression in fold %d", fold_idx)

        subset_cols = [c for c in CLUSTER_FEATURES if c in X_train_engineered.columns]
        subset_cols += [c for c in pca_cols if c in X_train_engineered.columns]
        inferred_pred = np.zeros(len(test_idx), dtype=float)
        pos_train_mask = y_train_bin == 1
        pos_test_mask = y_test_bin == 1
        if subset_cols and pos_train_mask.sum() >= 5:
            subset_idx = [X_train_engineered.columns.get_loc(c) for c in subset_cols]
            pos_features = X_train_engineered_scaled[pos_train_mask.values][:, subset_idx]
            kmeans = KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init=10)
            try:
                kmeans.fit(pos_features)
                pos_assign = kmeans.labels_
                pos_auc = X_train_engineered.loc[
                    pos_train_mask.values, "odor_auc_above_thr"
                ].to_numpy()
                cluster_means = [pos_auc[pos_assign == k].mean() if np.any(pos_assign == k) else -np.inf for k in range(kmeans.n_clusters)]
                ordering = np.argsort(cluster_means)
                score_map = {cluster: rank + 1 for rank, cluster in enumerate(ordering)}
                test_features_subset = X_test_engineered_scaled[:, subset_idx]
                test_assign = kmeans.predict(test_features_subset)
                for i, is_pos in enumerate(pos_test_mask.values):
                    inferred_pred[i] = score_map[test_assign[i]] if is_pos else 0.0
            except ValueError as exc:
                logger.warning("KMeans failed in fold %d (%s); defaulting to score=1", fold_idx, exc)
                inferred_pred[pos_test_mask.values] = 1.0
        else:
            if not subset_cols:
                logger.warning("No subset columns available for inferred scoring in fold %d", fold_idx)
            elif pos_train_mask.sum() < 5:
                logger.warning("Not enough positive samples for inferred scoring in fold %d", fold_idx)
            inferred_pred[pos_test_mask.values] = 1.0

        oof_bin[test_idx] = pred_test
        oof_reg[test_idx] = y_reg_pred
        oof_inf[test_idx] = inferred_pred
        oof_proba[test_idx] = prob_test
        fold_fly = test_features["fly"].iloc[0]
        fold_assign[test_idx] = fold_fly

        feature_names_extended = X_train_engineered.columns.tolist()
        pca_components_per_fold[str(fold_fly)] = pca_cols

        logger.info(
            "Fold %d completed for fly %s: n_train=%d n_test=%d accuracy=%.3f",
            fold_idx,
            fold_fly,
            len(train_idx),
            len(test_idx),
            accuracy_score(y_test_bin, pred_test),
        )

    cm = confusion_matrix(y_bin, oof_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(y_bin, oof_bin, average="binary")
    accuracy = accuracy_score(y_bin, oof_bin)

    reacted_mask = ~y_reg.isna()
    reacted_indices = np.where(reacted_mask)[0]
    regression_metrics = {"r2": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
    if reacted_indices.size > 0:
        true_reg = y_reg.iloc[reacted_indices]
        pred_reg = pd.Series(oof_reg[reacted_indices], index=true_reg.index)
        valid_mask = ~pred_reg.isna()
        if valid_mask.any():
            pred_vals = pred_reg[valid_mask]
            true_vals = true_reg[valid_mask]
            regression_metrics = {
                "r2": r2_score(true_vals, pred_vals),
                "rmse": math.sqrt(mean_squared_error(true_vals, pred_vals)),
                "spearman": spearmanr(true_vals, pred_vals).correlation,
            }

    inferred_metrics = {"rmse": float("nan"), "spearman": float("nan")}
    if reacted_indices.size > 0:
        inferred_true = y_reg.iloc[reacted_indices]
        inferred_pred = pd.Series(oof_inf[reacted_indices], index=inferred_true.index)
        valid_mask = inferred_pred > 0
        if valid_mask.any():
            pred_vals = inferred_pred[valid_mask]
            true_vals = inferred_true[valid_mask]
            inferred_metrics = {
                "rmse": math.sqrt(mean_squared_error(true_vals, pred_vals)),
                "spearman": spearmanr(true_vals, pred_vals).correlation,
            }

    _plot_annotated_confusion_matrix(
        cm,
        output_dir / "cm_binary.png",
        "Binary Reaction Confusion Matrix (LOGO)",
    )

    plt.figure(figsize=(6, 4))
    plt.hist(oof_proba, bins=20, range=(0, 1))
    plt.xlabel("Predicted P(react)")
    plt.ylabel("Count")
    plt.title("Probability Calibration")
    prob_path = output_dir / "prob_calibration.png"
    plt.savefig(prob_path, bbox_inches="tight")
    plt.close()

    cv_predictions = features[["dataset", "fly", "trial_type", "trial_label"]].copy()
    cv_predictions["fold_fly"] = fold_assign
    cv_predictions["y_bin_true"] = y_bin.values
    cv_predictions["y_bin_pred"] = oof_bin
    cv_predictions["y_reg_true"] = y_reg.values
    cv_predictions["y_reg_pred"] = oof_reg
    cv_predictions["y_inf_pred"] = oof_inf
    cv_predictions["p_react"] = oof_proba

    cv_path = output_dir / "cv_predictions.csv"
    cv_predictions.to_csv(cv_path, index=False)

    logger.info(
        "Cross-validation metrics: accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f",
        accuracy,
        precision,
        recall,
        f1,
    )
    logger.info("Regression metrics: %s", regression_metrics)
    logger.info("Inferred intensity metrics: %s", inferred_metrics)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "regression": regression_metrics,
        "inferred": inferred_metrics,
        "cv_predictions": cv_predictions,
        "engineered_feature_names": feature_names_extended or engineered_names,
        "raw_feature_names": raw_names,
    }


def run_train_test_splits(
    features: pd.DataFrame,
    y_bin: pd.Series,
    y_reg: pd.Series,
    output_dir: Path,
    *,
    n_epochs: int = 5,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Dict[str, object]:
    _ensure_output_dir(output_dir)
    X_engineered, _ = _prepare_engineered_frame(features)
    X_raw, _ = _extract_raw_trace_matrix(features)

    indices = np.arange(len(features))
    metrics_records: List[Dict[str, float]] = []
    aggregated_cm = np.zeros((2, 2), dtype=int)
    prediction_frames: List[pd.DataFrame] = []
    roc_points: List[pd.DataFrame] = []

    for epoch in range(n_epochs):
        epoch_seed = random_state + epoch
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=y_bin.iloc[indices],
            random_state=epoch_seed,
        )

        train_features = features.iloc[train_idx]
        test_features = features.iloc[test_idx]

        train_vectors = train_features["odor_window_vector"].tolist()
        test_vectors = test_features["odor_window_vector"].tolist()

        try:
            pca, target_len = _fit_fold_pca(train_vectors, FOLD_PCA_COMPONENTS)
            train_pca = _transform_pca(pca, train_vectors, target_len)
            test_pca = _transform_pca(pca, test_vectors, target_len)
            pca_cols = [f"fold_pca_{i+1}" for i in range(train_pca.shape[1])]
        except ValueError as exc:
            logger.warning(
                "Skipping PCA in epoch %d due to: %s", epoch + 1, exc
            )
            train_pca = np.zeros((len(train_idx), 0), dtype=np.float32)
            test_pca = np.zeros((len(test_idx), 0), dtype=np.float32)
            pca_cols = []

        X_train_engineered = X_engineered.iloc[train_idx].copy()
        X_test_engineered = X_engineered.iloc[test_idx].copy()

        _fill_latency_inplace(
            X_train_engineered,
            features.iloc[train_idx]["latency_to_thr"],
            features.iloc[train_idx]["odor_window_len_s"],
        )
        _fill_latency_inplace(
            X_test_engineered,
            features.iloc[test_idx]["latency_to_thr"],
            features.iloc[test_idx]["odor_window_len_s"],
        )

        if train_pca.shape[1] > 0:
            X_train_engineered = pd.concat(
                [
                    X_train_engineered.reset_index(drop=True),
                    pd.DataFrame(train_pca, columns=pca_cols),
                ],
                axis=1,
            )
            X_test_engineered = pd.concat(
                [
                    X_test_engineered.reset_index(drop=True),
                    pd.DataFrame(test_pca, columns=pca_cols),
                ],
                axis=1,
            )
        else:
            X_train_engineered = X_train_engineered.reset_index(drop=True)
            X_test_engineered = X_test_engineered.reset_index(drop=True)

        X_train_engineered, X_test_engineered = _impute_with_median(
            X_train_engineered, X_test_engineered
        )
        scaler_engineered = StandardScaler()
        X_train_engineered_scaled = scaler_engineered.fit_transform(
            X_train_engineered
        )
        X_test_engineered_scaled = scaler_engineered.transform(X_test_engineered)

        X_train_raw = X_raw.iloc[train_idx].copy()
        X_test_raw = X_raw.iloc[test_idx].copy()
        X_train_raw, X_test_raw = _impute_with_median(X_train_raw, X_test_raw)
        scaler_raw = StandardScaler()
        X_train_raw_scaled = scaler_raw.fit_transform(X_train_raw)
        X_test_raw_scaled = scaler_raw.transform(X_test_raw)

        y_train_bin = y_bin.iloc[train_idx]
        y_test_bin = y_bin.iloc[test_idx]

        logistic = LogisticRegression(
            solver="lbfgs", C=1.0, max_iter=2000, random_state=epoch_seed
        )
        logistic.fit(X_train_raw_scaled, y_train_bin)
        prob_test = logistic.predict_proba(X_test_raw_scaled)[:, 1]
        pred_test = (prob_test >= 0.5).astype(int)

        y_test_reg = y_reg.iloc[test_idx].to_numpy()
        y_reg_pred = np.full_like(y_test_reg, np.nan, dtype=float)
        y_train_reg = y_reg.iloc[train_idx]
        reg_mask_train = ~y_train_reg.isna()
        reg_mask_test = ~np.isnan(y_test_reg)
        if reg_mask_train.sum() >= 2:
            linear = LinearRegression()
            linear.fit(
                X_train_engineered_scaled[reg_mask_train],
                y_train_reg[reg_mask_train],
            )
            if reg_mask_test.any():
                y_reg_pred[reg_mask_test] = linear.predict(
                    X_test_engineered_scaled[reg_mask_test]
                )
        else:
            linear = None
            logger.warning(
                "Insufficient reacted samples for regression in epoch %d",
                epoch + 1,
            )

        subset_cols = [c for c in CLUSTER_FEATURES if c in X_train_engineered.columns]
        subset_cols += [c for c in pca_cols if c in X_train_engineered.columns]
        inferred_pred = np.zeros(len(test_idx), dtype=float)
        pos_train_mask = y_train_bin == 1
        pos_test_mask = y_test_bin == 1
        if subset_cols and pos_train_mask.sum() >= 5:
            subset_idx = [X_train_engineered.columns.get_loc(c) for c in subset_cols]
            pos_features = X_train_engineered_scaled[pos_train_mask.values][:, subset_idx]
            kmeans = KMeans(n_clusters=5, random_state=epoch_seed, n_init=10)
            try:
                kmeans.fit(pos_features)
                pos_assign = kmeans.labels_
                pos_auc = X_train_engineered.loc[
                    pos_train_mask.values, "odor_auc_above_thr"
                ].to_numpy()
                cluster_means = [
                    pos_auc[pos_assign == k].mean() if np.any(pos_assign == k) else -np.inf
                    for k in range(kmeans.n_clusters)
                ]
                ordering = np.argsort(cluster_means)
                score_map = {cluster: rank + 1 for rank, cluster in enumerate(ordering)}
                test_features_subset = X_test_engineered_scaled[:, subset_idx]
                test_assign = kmeans.predict(test_features_subset)
                for i, is_pos in enumerate(pos_test_mask.values):
                    inferred_pred[i] = score_map[test_assign[i]] if is_pos else 0.0
            except ValueError as exc:
                logger.warning(
                    "KMeans failed in epoch %d (%s); defaulting positives to score=1",
                    epoch + 1,
                    exc,
                )
                inferred_pred[pos_test_mask.values] = 1.0
        else:
            if not subset_cols:
                logger.warning(
                    "No subset columns available for inferred scoring in epoch %d",
                    epoch + 1,
                )
            elif pos_train_mask.sum() < 5:
                logger.warning(
                    "Not enough positive samples for inferred scoring in epoch %d",
                    epoch + 1,
                )
            inferred_pred[pos_test_mask.values] = 1.0

        cm = confusion_matrix(y_test_bin, pred_test)
        aggregated_cm += cm
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_bin, pred_test, average="binary"
        )
        accuracy = accuracy_score(y_test_bin, pred_test)
        roc_auc = roc_auc_score(y_test_bin, prob_test)

        reacted_mask = ~np.isnan(y_test_reg)
        regression_metrics = {"r2": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
        if reacted_mask.any() and linear is not None:
            true_vals = y_test_reg[reacted_mask]
            pred_vals = y_reg_pred[reacted_mask]
            valid_mask = ~np.isnan(pred_vals)
            if valid_mask.any():
                regression_metrics = {
                    "r2": r2_score(true_vals[valid_mask], pred_vals[valid_mask]),
                    "rmse": math.sqrt(mean_squared_error(true_vals[valid_mask], pred_vals[valid_mask])),
                    "spearman": spearmanr(true_vals[valid_mask], pred_vals[valid_mask]).correlation,
                }

        inferred_metrics = {"rmse": float("nan"), "spearman": float("nan")}
        if reacted_mask.any():
            pos_valid = inferred_pred[reacted_mask] > 0
            if np.any(pos_valid):
                inferred_metrics = {
                    "rmse": math.sqrt(
                        mean_squared_error(
                            y_test_reg[reacted_mask][pos_valid],
                            inferred_pred[reacted_mask][pos_valid],
                        )
                    ),
                    "spearman": spearmanr(
                        y_test_reg[reacted_mask][pos_valid],
                        inferred_pred[reacted_mask][pos_valid],
                    ).correlation,
                }

        _plot_annotated_confusion_matrix(
            cm,
            output_dir / f"cm_binary_epoch_{epoch + 1:02d}.png",
            f"Binary Confusion (Epoch {epoch + 1})",
        )

        plt.figure(figsize=(6, 4))
        plt.hist(prob_test, bins=20, range=(0, 1))
        plt.xlabel("Predicted P(react)")
        plt.ylabel("Count")
        plt.title(f"Probability Histogram (Epoch {epoch + 1})")
        plt.tight_layout()
        plt.savefig(output_dir / f"prob_calibration_epoch_{epoch + 1:02d}.png", bbox_inches="tight")
        plt.close()

        fpr, tpr, _ = roc_curve(y_test_bin, prob_test)
        roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "epoch": epoch + 1})
        roc_points.append(roc_df)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"Epoch {epoch + 1} (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve (Epoch {epoch + 1})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / f"roc_curve_epoch_{epoch + 1:02d}.png", bbox_inches="tight")
        plt.close()

        epoch_predictions = features.iloc[test_idx][
            ["dataset", "fly", "trial_type", "trial_label"]
        ].copy()
        epoch_predictions["epoch"] = epoch + 1
        epoch_predictions["y_bin_true"] = y_test_bin.values
        epoch_predictions["y_bin_pred"] = pred_test
        epoch_predictions["p_react"] = prob_test
        epoch_predictions["y_reg_true"] = y_test_reg
        epoch_predictions["y_reg_pred"] = y_reg_pred
        epoch_predictions["y_inf_pred"] = inferred_pred
        prediction_frames.append(epoch_predictions)

        metrics_records.append(
            {
                "epoch": epoch + 1,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "tn": cm[0, 0],
                "fp": cm[0, 1],
                "fn": cm[1, 0],
                "tp": cm[1, 1],
                "reg_r2": regression_metrics["r2"],
                "reg_rmse": regression_metrics["rmse"],
                "reg_spearman": regression_metrics["spearman"],
                "inf_rmse": inferred_metrics["rmse"],
                "inf_spearman": inferred_metrics["spearman"],
            }
        )

        logger.info(
            "Epoch %d hold-out metrics: accuracy=%.3f precision=%.3f recall=%.3f f1=%.3f roc_auc=%.3f",
            epoch + 1,
            accuracy,
            precision,
            recall,
            f1,
            roc_auc,
        )

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = output_dir / "holdout_epoch_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    predictions_path = output_dir / "holdout_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    if metrics_records:
        metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        mean_metrics = metrics_df[metric_columns].mean().to_dict()
        std_metrics = metrics_df[metric_columns].std(ddof=0).to_dict()
    else:
        mean_metrics = {}
        std_metrics = {}

    _plot_annotated_confusion_matrix(
        aggregated_cm,
        output_dir / "cm_binary_holdout_total.png",
        "Binary Confusion (Summed Hold-out Epochs)",
    )

    plt.figure(figsize=(8, 5))
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if metric in metrics_df:
            plt.plot(metrics_df["epoch"], metrics_df[metric], marker="o", label=metric)
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Hold-out Metric Trends")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "holdout_metric_trends.png", bbox_inches="tight")
    plt.close()

    if roc_points:
        roc_concat = pd.concat(roc_points, ignore_index=True)
    else:
        roc_concat = pd.DataFrame(columns=["fpr", "tpr", "epoch"])

    return {
        "epoch_metrics": metrics_df,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "aggregated_confusion": aggregated_cm,
        "predictions": predictions_df,
        "roc_points": roc_concat,
    }


def _plot_feature_weights(model, feature_names: Sequence[str], path: Path, title: str) -> None:
    if not hasattr(model, "coef_"):
        logger.warning("Model %s does not expose coef_; skipping weight plot", model)
        return
    coef = model.coef_[0] if getattr(model.coef_, "ndim", 1) > 1 else model.coef_
    series = pd.Series(coef, index=feature_names, dtype=float)
    series = series.reindex(series.abs().sort_values(ascending=False).index)
    top = series.iloc[:20]
    plt.figure(figsize=(8, 6))
    top.sort_values().plot(kind="barh")
    plt.title(title)
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _save_feature_report(
    feature_names: Sequence[str],
    coefficients: Sequence[float],
    path: Path,
) -> None:
    series = pd.Series(coefficients, index=feature_names, dtype=float)
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "category": [_categorize_feature(name) for name in feature_names],
            "coefficient": series.values,
            "abs_coefficient": series.abs().values,
        }
    )
    df.sort_values("abs_coefficient", ascending=False, inplace=True)
    df.to_csv(path, index=False)


def train_full_models(
    features: pd.DataFrame,
    y_bin: pd.Series,
    y_reg: pd.Series,
    output_dir: Path,
) -> Dict[str, object]:
    _ensure_output_dir(output_dir)
    X_engineered, engineered_names = _prepare_engineered_frame(features)
    X_raw, raw_names = _extract_raw_trace_matrix(features)

    vectors = features["odor_window_vector"].tolist()
    try:
        pca, target_len = _fit_fold_pca(vectors, FOLD_PCA_COMPONENTS)
        pca_matrix = _transform_pca(pca, vectors, target_len)
        pca_cols = [f"fold_pca_{i+1}" for i in range(pca_matrix.shape[1])]
        X_engineered_full = pd.concat(
            [X_engineered.reset_index(drop=True), pd.DataFrame(pca_matrix, columns=pca_cols)],
            axis=1,
        )
    except ValueError as exc:
        logger.warning("Skipping PCA for full-data training due to: %s", exc)
        pca = None
        target_len = None
        pca_cols = []
        X_engineered_full = X_engineered.reset_index(drop=True)

    _fill_latency_inplace(
        X_engineered_full,
        features["latency_to_thr"],
        features["odor_window_len_s"],
    )
    X_engineered_full = X_engineered_full.replace([np.inf, -np.inf], np.nan)
    engineered_medians = X_engineered_full.median()
    X_engineered_full = X_engineered_full.fillna(engineered_medians)
    _log_feature_breakdown(
        X_engineered_full.columns.tolist(), "Full engineered training matrix"
    )

    X_raw_full = X_raw.replace([np.inf, -np.inf], np.nan)
    raw_medians = X_raw_full.median()
    X_raw_full = X_raw_full.fillna(raw_medians)
    logger.info(
        "Full raw trace matrix contains %d features spanning dir_val_%d..dir_val_%d",
        X_raw_full.shape[1],
        ODOR_ON_COLUMN,
        ODOR_OFF_COLUMN,
    )

    logistic_scaler = StandardScaler()
    X_raw_scaled = logistic_scaler.fit_transform(X_raw_full)

    logistic = LogisticRegression(
        solver="lbfgs", C=1.0, max_iter=2000, random_state=RANDOM_SEED
    )
    logistic.fit(X_raw_scaled, y_bin)

    engineered_scaler = StandardScaler()
    X_engineered_scaled = engineered_scaler.fit_transform(X_engineered_full)

    reg_mask = ~y_reg.isna()
    linear: LinearRegression | None = None
    if reg_mask.sum() >= 2:
        linear = LinearRegression()
        linear.fit(X_engineered_scaled[reg_mask], y_reg[reg_mask])
    else:
        logger.warning(
            "Insufficient reacted samples to fit linear regression on full data"
        )

    joblib.dump(
        {
            "model": logistic,
            "feature_names": raw_names,
            "odor_window_range": (ODOR_ON_COLUMN, ODOR_OFF_COLUMN),
        },
        output_dir / "final_logistic.pkl",
    )
    joblib.dump(
        {
            "model": linear,
            "feature_names": X_engineered_full.columns.tolist(),
            "pca": pca,
            "pca_target_len": target_len,
        },
        output_dir / "final_linear.pkl",
    )
    joblib.dump(
        {
            "logistic_scaler": logistic_scaler,
            "logistic_medians": raw_medians,
            "engineered_scaler": engineered_scaler,
            "engineered_medians": engineered_medians,
            "pca": pca,
            "pca_target_len": target_len,
        },
        output_dir / "final_scaler.pkl",
    )

    _plot_feature_weights(
        logistic,
        raw_names,
        output_dir / "feature_weights_logistic.png",
        "Logistic Regression Weights",
    )
    _save_feature_report(
        raw_names,
        logistic.coef_[0] if getattr(logistic.coef_, "ndim", 1) > 1 else logistic.coef_,
        output_dir / "logistic_feature_usage.csv",
    )
    if linear is not None:
        _plot_feature_weights(
            linear,
            X_engineered_full.columns,
            output_dir / "feature_weights_linear.png",
            "Linear Regression Weights",
        )
        _save_feature_report(
            X_engineered_full.columns,
            linear.coef_,
            output_dir / "linear_feature_usage.csv",
        )

    logistic_series = pd.Series(
        logistic.coef_[0] if getattr(logistic.coef_, "ndim", 1) > 1 else logistic.coef_,
        index=raw_names,
    )
    top_logistic = logistic_series.abs().sort_values(ascending=False).head(10)
    logger.info(
        "Top logistic raw contributors: %s",
        ", ".join(f"{idx} ({logistic_series[idx]:.3f})" for idx in top_logistic.index),
    )
    if linear is not None:
        linear_series = pd.Series(linear.coef_, index=X_engineered_full.columns)
        top_linear = linear_series.abs().sort_values(ascending=False).head(10)
        logger.info(
            "Top linear engineered contributors: %s",
            ", ".join(
                f"{idx} ({linear_series[idx]:.3f})" for idx in top_linear.index
            ),
        )

    return {
        "logistic": logistic,
        "linear": linear,
        "logistic_scaler": logistic_scaler,
        "engineered_scaler": engineered_scaler,
        "logistic_medians": raw_medians,
        "engineered_medians": engineered_medians,
        "pca": pca,
        "pca_target_len": target_len,
        "raw_feature_names": raw_names,
        "engineered_feature_names": X_engineered_full.columns.tolist(),
    }


__all__ = [
    "run_cross_validation",
    "run_train_test_splits",
    "train_full_models",
]

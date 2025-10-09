"""Join engineered features with scoring spreadsheets and cluster outputs."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCORE_PATH = "/home/ramanlab/Documents/cole/model/FlyBehaviorPER/scoring_results_opto_new.csv"
CLUSTER_PATH = "/home/ramanlab/PycharmProjects/FlyBehaviorScoring/outputs/unsup/20251003_155149/trial_clusters_reaction_clusters.csv"
REPORT_PATH = "/home/ramanlab/PycharmProjects/FlyBehaviorScoring/outputs/unsup/20251003_155149/report_reaction_clusters.csv"


TRIAL_KEY_TEMPLATE = "{dataset}__{fly}__{trial_type}__{trial_label}"


def _build_trial_key(df: pd.DataFrame) -> pd.Series:
    return df["dataset"].astype(str) + "__" + df["fly"].astype(str) + "__" + df["trial_type"].astype(str) + "__" + df["trial_label"].astype(str)


def load_scoring(path: str = SCORE_PATH) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Scoring CSV not found at {path_obj}")
    logger.info("Loading scoring data from %s", path_obj)
    scoring = pd.read_csv(path_obj)
    if "user_score_odor" not in scoring.columns:
        raise ValueError("Scoring CSV missing required column 'user_score_odor'")

    has_id_cols = {"dataset", "fly", "trial_type", "trial_label"} <= set(scoring.columns)
    if has_id_cols:
        scoring["dataset"] = scoring["dataset"].astype(str)
        scoring["fly"] = scoring["fly"].astype(str)
        scoring["trial_type"] = scoring["trial_type"].astype(str)
        scoring["trial_label"] = scoring["trial_label"].astype(str)
        scoring["trial_key"] = _build_trial_key(scoring)
    elif "matrix_row_index" not in scoring.columns:
        raise ValueError(
            "Scoring CSV must include either dataset/fly/trial identifiers or matrix_row_index"
        )

    if "matrix_row_index" in scoring.columns:
        scoring["matrix_row_index"] = pd.to_numeric(
            scoring["matrix_row_index"], errors="coerce"
        )
        invalid = scoring["matrix_row_index"].isna() | (scoring["matrix_row_index"] < 0)
        if invalid.any():
            logger.warning(
                "Dropping %d scoring rows with invalid matrix_row_index",
                int(invalid.sum()),
            )
            scoring = scoring.loc[~invalid]
        scoring["matrix_row_index"] = scoring["matrix_row_index"].astype(int)

    scoring["user_score_odor"] = pd.to_numeric(scoring["user_score_odor"], errors="coerce")
    if scoring["user_score_odor"].isna().any():
        logger.warning("Found NaN user_score_odor values; they will be dropped during join")
    scoring["y_bin"] = (scoring["user_score_odor"] > 0).astype(int)
    scoring["y_reg"] = scoring["user_score_odor"].where(scoring["user_score_odor"] > 0, np.nan)
    if "trial_key" in scoring.columns:
        dup_mask = scoring["trial_key"].duplicated()
        if dup_mask.any():
            logger.warning(
                "Scoring CSV has %d duplicated trials; keeping first",
                int(dup_mask.sum()),
            )
            scoring = scoring.loc[~dup_mask]
    elif "matrix_row_index" in scoring.columns:
        dup_mask = scoring["matrix_row_index"].duplicated()
        if dup_mask.any():
            logger.warning(
                "Scoring CSV has %d duplicated matrix rows; keeping first",
                int(dup_mask.sum()),
            )
            scoring = scoring.loc[~dup_mask]
    logger.info("Loaded %d scoring rows", len(scoring))
    return scoring


def load_clusters(path: str = CLUSTER_PATH) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Cluster CSV not found at {path_obj}")
    logger.info("Loading cluster data from %s", path_obj)
    clusters = pd.read_csv(path_obj)
    required = {"dataset", "fly", "trial_type", "trial_label"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(f"Cluster CSV missing columns: {sorted(missing)}")
    clusters["dataset"] = clusters["dataset"].astype(str)
    clusters["fly"] = clusters["fly"].astype(str)
    clusters["trial_type"] = clusters["trial_type"].astype(str)
    clusters["trial_label"] = clusters["trial_label"].astype(str)
    clusters["trial_key"] = _build_trial_key(clusters)
    cluster_cols = [c for c in clusters.columns if c.startswith("pc_")]
    if "cluster_label" in clusters.columns:
        clusters["cluster_label"] = clusters["cluster_label"].astype(str)
        dummy = pd.get_dummies(clusters["cluster_label"], prefix="cluster", drop_first=True)
        clusters = pd.concat([clusters, dummy], axis=1)
        logger.info("One-hot encoded cluster labels into %d columns", dummy.shape[1])
    else:
        logger.warning("cluster_label column missing; skipping one-hot encoding")
    keep_cols = [
        "trial_key",
        "cluster_label",
    ] + cluster_cols + [c for c in clusters.columns if c.startswith("cluster_")]
    keep_cols = [c for c in keep_cols if c in clusters.columns]
    clusters = clusters[keep_cols].drop_duplicates(subset="trial_key")
    logger.info("Cluster table reduced to %d unique trials", len(clusters))
    return clusters


def _load_optional_report(path: str = REPORT_PATH) -> pd.DataFrame | None:
    path_obj = Path(path)
    if not path_obj.exists():
        logger.info("Optional report %s not found; skipping", path_obj)
        return None
    logger.info("Loading optional report from %s", path_obj)
    report = pd.read_csv(path_obj)
    required = {"dataset", "fly", "trial_type", "trial_label"}
    if not required <= set(report.columns):
        logger.warning("Optional report missing trial-level identifiers; skipping merge")
        return None
    report = report.copy()
    report["dataset"] = report["dataset"].astype(str)
    report["fly"] = report["fly"].astype(str)
    report["trial_type"] = report["trial_type"].astype(str)
    report["trial_label"] = report["trial_label"].astype(str)
    report["trial_key"] = _build_trial_key(report)
    duplicated = report["trial_key"].duplicated()
    if duplicated.any():
        logger.warning(
            "Optional report has %d duplicated trials; keeping first occurrences",
            duplicated.sum(),
        )
        report = report[~duplicated]
    return report


def join_features(
    features: pd.DataFrame,
    scoring_path: str = SCORE_PATH,
    clusters_path: str = CLUSTER_PATH,
    report_path: str = REPORT_PATH,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if "trial_key" not in features.columns:
        raise ValueError("Features DataFrame must contain trial_key")

    scoring = load_scoring(scoring_path)
    clusters = load_clusters(clusters_path)
    report = _load_optional_report(report_path)

    logger.info("Joining engineered features (%d rows) with scoring", len(features))
    merged = features.copy()
    if "trial_key" in scoring.columns:
        merged = merged.merge(
            scoring[["trial_key", "y_bin", "y_reg"]],
            on="trial_key",
            how="left",
        )
    if "matrix_row_index" in scoring.columns:
        if "matrix_row_index" not in merged.columns:
            raise ValueError("Features DataFrame missing matrix_row_index for scoring join")
        index_join = scoring[["matrix_row_index", "y_bin", "y_reg"]].drop_duplicates(
            subset="matrix_row_index"
        )
        merged = merged.merge(
            index_join,
            on="matrix_row_index",
            how="left",
            suffixes=("", "_by_index"),
        )
        if "y_bin_by_index" in merged.columns:
            merged["y_bin"] = merged["y_bin"].fillna(merged.pop("y_bin_by_index"))
        if "y_reg_by_index" in merged.columns:
            merged["y_reg"] = merged["y_reg"].fillna(merged.pop("y_reg_by_index"))
    matched = merged["y_bin"].notna().sum()
    logger.info("Scoring provided labels for %d trials", int(matched))
    before_drop = len(merged)
    merged = merged.dropna(subset=["y_bin"])
    if len(merged) < before_drop:
        logger.warning("Dropped %d trials without binary labels", before_drop - len(merged))
    merged = merged.merge(clusters, on="trial_key", how="left")
    if report is not None:
        report_cols = [c for c in report.columns if c not in {"dataset", "fly", "trial_type", "trial_label"}]
        merged = merged.merge(report[["trial_key"] + report_cols], on="trial_key", how="left")
    merged = merged.drop_duplicates(subset="trial_key")
    logger.info("Final merged dataset contains %d trials", len(merged))

    cluster_feature_cols = [c for c in merged.columns if c.startswith("cluster_")]
    if cluster_feature_cols:
        if merged[cluster_feature_cols].isna().any(axis=None):
            logger.warning("Some cluster features are missing; consider checking cluster inputs")

    y_bin = merged["y_bin"].astype(int)
    y_reg = merged["y_reg"].astype(float)
    return merged, y_bin, y_reg


__all__ = ["join_features", "load_scoring", "load_clusters"]

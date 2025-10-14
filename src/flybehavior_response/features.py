"""Feature selection and preprocessing utilities."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .io import FEATURE_COLUMNS
from .logging_utils import get_logger

DEFAULT_FEATURES = ["AUC-During", "TimeToPeak-During", "Peak-Value"]
DISCOURAGED_FEATURES = {"AUC-During-Before-Ratio", "AUC-After-Before-Ratio"}
AUC_BEFORE = "AUC-Before"


def parse_feature_list(raw: str | None, include_auc_before: bool) -> List[str]:
    """Parse the CLI feature string into a list."""
    if raw:
        features = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        features = list(DEFAULT_FEATURES)
    if include_auc_before and AUC_BEFORE not in features:
        features.append(AUC_BEFORE)
    return features


def validate_features(selected: Sequence[str], available: Iterable[str], logger_name: str = __name__) -> List[str]:
    """Ensure selected features are present in dataset."""
    available_set = set(available)
    missing = [feat for feat in selected if feat not in available_set]
    if missing:
        raise ValueError(f"Requested features not found in dataset: {missing}")
    logger = get_logger(logger_name)
    discouraged = sorted(set(selected) & DISCOURAGED_FEATURES)
    if discouraged:
        logger.warning(
            "Selected ratio features %s are discouraged due to instability. Proceed with caution.",
            discouraged,
        )
    ordered = [feat for feat in available if feat in selected]
    return ordered


def build_column_transformer(
    trace_columns: Sequence[str],
    feature_columns: Sequence[str],
    selected_features: Sequence[str],
    *,
    use_raw_pca: bool,
    n_pcs: int,
    seed: int,
) -> ColumnTransformer:
    """Construct the preprocessing ColumnTransformer."""
    transformers = []
    if use_raw_pca:
        if not trace_columns:
            raise ValueError("Trace columns required for PCA preprocessing.")
        pca_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("pca", PCA(n_components=min(n_pcs, len(trace_columns)), random_state=seed)),
            ]
        )
        transformers.append(("trace_pca", pca_pipeline, list(trace_columns)))
    if selected_features:
        feature_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("feat_scale", feature_pipeline, list(selected_features)))
    if not transformers:
        raise ValueError("At least one transformer must be constructed.")
    return ColumnTransformer(transformers=transformers, remainder="drop")

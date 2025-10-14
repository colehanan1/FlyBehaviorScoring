"""Model factories for flybehavior_response."""

from __future__ import annotations

from typing import Iterable

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .logging_utils import get_logger

MODEL_LDA = "lda"
MODEL_LOGREG = "logreg"


def create_estimator(model_type: str, seed: int) -> object:
    if model_type == MODEL_LDA:
        return LinearDiscriminantAnalysis()
    if model_type == MODEL_LOGREG:
        return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    raise ValueError(f"Unsupported model type: {model_type}")


def build_model_pipeline(preprocessor, model_type: str, seed: int) -> Pipeline:
    """Construct a full pipeline with preprocessing and estimator."""
    estimator = create_estimator(model_type, seed)
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


def supported_models() -> Iterable[str]:
    return [MODEL_LDA, MODEL_LOGREG]

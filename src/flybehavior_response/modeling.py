"""Model factories for flybehavior_response."""

from __future__ import annotations

from typing import Iterable

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

MODEL_LDA = "lda"
MODEL_LOGREG = "logreg"
MODEL_MLP = "mlp"


def create_estimator(
    model_type: str,
    seed: int,
    *,
    logreg_solver: str = "lbfgs",
    logreg_max_iter: int = 1000,
) -> object:
    if model_type == MODEL_LDA:
        return LinearDiscriminantAnalysis()
    if model_type == MODEL_LOGREG:
        if logreg_solver not in {"lbfgs", "liblinear", "saga"}:
            raise ValueError(f"Unsupported logistic regression solver: {logreg_solver}")
        return LogisticRegression(
            max_iter=logreg_max_iter,
            solver=logreg_solver,
            random_state=seed,
        )
    if model_type == MODEL_MLP:
        return MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def build_model_pipeline(
    preprocessor,
    *,
    model_type: str,
    seed: int,
    logreg_solver: str = "lbfgs",
    logreg_max_iter: int = 1000,
) -> Pipeline:
    """Construct a full pipeline with preprocessing and estimator."""
    estimator = create_estimator(
        model_type,
        seed,
        logreg_solver=logreg_solver,
        logreg_max_iter=logreg_max_iter,
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


def supported_models() -> Iterable[str]:
    return [MODEL_LDA, MODEL_LOGREG, MODEL_MLP]

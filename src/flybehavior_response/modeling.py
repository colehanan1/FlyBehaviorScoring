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
MODEL_MLP_ADAM = "mlp_adam"


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
            hidden_layer_sizes=20000,
            max_iter=1000,
            random_state=seed,
        )
    if model_type == MODEL_MLP_ADAM:
        return MLPClassifier(
            hidden_layer_sizes=(512, 256, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=1500,
            early_stopping=False,
            random_state=seed,
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
    return [MODEL_LDA, MODEL_LOGREG, MODEL_MLP, MODEL_MLP_ADAM]

"""Model factories for flybehavior_response."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .sample_weighted_mlp import SampleWeightedMLPClassifier

MODEL_LDA = "lda"
MODEL_LOGREG = "logreg"
MODEL_MLP = "mlp"
MODEL_FP_OPTIMIZED_MLP = "fp_optimized_mlp"

ARCHITECTURE_SINGLE = "single"
ARCHITECTURE_TWO_LAYER = "two_layer"


def _parse_int_sequence(raw: Sequence[object] | str) -> Tuple[int, ...]:
    """Coerce a sequence representation into integer layer widths."""

    if isinstance(raw, str):
        cleaned = raw.strip().replace("(", "").replace(")", "")
        cleaned = cleaned.replace(" ", "")
        if not cleaned:
            raise ValueError("hidden_layer_sizes string cannot be empty")
        if "_" in cleaned:
            tokens = cleaned.split("_")
        else:
            tokens = [token for token in cleaned.split(",") if token]
    else:
        tokens = [str(value) for value in raw]
    return tuple(int(token) for token in tokens if token)


def resolve_hidden_layer_sizes_from_params(params: Mapping[str, object]) -> Tuple[int, ...]:
    """Derive the hidden-layer configuration from a parameter mapping."""

    if "hidden_layer_sizes" in params and params["hidden_layer_sizes"] is not None:
        raw_sizes = params["hidden_layer_sizes"]
        if isinstance(raw_sizes, (list, tuple)):
            return tuple(int(size) for size in raw_sizes)
        if isinstance(raw_sizes, str):
            return _parse_int_sequence(raw_sizes)
        if isinstance(raw_sizes, int):
            return (int(raw_sizes),)
        raise TypeError(
            "Unsupported hidden_layer_sizes type: " f"{type(raw_sizes)!r}."
        )

    architecture = params.get("architecture")
    if architecture is None:
        raise ValueError(
            "Unable to resolve hidden_layer_sizes. Provide hidden_layer_sizes or architecture metadata."
        )

    arch_token = str(architecture).strip().lower()
    if arch_token == ARCHITECTURE_SINGLE:
        h1 = params.get("h1")
        if h1 is None:
            raise ValueError("Single-layer architecture requires an 'h1' parameter.")
        return (int(h1),)

    if arch_token == ARCHITECTURE_TWO_LAYER:
        layer_config = params.get("layer_config")
        if layer_config is None:
            raise ValueError("Two-layer architecture requires a 'layer_config' parameter.")
        if isinstance(layer_config, (list, tuple)):
            tokens = [str(value) for value in layer_config]
        else:
            tokens = _parse_int_sequence(layer_config)
            return tokens
        return tuple(int(token) for token in tokens if token)

    raise ValueError(
        "Unsupported architecture token when resolving hidden_layer_sizes: "
        f"{architecture!r}"
    )


def normalise_mlp_params(params: Mapping[str, object]) -> dict:
    """Standardise Optuna-derived parameters for downstream training."""

    required_keys = ["n_components", "alpha", "batch_size"]
    missing = [key for key in required_keys if key not in params]
    if missing:
        raise ValueError(f"Missing required MLP parameters: {missing}")

    learning_rate_value = params.get("learning_rate_init", params.get("learning_rate"))
    if learning_rate_value is None:
        raise ValueError("Missing learning rate parameter (learning_rate_init or learning_rate).")

    hidden_layers = resolve_hidden_layer_sizes_from_params(params)

    consolidated: dict = {
        "n_components": int(params["n_components"]),
        "alpha": float(params["alpha"]),
        "batch_size": int(params["batch_size"]),
        "learning_rate_init": float(learning_rate_value),
        "hidden_layer_sizes": hidden_layers,
    }

    architecture = params.get("architecture")
    if architecture is not None:
        arch_token = str(architecture).strip().lower()
        consolidated["architecture"] = arch_token
        if arch_token == ARCHITECTURE_SINGLE:
            h1 = params.get("h1")
            if h1 is not None:
                consolidated["h1"] = int(h1)
        elif arch_token == ARCHITECTURE_TWO_LAYER:
            layer_config = params.get("layer_config")
            if layer_config is None:
                layer_config = "_".join(str(size) for size in hidden_layers)
            consolidated["layer_config"] = str(layer_config)

    return consolidated


def _build_mlp_from_params(params: Mapping[str, object], seed: int) -> SampleWeightedMLPClassifier:
    """Instantiate an MLP classifier from a consolidated parameter mapping."""

    normalised = normalise_mlp_params(params)
    return SampleWeightedMLPClassifier(
        hidden_layer_sizes=normalised["hidden_layer_sizes"],
        activation="relu",
        solver="adam",
        alpha=normalised["alpha"],
        batch_size=normalised["batch_size"],
        learning_rate_init=normalised["learning_rate_init"],
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
        random_state=seed,
    )


def create_estimator(
    model_type: str,
    seed: int,
    *,
    logreg_solver: str = "lbfgs",
    logreg_max_iter: int = 1000,
    mlp_params: Mapping[str, object] | None = None,
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
        if mlp_params is not None:
            return _build_mlp_from_params(mlp_params, seed)
        return SampleWeightedMLPClassifier(
            hidden_layer_sizes=10000,
            max_iter=1000,
            random_state=seed,
        )
    if model_type == MODEL_FP_OPTIMIZED_MLP:
        return SampleWeightedMLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            max_iter=100,
            batch_size=32,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
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
    mlp_params: Mapping[str, object] | None = None,
) -> Pipeline:
    """Construct a full pipeline with preprocessing and estimator."""
    estimator = create_estimator(
        model_type,
        seed,
        logreg_solver=logreg_solver,
        logreg_max_iter=logreg_max_iter,
        mlp_params=mlp_params,
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


def supported_models() -> Iterable[str]:
    return [MODEL_LDA, MODEL_LOGREG, MODEL_MLP, MODEL_FP_OPTIMIZED_MLP]

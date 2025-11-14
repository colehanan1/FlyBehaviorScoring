"""Model factories for flybehavior_response."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .sample_weighted_mlp import SampleWeightedMLPClassifier

MODEL_LDA = "lda"
MODEL_LOGREG = "logreg"
MODEL_RF = "random_forest"
MODEL_MLP = "mlp"
MODEL_FP_OPTIMIZED_MLP = "fp_optimized_mlp"

ARCHITECTURE_SINGLE = "single"
ARCHITECTURE_TWO_LAYER = "two_layer"

ALLOWED_BATCH_SIZES: Tuple[int, ...] = (8, 16, 32)
ALLOWED_LAYER_WIDTHS: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024)


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
        if h1 is not None:
            return (int(h1),)
        hidden_keys = sorted(
            [key for key in params if key.lower().startswith("h") and key[1:].isdigit()],
            key=lambda name: int(name[1:]),
        )
        if hidden_keys:
            return (int(params[hidden_keys[0]]),)
        raise ValueError("Single-layer architecture requires an 'h1' parameter or explicit hidden sizes.")

    if arch_token == ARCHITECTURE_TWO_LAYER:
        layer_config = params.get("layer_config")
        if layer_config is not None:
            if isinstance(layer_config, (list, tuple)):
                tokens = [str(value) for value in layer_config]
                return tuple(int(token) for token in tokens if token)
            return _parse_int_sequence(layer_config)

        hidden_keys = sorted(
            [key for key in params if key.lower().startswith("h") and key[1:].isdigit()],
            key=lambda name: int(name[1:]),
        )
        if hidden_keys:
            widths = tuple(int(params[key]) for key in hidden_keys)
            if len(widths) < 2:
                raise ValueError("Two-layer architecture requires at least h1 and h2 values.")
            return widths

        raise ValueError(
            "Two-layer architecture requires layer widths via layer_config, hidden_layer_sizes, or explicit h1/h2 keys."
        )

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

    raw_model_type = params.get("model_type")
    model_variant = str(raw_model_type) if raw_model_type is not None else MODEL_MLP
    if model_variant == MODEL_FP_OPTIMIZED_MLP and len(hidden_layers) != 2:
        raise ValueError(
            "fp_optimized_mlp requires exactly two hidden layers; received "
            f"{hidden_layers}."
        )

    batch_size_value = int(params["batch_size"])
    if batch_size_value not in ALLOWED_BATCH_SIZES:
        raise ValueError(
            "Batch size must be one of the supported powers of two: "
            f"{ALLOWED_BATCH_SIZES}. Received {batch_size_value}."
        )

    consolidated: dict = {
        "n_components": int(params["n_components"]),
        "alpha": float(params["alpha"]),
        "batch_size": batch_size_value,
        "learning_rate_init": float(learning_rate_value),
        "hidden_layer_sizes": hidden_layers,
    }

    for width in hidden_layers:
        if int(width) not in ALLOWED_LAYER_WIDTHS:
            raise ValueError(
                "Hidden layer widths must be selected from the supported set "
                f"{ALLOWED_LAYER_WIDTHS}. Received {hidden_layers}."
            )

    if "selected_features" in params and params["selected_features"] is not None:
        feature_subset = params["selected_features"]
        if isinstance(feature_subset, str):
            tokens = [token.strip() for token in feature_subset.split(",") if token.strip()]
            consolidated["selected_features"] = tuple(tokens)
        elif isinstance(feature_subset, (list, tuple)):
            consolidated["selected_features"] = tuple(str(item) for item in feature_subset)
        else:
            raise TypeError(
                "selected_features must be provided as a sequence or comma-separated string."
            )

    if "class_weight" in params and params["class_weight"] is not None:
        class_weight = params["class_weight"]
        if not isinstance(class_weight, Mapping):
            raise TypeError("class_weight must be a mapping from class label to multiplier.")
        consolidated["class_weight"] = {
            int(class_label): float(multiplier)
            for class_label, multiplier in class_weight.items()
        }

    if raw_model_type is not None:
        consolidated["model_type"] = model_variant

    architecture = params.get("architecture")
    if architecture is not None:
        arch_token = str(architecture).strip().lower()
        consolidated["architecture"] = arch_token
        if model_variant == MODEL_FP_OPTIMIZED_MLP and arch_token != ARCHITECTURE_TWO_LAYER:
            raise ValueError(
                "fp_optimized_mlp requires a two-layer architecture; received "
                f"{arch_token}."
            )
        if arch_token == ARCHITECTURE_SINGLE:
            if hidden_layers:
                consolidated["h1"] = int(hidden_layers[0])
        elif arch_token == ARCHITECTURE_TWO_LAYER:
            layer_config = params.get("layer_config")
            if layer_config is None:
                layer_config = "_".join(str(size) for size in hidden_layers)
            consolidated["layer_config"] = str(layer_config)
            for idx, width in enumerate(hidden_layers, start=1):
                consolidated[f"h{idx}"] = int(width)

    elif model_variant == MODEL_FP_OPTIMIZED_MLP:
        consolidated["architecture"] = ARCHITECTURE_TWO_LAYER
        consolidated["layer_config"] = "_".join(str(int(size)) for size in hidden_layers)
        for idx, width in enumerate(hidden_layers, start=1):
            consolidated[f"h{idx}"] = int(width)

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
    logreg_solver: str = "liblinear",
    logreg_max_iter: int = 1000,
    logreg_class_weight: Mapping[int, float] | str | None = None,
    rf_n_estimators: int = 100,
    rf_max_depth: int | None = None,
    rf_class_weight: Mapping[int, float] | str | None = None,
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
            class_weight=logreg_class_weight,
            random_state=seed,
        )
    if model_type == MODEL_RF:
        return RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            class_weight=rf_class_weight,
            random_state=seed,
            n_jobs=-1,  # Use all available cores
        )
    if model_type == MODEL_MLP:
        if mlp_params is not None:
            return _build_mlp_from_params(mlp_params, seed)
        return SampleWeightedMLPClassifier(
            hidden_layer_sizes=5000,
            max_iter=1000,
            random_state=seed,
        )
    if model_type == MODEL_FP_OPTIMIZED_MLP:
        return SampleWeightedMLPClassifier(
            hidden_layer_sizes=(1024, 512),
            activation="relu",
            solver="adam",
            max_iter=1000,
            batch_size=16,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=50,
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
    logreg_class_weight: Mapping[int, float] | str | None = None,
    rf_n_estimators: int = 100,
    rf_max_depth: int | None = None,
    rf_class_weight: Mapping[int, float] | str | None = None,
    mlp_params: Mapping[str, object] | None = None,
) -> Pipeline:
    """Construct a full pipeline with preprocessing and estimator."""
    estimator = create_estimator(
        model_type,
        seed,
        logreg_solver=logreg_solver,
        logreg_max_iter=logreg_max_iter,
        logreg_class_weight=logreg_class_weight,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        rf_class_weight=rf_class_weight,
        mlp_params=mlp_params,
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


def supported_models() -> Iterable[str]:
    return [MODEL_LDA, MODEL_LOGREG, MODEL_RF, MODEL_MLP, MODEL_FP_OPTIMIZED_MLP]

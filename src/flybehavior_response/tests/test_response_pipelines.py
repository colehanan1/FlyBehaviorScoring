from __future__ import annotations

from pathlib import Path

import json
import numpy as np
import pandas as pd
import pytest

from flybehavior_response.features import build_column_transformer, validate_features
from flybehavior_response.io import LABEL_COLUMN, LABEL_INTENSITY_COLUMN, load_and_merge
from flybehavior_response.modeling import (
    MODEL_LDA,
    MODEL_LOGREG,
    MODEL_MLP,
    build_model_pipeline,
    normalise_mlp_params,
)
from flybehavior_response.train import train_models
from flybehavior_response.weights import expand_samples_by_weight


def _create_dataset(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "dataset": ["ds"] * 6,
            "fly": ["a", "b", "c", "d", "e", "f"],
            "fly_number": [1, 2, 3, 4, 5, 6],
            "trial_type": ["testing"] * 6,
            "trial_label": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "dir_val_0": rng.normal(size=6),
            "dir_val_1": rng.normal(size=6),
            "dir_val_2": rng.normal(size=6),
            "AUC-During": rng.normal(size=6),
            "TimeToPeak-During": rng.normal(loc=5, scale=1, size=6),
            "Peak-Value": rng.normal(loc=1, scale=0.5, size=6),
        }
    )
    labels = pd.DataFrame(
        {
            "dataset": ["ds"] * 6,
            "fly": ["a", "b", "c", "d", "e", "f"],
            "fly_number": [1, 2, 3, 4, 5, 6],
            "trial_type": ["testing"] * 6,
            "trial_label": ["t1", "t2", "t3", "t4", "t5", "t6"],
            LABEL_COLUMN: [0, 1, 0, 2, 0, 5],
        }
    )
    data_path = tmp_path / "data.csv"
    labels_path = tmp_path / "labels.csv"
    data.to_csv(data_path, index=False)
    labels.to_csv(labels_path, index=False)
    return data_path, labels_path


def test_model_pipelines_fit(tmp_path: Path) -> None:
    data_path, labels_path = _create_dataset(tmp_path)
    dataset = load_and_merge(data_path, labels_path)
    features = validate_features(["AUC-During", "TimeToPeak-During", "Peak-Value"], dataset.feature_columns)
    preprocessor = build_column_transformer(
        dataset.trace_columns,
        dataset.feature_columns,
        features,
        use_raw_pca=True,
        n_pcs=2,
        seed=42,
    )
    X = dataset.frame.drop(columns=[LABEL_COLUMN, LABEL_INTENSITY_COLUMN])
    y = dataset.frame[LABEL_COLUMN].astype(int)
    lda_pipeline = build_model_pipeline(preprocessor, model_type=MODEL_LDA, seed=42)
    X_expanded, y_expanded = expand_samples_by_weight(X, y, dataset.sample_weights)
    lda_pipeline.fit(X_expanded, y_expanded)
    assert lda_pipeline.predict(X).shape == (6,)
    logreg_pipeline = build_model_pipeline(preprocessor, model_type=MODEL_LOGREG, seed=42)
    logreg_pipeline.fit(X, y, model__sample_weight=dataset.sample_weights.to_numpy())
    proba = logreg_pipeline.predict_proba(X)
    assert proba.shape == (6, 2)

    liblinear_pipeline = build_model_pipeline(
        preprocessor,
        model_type=MODEL_LOGREG,
        seed=0,
        logreg_solver="liblinear",
        logreg_max_iter=200,
    )
    liblinear_pipeline.fit(X, y, model__sample_weight=dataset.sample_weights.to_numpy())
    assert liblinear_pipeline.predict(X).shape == (6,)


def test_build_model_pipeline_applies_mlp_params(tmp_path: Path) -> None:
    data_path, labels_path = _create_dataset(tmp_path)
    dataset = load_and_merge(data_path, labels_path)
    features = validate_features(["AUC-During", "TimeToPeak-During", "Peak-Value"], dataset.feature_columns)
    preprocessor = build_column_transformer(
        dataset.trace_columns,
        dataset.feature_columns,
        features,
        use_raw_pca=True,
        n_pcs=2,
        seed=42,
    )

    params = normalise_mlp_params(
        {
            "n_components": 2,
            "alpha": 0.001,
            "batch_size": 16,
            "learning_rate_init": 0.005,
            "hidden_layer_sizes": [128, 64],
        }
    )
    pipeline = build_model_pipeline(
        preprocessor,
        model_type=MODEL_MLP,
        seed=42,
        mlp_params=params,
    )
    model = pipeline.named_steps["model"]
    assert model.hidden_layer_sizes == params["hidden_layer_sizes"]
    assert model.alpha == params["alpha"]
    assert model.batch_size == params["batch_size"]
    assert model.learning_rate_init == params["learning_rate_init"]
    assert model.early_stopping is True


def test_normalise_mlp_params_handles_two_layer_architecture() -> None:
    params = normalise_mlp_params(
        {
            "n_components": 12,
            "alpha": 0.0003,
            "batch_size": 64,
            "learning_rate_init": 0.0015,
            "architecture": "two_layer",
            "h1": 512,
            "h2": 256,
        }
    )

    assert params["hidden_layer_sizes"] == (512, 256)
    assert params["h1"] == 512
    assert params["h2"] == 256
    assert params["layer_config"] == "512_256"


def test_normalise_mlp_params_preserves_selected_features() -> None:
    params = normalise_mlp_params(
        {
            "n_components": 20,
            "alpha": 0.0001,
            "batch_size": 32,
            "learning_rate_init": 0.001,
            "hidden_layer_sizes": [256],
            "selected_features": ["AUC-During", "Peak-Value"],
        }
    )

    assert params["selected_features"] == ("AUC-During", "Peak-Value")


def test_train_models_returns_metrics(tmp_path: Path) -> None:
    data_path, labels_path = _create_dataset(tmp_path)
    metrics = train_models(
        data_csv=data_path,
        labels_csv=labels_path,
        features=["AUC-During", "TimeToPeak-During", "Peak-Value"],
        include_traces=True,
        use_raw_pca=True,
        n_pcs=2,
        models=["both"],
        artifacts_dir=tmp_path,
        cv=0,
        seed=42,
        verbose=False,
        dry_run=True,
    )
    assert set(metrics["models"].keys()) == {MODEL_LDA, MODEL_LOGREG}
    assert "accuracy" in metrics["models"][MODEL_LDA]
    assert "test" in metrics["models"][MODEL_LDA]
    assert "cross_validation" not in metrics["models"][MODEL_LDA]
    assert "weighted" in metrics["models"][MODEL_LOGREG]


def test_train_models_raises_on_missing_features(tmp_path: Path) -> None:
    data_path, labels_path = _create_dataset(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        train_models(
            data_csv=data_path,
            labels_csv=labels_path,
            features=["AUC-During", "Not-A-Feature"],
            include_traces=True,
            use_raw_pca=True,
            n_pcs=2,
            models=[MODEL_LOGREG],
            artifacts_dir=tmp_path,
            cv=0,
            seed=42,
            verbose=False,
            dry_run=True,
        )
    assert "Not-A-Feature" in str(excinfo.value)


def test_train_models_applies_optuna_selected_features(tmp_path: Path) -> None:
    data_path, labels_path = _create_dataset(tmp_path)
    data_df = pd.read_csv(data_path)
    labels_df = pd.read_csv(labels_path)
    expanded_data = [data_df]
    expanded_labels = [labels_df]
    for idx in range(4):
        suffix = f"_rep{idx}"
        clone_data = data_df.copy()
        clone_labels = labels_df.copy()
        clone_data["trial_label"] = clone_data["trial_label"] + suffix
        clone_labels["trial_label"] = clone_labels["trial_label"] + suffix
        expanded_data.append(clone_data)
        expanded_labels.append(clone_labels)
    pd.concat(expanded_data, ignore_index=True).to_csv(data_path, index=False)
    pd.concat(expanded_labels, ignore_index=True).to_csv(labels_path, index=False)

    best_params = normalise_mlp_params(
        {
            "n_components": 2,
            "alpha": 0.001,
            "batch_size": 16,
            "learning_rate_init": 0.005,
            "hidden_layer_sizes": [128],
        }
    )
    best_params["selected_features"] = ("AUC-During", "Peak-Value")

    metrics = train_models(
        data_csv=data_path,
        labels_csv=labels_path,
        features=["AUC-During", "TimeToPeak-During", "Peak-Value"],
        include_traces=True,
        use_raw_pca=True,
        n_pcs=2,
        models=[MODEL_MLP],
        artifacts_dir=tmp_path,
        cv=0,
        seed=42,
        verbose=False,
        dry_run=True,
        mlp_params=best_params,
    )

    assert MODEL_MLP in metrics["models"]


def test_train_models_writes_prediction_csv(tmp_path: Path) -> None:
    data_path, labels_path = _create_dataset(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    metrics = train_models(
        data_csv=data_path,
        labels_csv=labels_path,
        features=["AUC-During", "TimeToPeak-During", "Peak-Value"],
        include_traces=True,
        use_raw_pca=True,
        n_pcs=2,
        models=[MODEL_LDA, MODEL_LOGREG, MODEL_MLP],
        artifacts_dir=artifacts_dir,
        cv=0,
        seed=0,
        verbose=False,
        dry_run=False,
    )

    assert set(metrics["models"].keys()) == {MODEL_LDA, MODEL_LOGREG, MODEL_MLP}

    run_dirs = [path for path in artifacts_dir.iterdir() if path.is_dir()]
    assert run_dirs, "Expected at least one run directory with artifacts"
    latest_run = max(run_dirs, key=lambda path: path.stat().st_mtime)

    expected_files = {
        latest_run / f"predictions_{MODEL_LDA}_train.csv",
        latest_run / f"predictions_{MODEL_LDA}_test.csv",
        latest_run / f"predictions_{MODEL_LOGREG}_train.csv",
        latest_run / f"predictions_{MODEL_LOGREG}_test.csv",
        latest_run / f"predictions_{MODEL_MLP}_train.csv",
        latest_run / f"predictions_{MODEL_MLP}_test.csv",
    }
    for path in expected_files:
        assert path.exists(), f"Missing predictions export: {path}"

    sample = pd.read_csv(latest_run / f"predictions_{MODEL_MLP}_test.csv")
    assert {"model", "split", "predicted_label", "correct"}.issubset(sample.columns)

    manifest_path = latest_run / "split_manifest.csv"
    assert manifest_path.exists(), "Expected split_manifest.csv in artifacts"
    manifest = pd.read_csv(manifest_path)
    assert set(manifest["split"]) == {"train", "test"}
    train_ids = set(manifest.loc[manifest["split"] == "train", "fly"])
    test_ids = set(manifest.loc[manifest["split"] == "test", "fly"])
    assert train_ids.isdisjoint(test_ids), "Detected group leakage between train and test splits"


def test_train_models_serialises_mlp_params(tmp_path: Path) -> None:
    data_path, labels_path = _create_dataset(tmp_path)
    data_df = pd.read_csv(data_path)
    labels_df = pd.read_csv(labels_path)
    augmented_data = [data_df]
    augmented_labels = [labels_df]
    for idx in range(3):
        suffix = f"_dup{idx}"
        duplicated_data = data_df.copy()
        duplicated_labels = labels_df.copy()
        duplicated_data["trial_label"] = duplicated_data["trial_label"] + suffix
        duplicated_labels["trial_label"] = duplicated_labels["trial_label"] + suffix
        augmented_data.append(duplicated_data)
        augmented_labels.append(duplicated_labels)
    pd.concat(augmented_data, ignore_index=True).to_csv(data_path, index=False)
    pd.concat(augmented_labels, ignore_index=True).to_csv(labels_path, index=False)

    best_params = normalise_mlp_params(
        {
            "n_components": 3,
            "alpha": 0.0005,
            "batch_size": 32,
            "learning_rate_init": 0.001,
            "hidden_layer_sizes": [128, 32],
        }
    )

    artifacts_dir = tmp_path / "artifacts"
    metrics = train_models(
        data_csv=data_path,
        labels_csv=labels_path,
        features=["AUC-During", "TimeToPeak-During", "Peak-Value"],
        include_traces=True,
        use_raw_pca=False,
        n_pcs=2,
        models=[MODEL_MLP],
        artifacts_dir=artifacts_dir,
        cv=0,
        seed=0,
        verbose=False,
        dry_run=False,
        mlp_params=best_params,
    )

    assert MODEL_MLP in metrics["models"]

    run_dirs = [path for path in artifacts_dir.iterdir() if path.is_dir()]
    assert run_dirs, "Expected artifacts directory to be created"
    latest = max(run_dirs, key=lambda path: path.stat().st_mtime)
    config_path = latest / "config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert config["use_raw_pca"] is True
    assert config["n_pcs"] == 3
    assert config["mlp_params"]["n_components"] == 3
    assert config["mlp_params"]["hidden_layer_sizes"] == [128, 32]
    assert config["mlp_params"]["batch_size"] == 32


def test_train_models_geometry_without_traces(tmp_path: Path) -> None:
    trials = [
        ("f1", "t1", 0),
        ("f2", "t2", 5),
        ("f3", "t3", 0),
        ("f4", "t4", 5),
        ("f5", "t5", 0),
        ("f6", "t6", 5),
    ]
    frame_records = []
    for idx, (fly, trial_label, score) in enumerate(trials, start=1):
        for frame_idx in range(4):
            frame_records.append(
                {
                    "dataset": "d",
                    "fly": fly,
                    "fly_number": idx,
                    "trial_type": "testing",
                    "trial_label": trial_label,
                    "frame": frame_idx,
                    "eye_x": 0.1 * (frame_idx + idx),
                    "eye_y": 0.05 * (frame_idx + idx),
                    "prob_x": 0.2 * (idx - frame_idx * 0.1),
                    "prob_y": 0.15 * (idx + frame_idx * 0.2),
                }
            )
    frames = pd.DataFrame(frame_records)
    labels = pd.DataFrame(
        {
            "dataset": ["d"] * len(trials),
            "fly": [fly for fly, _, _ in trials],
            "fly_number": list(range(1, len(trials) + 1)),
            "trial_type": ["testing"] * len(trials),
            "trial_label": [trial for _, trial, _ in trials],
            LABEL_COLUMN: [score for _, _, score in trials],
        }
    )
    frames_path = tmp_path / "geom.csv"
    labels_path = tmp_path / "labels.csv"
    frames.to_csv(frames_path, index=False)
    labels.to_csv(labels_path, index=False)

    metrics = train_models(
        data_csv=None,
        labels_csv=labels_path,
        features=["eye_x_mean", "prob_x_mean"],
        include_traces=False,
        use_raw_pca=True,
        n_pcs=2,
        models=[MODEL_LOGREG],
        artifacts_dir=tmp_path,
        cv=0,
        seed=1,
        verbose=False,
        dry_run=True,
        geometry_source=frames_path,
        geom_granularity="trial",
        group_override="none",
        test_size=0.5,
    )

    assert set(metrics["models"].keys()) == {MODEL_LOGREG}

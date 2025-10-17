from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flybehavior_response.features import build_column_transformer, validate_features
from flybehavior_response.io import LABEL_COLUMN, LABEL_INTENSITY_COLUMN, load_and_merge
from flybehavior_response.modeling import (
    MODEL_LDA,
    MODEL_LOGREG,
    MODEL_MLP,
    MODEL_MLP_ADAM,
    build_model_pipeline,
    supported_models,
)
from flybehavior_response.sample_weighted_lda import SampleWeightedLDA
from flybehavior_response.sample_weighted_mlp import SampleWeightedMLPClassifier
from flybehavior_response.train import train_models


def _create_dataset(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "fly": ["a", "b", "c", "d", "e", "f"],
            "fly_number": [1, 2, 3, 4, 5, 6],
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
            "fly": ["a", "b", "c", "d", "e", "f"],
            "fly_number": [1, 2, 3, 4, 5, 6],
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
    lda_pipeline.fit(
        X,
        y,
        model__sample_weight=dataset.sample_weights.to_numpy(),
    )
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


def test_sample_weighted_mlp_supports_sample_weight() -> None:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    weights = np.linspace(0.5, 2.0, num=60)
    clf = SampleWeightedMLPClassifier(
        hidden_layer_sizes=(16,),
        learning_rate=5e-3,
        max_iter=300,
        batch_size=32,
        random_state=0,
    )
    clf.fit(X, y, sample_weight=weights)
    proba = clf.predict_proba(X)
    assert proba.shape == (60, 2)
    preds = clf.predict(X)
    assert preds.shape == (60,)


def test_sample_weighted_lda_supports_sample_weight() -> None:
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [2.0, 2.1],
            [2.2, 2.0],
        ]
    )
    y = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 2.0, 1.0, 3.0])
    clf = SampleWeightedLDA(reg=1e-5)
    clf.fit(X, y, sample_weight=weights)
    probs = clf.predict_proba(X)
    assert probs.shape == (4, 2)
    preds = clf.predict(X)
    assert np.array_equal(preds, np.array([0, 0, 1, 1]))


def test_train_models_returns_metrics(tmp_path: Path) -> None:
    data_path, labels_path = _create_dataset(tmp_path)
    metrics = train_models(
        data_csv=data_path,
        labels_csv=labels_path,
        features=["AUC-During", "TimeToPeak-During", "Peak-Value"],
        use_raw_pca=True,
        n_pcs=2,
        models=["both"],
        artifacts_dir=tmp_path,
        cv=2,
        seed=42,
        verbose=False,
        dry_run=True,
    )
    assert set(metrics["models"].keys()) == {MODEL_LDA, MODEL_LOGREG}
    assert "accuracy" in metrics["models"][MODEL_LDA]
    assert "test" in metrics["models"][MODEL_LDA]
    assert "cross_validation" in metrics["models"][MODEL_LDA]
    assert "weighted" in metrics["models"][MODEL_LOGREG]


def test_train_models_writes_prediction_csv(tmp_path: Path) -> None:
    data_path, labels_path = _create_dataset(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    metrics = train_models(
        data_csv=data_path,
        labels_csv=labels_path,
        features=["AUC-During", "TimeToPeak-During", "Peak-Value"],
        use_raw_pca=True,
        n_pcs=2,
        models=list(supported_models()),
        artifacts_dir=artifacts_dir,
        cv=0,
        seed=0,
        verbose=False,
        dry_run=False,
    )

    assert set(metrics["models"].keys()) == set(supported_models())

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
        latest_run / f"predictions_{MODEL_MLP_ADAM}_train.csv",
        latest_run / f"predictions_{MODEL_MLP_ADAM}_test.csv",
    }
    for path in expected_files:
        assert path.exists(), f"Missing predictions export: {path}"

    sample = pd.read_csv(latest_run / f"predictions_{MODEL_MLP}_test.csv")
    assert {"model", "split", "predicted_label", "correct"}.issubset(sample.columns)

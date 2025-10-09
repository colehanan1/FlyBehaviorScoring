"""End-to-end training entrypoint for the ML behavior models."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .data_join import join_features
from .features_from_envelope import assemble_envelope_df
from .modeling import run_cross_validation, run_train_test_splits, train_full_models

DATA_DIR = Path("data")
ENVELOPE_PATH = DATA_DIR / "envelope_matrix_float16.npy"
CODE_MAP_PATH = DATA_DIR / "code_maps.json"
OUTPUT_DIR = Path("outputs/ml")
SCORE_PATH = DATA_DIR / "scoring_results_opto_new.csv"
CLUSTER_PATH = DATA_DIR / "trial_clusters_reaction_clusters.csv"
REPORT_PATH = DATA_DIR / "report_reaction_clusters.csv"
METRICS_BINARY_PATH = OUTPUT_DIR / "metrics_binary.txt"
METRICS_REGRESSION_PATH = OUTPUT_DIR / "metrics_regression.txt"
METRICS_INFERRED_PATH = OUTPUT_DIR / "metrics_inferred.txt"
METRICS_HOLDOUT_PATH = OUTPUT_DIR / "metrics_holdout.txt"
TRAIN_LOG_PATH = OUTPUT_DIR / "train.log"
HOLDOUT_EPOCHS = 5
HOLDOUT_TEST_SIZE = 0.2


np.random.seed(42)


def _setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    handlers = []
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    file_handler = logging.FileHandler(TRAIN_LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    for handler in handlers:
        root_logger.addHandler(handler)


def _write_metrics(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> None:
    _setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting ML training pipeline")
    if not ENVELOPE_PATH.exists() or not CODE_MAP_PATH.exists():
        raise FileNotFoundError("Required envelope inputs are missing; aborting")

    envelope_df = assemble_envelope_df(str(ENVELOPE_PATH), str(CODE_MAP_PATH))
    if envelope_df.filter(like="dir_val_").empty:
        raise AssertionError("No dir_val_* columns found after assembly")
    logger.info("Envelope DataFrame columns: %d", envelope_df.shape[1])

    merged_df, y_bin, y_reg = join_features(
        envelope_df,
        scoring_path=str(SCORE_PATH),
        clusters_path=str(CLUSTER_PATH),
        report_path=str(REPORT_PATH),
    )

    if merged_df["trial_key"].duplicated().any():
        raise AssertionError("Duplicate trial keys detected after join")

    unique_flies = merged_df["fly"].unique()
    if unique_flies.size == 0:
        raise AssertionError("No flies available for cross-validation")
    logger.info("Found %d unique flies", unique_flies.size)

    holdout_results = run_train_test_splits(
        merged_df,
        y_bin,
        y_reg,
        OUTPUT_DIR,
        n_epochs=HOLDOUT_EPOCHS,
        test_size=HOLDOUT_TEST_SIZE,
    )

    if holdout_results["epoch_metrics"].empty:
        logger.warning("Hold-out evaluation produced no metrics")
        holdout_lines = ["No hold-out metrics available"]
    else:
        mean_metrics = holdout_results["mean_metrics"]
        std_metrics = holdout_results["std_metrics"]
        holdout_lines = [
            f"Epochs: {HOLDOUT_EPOCHS}",
            f"Split: train={1 - HOLDOUT_TEST_SIZE:.1%} test={HOLDOUT_TEST_SIZE:.1%}",
        ]
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            if metric in mean_metrics:
                holdout_lines.append(
                    f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics.get(metric, float('nan')):.4f}"
                )
        agg_cm = holdout_results["aggregated_confusion"]
        holdout_lines.append(
            "Summed confusion matrix: "
            f"[[{agg_cm[0,0]}, {agg_cm[0,1]}], [{agg_cm[1,0]}, {agg_cm[1,1]}]]"
        )
        holdout_lines.append(
            "Hold-out metrics CSV: holdout_epoch_metrics.csv"
        )
        holdout_lines.append(
            "Hold-out predictions CSV: holdout_predictions.csv"
        )
    _write_metrics(METRICS_HOLDOUT_PATH, holdout_lines)

    cv_results = run_cross_validation(merged_df, y_bin, y_reg, OUTPUT_DIR)

    metrics_lines = [
        f"Accuracy: {cv_results['accuracy']:.4f}",
        f"Precision: {cv_results['precision']:.4f}",
        f"Recall: {cv_results['recall']:.4f}",
        f"F1: {cv_results['f1']:.4f}",
        f"Confusion matrix: {cv_results['confusion_matrix'].tolist()}",
    ]
    _write_metrics(METRICS_BINARY_PATH, metrics_lines)

    reg_metrics = cv_results["regression"]
    reg_lines = [
        f"R2: {reg_metrics['r2']}",
        f"RMSE: {reg_metrics['rmse']}",
        f"Spearman: {reg_metrics['spearman']}",
    ]
    _write_metrics(METRICS_REGRESSION_PATH, reg_lines)

    inf_metrics = cv_results["inferred"]
    inferred_lines = [
        f"RMSE: {inf_metrics['rmse']}",
        f"Spearman: {inf_metrics['spearman']}",
        "Inferred score distribution:"
    ]

    try:
        preds = cv_results["cv_predictions"]["y_inf_pred"]
        counts = preds.value_counts().sort_index()
        for score, count in counts.items():
            inferred_lines.append(f"  Score {score}: {count}")
    except Exception as exc:
        logger.warning("Failed to summarize inferred score distribution: %s", exc)
    _write_metrics(METRICS_INFERRED_PATH, inferred_lines)

    final_models = train_full_models(
        merged_df,
        y_bin,
        y_reg,
        OUTPUT_DIR,
    )

    logger.info("Training complete; artifacts written to %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()

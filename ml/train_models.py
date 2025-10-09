"""End-to-end training entrypoint for the ML behavior models."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .data_join import join_features
from .features_from_envelope import assemble_envelope_df
from .modeling import run_cross_validation, train_full_models

DATA_DIR = Path("/home/ramanlab/Documents/cole/Data/Opto/Combined/matrix/")
ENVELOPE_PATH = Path("/home/ramanlab/Documents/cole/Data/Opto/Combined/matrix/") / "envelope_matrix_float16.npy"
CODE_MAP_PATH = Path("/home/ramanlab/Documents/cole/Data/Opto/Combined/matrix/") / "code_maps.json"
OUTPUT_DIR = Path("outputs/ml")
SCORE_PATH = Path("/home/ramanlab/Documents/cole/model/FlyBehaviorPER/") / "scoring_results_opto_new.csv"
CLUSTER_PATH = Path("/home/ramanlab/PycharmProjects/FlyBehaviorScoring/outputs/unsup/20251003_155149/") / "trial_clusters_reaction_clusters.csv"
REPORT_PATH = Path("/home/ramanlab/PycharmProjects/FlyBehaviorScoring/outputs/unsup/20251003_155149/") / "report_reaction_clusters.csv"
METRICS_BINARY_PATH = OUTPUT_DIR / "metrics_binary.txt"
METRICS_REGRESSION_PATH = OUTPUT_DIR / "metrics_regression.txt"
METRICS_INFERRED_PATH = OUTPUT_DIR / "metrics_inferred.txt"
TRAIN_LOG_PATH = OUTPUT_DIR / "train.log"


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

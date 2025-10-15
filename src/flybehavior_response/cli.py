"""Command line interface for flybehavior_response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from .evaluate import evaluate_models, load_pipeline, save_metrics
from .features import DEFAULT_FEATURES, parse_feature_list
from .io import LABEL_COLUMN, LABEL_INTENSITY_COLUMN, load_and_merge, write_parquet
from .logging_utils import get_logger, set_global_logging
from .modeling import supported_models
from .train import train_models
from .visualize import generate_visuals

DEFAULT_ARTIFACTS_DIR = Path("./artifacts")
DEFAULT_PLOTS_DIR = DEFAULT_ARTIFACTS_DIR / "plots"


def _resolve_run_dir(artifacts_dir: Path, run_dir: Path | None) -> Path:
    if run_dir:
        if not run_dir.exists():
            raise FileNotFoundError(f"Specified run directory does not exist: {run_dir}")
        return run_dir
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")
    candidates: List[Tuple[float, Path]] = []
    for candidate in artifacts_dir.iterdir():
        if not candidate.is_dir():
            continue
        has_model = any((candidate / f"model_{name}.joblib").exists() for name in supported_models())
        if not has_model:
            continue
        candidates.append((candidate.stat().st_mtime, candidate))
    if not candidates:
        raise FileNotFoundError(
            f"No trained models found under {artifacts_dir}. Provide --run-dir to select a specific training output."
        )
    return max(candidates, key=lambda item: item[0])[1]


def _parse_models(value: str | None) -> List[str]:
    if value is None:
        return list(supported_models())
    if value == "all":
        return list(supported_models())
    if value == "both":
        return ["lda", "logreg"]
    if value not in supported_models():
        raise ValueError(f"Unsupported model choice: {value}")
    return [value]


def _configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fly behavior response modeling CLI")

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--data-csv", type=Path, help="Path to data CSV")
    common_parser.add_argument("--labels-csv", type=Path, help="Path to labels CSV")
    common_parser.add_argument(
        "--features",
        type=str,
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated list of engineered features to include",
    )
    common_parser.add_argument(
        "--include-auc-before",
        action="store_true",
        help="Include AUC-Before feature in addition to selected features",
    )
    common_parser.add_argument(
        "--use-raw-pca",
        dest="use_raw_pca",
        action="store_true",
        default=True,
        help="Include PCA on raw trace columns (default: enabled)",
    )
    common_parser.add_argument(
        "--no-use-raw-pca",
        dest="use_raw_pca",
        action="store_false",
        help="Disable PCA on raw trace columns",
    )
    common_parser.add_argument("--n-pcs", type=int, default=5, help="Number of principal components to use for traces")
    common_parser.add_argument(
        "--model",
        type=str,
        choices=["lda", "logreg", "mlp", "both", "all"],
        default="all",
        help="Model to train/evaluate ('all' runs every supported model; 'both' keeps LDA+logreg")",
    )
    common_parser.add_argument("--cv", type=int, default=0, help="Number of stratified folds for cross-validation")
    common_parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Directory to store generated plots",
    )
    common_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory to store artifacts",
    )
    common_parser.add_argument("--run-dir", type=Path, help="Specific run directory to use for evaluation/visualization")
    common_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    common_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    common_parser.add_argument("--dry-run", action="store_true", help="Execute without writing artifacts")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "prepare",
        parents=[common_parser],
        help="Validate inputs and create merged parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser = subparsers.add_parser(
        "train",
        parents=[common_parser],
        help="Train model pipelines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--logreg-solver",
        type=str,
        choices=["lbfgs", "liblinear", "saga"],
        default="lbfgs",
        help="Solver to use for logistic regression (iterative training)",
    )
    train_parser.add_argument(
        "--logreg-max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for logistic regression; increase if convergence warnings occur",
    )
    subparsers.add_parser(
        "eval",
        parents=[common_parser],
        help="Evaluate trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers.add_parser(
        "viz",
        parents=[common_parser],
        help="Generate visualizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    predict_parser = subparsers.add_parser(
        "predict",
        parents=[common_parser],
        help="Score new data with a trained pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    predict_parser.add_argument("--model-path", type=Path, required=True, help="Path to trained model joblib")
    predict_parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR / "predictions.csv",
        help="Path to write predictions CSV",
    )

    return parser


def _handle_prepare(args: argparse.Namespace) -> None:
    if not args.data_csv or not args.labels_csv:
        raise ValueError("--data-csv and --labels-csv are required for prepare")
    logger = get_logger("prepare", verbose=args.verbose)
    dataset = load_and_merge(args.data_csv, args.labels_csv, logger_name="prepare")
    balance = dataset.frame[LABEL_COLUMN].astype(int).value_counts(normalize=True).to_dict()
    logger.info("Class balance: %s", balance)
    if args.dry_run:
        logger.info("Dry run enabled; not writing parquet")
        return
    parquet_path = args.artifacts_dir / "merged.parquet"
    write_parquet(dataset, parquet_path)
    logger.info("Wrote merged parquet to %s", parquet_path)


def _handle_train(args: argparse.Namespace) -> None:
    if not args.data_csv or not args.labels_csv:
        raise ValueError("--data-csv and --labels-csv are required for train")
    features = parse_feature_list(args.features, args.include_auc_before)
    metrics = train_models(
        data_csv=args.data_csv,
        labels_csv=args.labels_csv,
        features=features,
        use_raw_pca=args.use_raw_pca,
        n_pcs=args.n_pcs,
        models=_parse_models(args.model),
        artifacts_dir=args.artifacts_dir,
        cv=args.cv,
        seed=args.seed,
        verbose=args.verbose,
        dry_run=args.dry_run,
        logreg_solver=args.logreg_solver,
        logreg_max_iter=args.logreg_max_iter,
    )
    logger = get_logger("train", verbose=args.verbose)
    logger.info("Training metrics: %s", json.dumps(metrics))


def _load_models(run_dir: Path) -> dict[str, object]:
    models = {}
    for name in supported_models():
        path = run_dir / f"model_{name}.joblib"
        if path.exists():
            models[name] = load_pipeline(path)
    if not models:
        raise FileNotFoundError(f"No models found in {run_dir}")
    return models


def _handle_eval(args: argparse.Namespace) -> None:
    if not args.data_csv or not args.labels_csv:
        raise ValueError("--data-csv and --labels-csv are required for eval")
    run_dir = _resolve_run_dir(args.artifacts_dir, args.run_dir)
    logger = get_logger("eval", verbose=args.verbose)
    logger.info("Using run directory: %s", run_dir)
    dataset = load_and_merge(args.data_csv, args.labels_csv, logger_name="eval")
    models = _load_models(run_dir)
    drop_cols = [LABEL_COLUMN]
    if LABEL_INTENSITY_COLUMN in dataset.frame.columns:
        drop_cols.append(LABEL_INTENSITY_COLUMN)
    features = dataset.frame.drop(columns=drop_cols)
    metrics = evaluate_models(
        models,
        features,
        dataset.frame[LABEL_COLUMN].astype(int),
        sample_weight=dataset.sample_weights,
    )
    payload = {"models": metrics}
    logger.info("Evaluation metrics: %s", json.dumps(payload))
    if args.dry_run:
        logger.info("Dry run enabled; metrics not written")
        return
    save_metrics(payload, run_dir / "metrics.json")
    logger.info("Metrics saved to %s", run_dir / "metrics.json")


def _handle_viz(args: argparse.Namespace) -> None:
    if not args.data_csv or not args.labels_csv:
        raise ValueError("--data-csv and --labels-csv are required for viz")
    if args.dry_run:
        logger = get_logger("viz", verbose=args.verbose)
        logger.info("Dry run enabled; skipping visualization generation")
        return
    run_dir = _resolve_run_dir(args.artifacts_dir, args.run_dir)
    logger = get_logger("viz", verbose=args.verbose)
    logger.info("Using run directory: %s", run_dir)
    generate_visuals(
        data_csv=args.data_csv,
        labels_csv=args.labels_csv,
        run_dir=run_dir,
        seed=args.seed,
        output_dir=args.plots_dir,
        verbose=args.verbose,
    )


def _handle_predict(args: argparse.Namespace) -> None:
    if not args.data_csv:
        raise ValueError("--data-csv is required for predict")
    model = load_pipeline(args.model_path)
    data_df = pd.read_csv(args.data_csv)
    keys = [col for col in ["fly", "fly_number", "trial_label"] if col in data_df.columns]
    output = data_df[keys].copy() if keys else pd.DataFrame()
    output["prediction"] = model.predict(data_df)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(data_df)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            output["probability"] = proba[:, 1]
    if args.dry_run:
        logger = get_logger("predict", verbose=args.verbose)
        logger.info("Dry run enabled; predictions not written")
        return
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    logger = get_logger("predict", verbose=args.verbose)
    logger.info("Predictions written to %s", args.output_csv)


def main(argv: list[str] | None = None) -> None:
    parser = _configure_parser()
    args = parser.parse_args(argv)
    set_global_logging(verbose=args.verbose)

    if args.command == "prepare":
        _handle_prepare(args)
    elif args.command == "train":
        _handle_train(args)
    elif args.command == "eval":
        _handle_eval(args)
    elif args.command == "viz":
        _handle_viz(args)
    elif args.command == "predict":
        _handle_predict(args)
    else:
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()

"""Command line interface for flybehavior_response."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd
import typer

from .config import PipelineConfig
from .evaluate import evaluate_models, load_pipeline, save_metrics
from .features import DEFAULT_FEATURES, parse_feature_list
from .io import (
    DEFAULT_TRACE_PREFIXES,
    LABEL_COLUMN,
    LABEL_INTENSITY_COLUMN,
    load_and_merge,
    write_parquet,
)
from .logging_utils import get_logger, set_global_logging
from .modeling import supported_models
from .prepare_raw import (
    DEFAULT_OUTPUT_PATH as RAW_DEFAULT_OUTPUT_PATH,
    DEFAULT_PREFIXES as RAW_DEFAULT_PREFIXES,
    prepare_raw,
)
from .train import train_models
from .visualize import generate_visuals

DEFAULT_ARTIFACTS_DIR = Path("./artifacts")
DEFAULT_PLOTS_DIR = DEFAULT_ARTIFACTS_DIR / "plots"


prepare_raw_app = typer.Typer(add_completion=False)


@prepare_raw_app.callback(invoke_without_command=True, no_args_is_help=True)
def prepare_raw_cli(
    data_csv_arg: Optional[Path] = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Optional positional path to per-trial raw coordinate CSV",
    ),
    *,
    data_csv: Optional[Path] = typer.Option(
        None,
        "--data-csv",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to per-trial raw coordinate CSV",
    ),
    data_npy: Optional[Path] = typer.Option(
        None,
        "--data-npy",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to per-trial raw coordinate matrix (.npy)",
    ),
    matrix_meta: Optional[Path] = typer.Option(
        None,
        "--matrix-meta",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="JSON file describing the matrix layout and per-trial metadata",
    ),
    labels_csv: Path = typer.Option(
        ...,
        "--labels-csv",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to labels CSV",
    ),
    out: Path = typer.Option(
        RAW_DEFAULT_OUTPUT_PATH,
        "--out",
        help="Destination CSV for prepared coordinates",
    ),
    fps: int = typer.Option(40, "--fps", help="Frame rate (frames per second)"),
    odor_on_idx: int = typer.Option(1230, "--odor-on-idx", help="Index where odor stimulus begins"),
    odor_off_idx: int = typer.Option(2430, "--odor-off-idx", help="Index where odor stimulus ends"),
    truncate_before: int = typer.Option(
        0,
        "--truncate-before",
        help="Number of frames to keep before odor onset (0 keeps all)",
    ),
    truncate_after: int = typer.Option(
        0,
        "--truncate-after",
        help="Number of frames to keep after odor offset (0 keeps all)",
    ),
    series_prefixes: str = typer.Option(
        ",".join(RAW_DEFAULT_PREFIXES),
        "--series-prefixes",
        help="Comma-separated list of time-series prefixes to extract",
    ),
    compute_dir_val: bool = typer.Option(
        False,
        "--compute-dir-val/--no-compute-dir-val",
        help="Also compute dir_val distances between proboscis and eye coordinates",
    ),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Enable verbose logging"),
) -> None:
    if data_npy is not None:
        if data_csv is not None or data_csv_arg is not None:
            raise typer.BadParameter(
                "When using --data-npy, do not also supply a CSV path. Provide only the matrix and metadata JSON."
            )
        if matrix_meta is None:
            raise typer.BadParameter("--matrix-meta is required when using --data-npy inputs.")
    else:
        if matrix_meta is not None:
            raise typer.BadParameter("--matrix-meta is only valid together with --data-npy.")
        if data_csv is None:
            if data_csv_arg is None:
                raise typer.BadParameter(
                    "Provide --data-csv or a positional raw CSV path when invoking prepare-raw."
                )
            if data_csv_arg.suffix.lower() == ".npy":
                raise typer.BadParameter(
                    "Detected positional .npy input; re-run with --data-npy and provide --matrix-meta for metadata."
                )
            data_csv = data_csv_arg
        elif data_csv_arg is not None:
            raise typer.BadParameter(
                "Received raw CSV as both positional argument and --data-csv. Specify it only once."
            )

    prefixes = [item.strip() for item in series_prefixes.split(",") if item.strip()]
    if not prefixes:
        raise typer.BadParameter("Provide at least one series prefix.")
    set_global_logging(verbose=verbose)
    prepared_df = prepare_raw(
        data_csv=data_csv,
        data_npy=data_npy,
        matrix_meta=matrix_meta,
        labels_csv=labels_csv,
        out_path=out,
        fps=fps,
        odor_on_idx=odor_on_idx,
        odor_off_idx=odor_off_idx,
        truncate_before=truncate_before,
        truncate_after=truncate_after,
        series_prefixes=prefixes,
        compute_dir_val=compute_dir_val,
        verbose=verbose,
    )
    global_frames = int(prepared_df["total_frames"].iat[0]) if not prepared_df.empty else 0
    typer.echo(
        f"Prepared {len(prepared_df)} trials with {global_frames} frames per trial using prefixes {prefixes}. Output -> {out}"
    )


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


def _parse_series_prefixes(raw: str | None) -> List[str]:
    if raw is None:
        return list(DEFAULT_TRACE_PREFIXES)
    prefixes = [item.strip() for item in raw.split(",") if item.strip()]
    if not prefixes:
        raise ValueError("At least one series prefix must be provided.")
    return prefixes


def _select_trace_prefixes(
    args: argparse.Namespace, *, fallback: Sequence[str] | None = None
) -> List[str] | None:
    if getattr(args, "raw_series", False):
        return list(RAW_DEFAULT_PREFIXES)
    if args.series_prefixes is not None:
        return _parse_series_prefixes(args.series_prefixes)
    if fallback:
        return list(fallback)
    return None


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
        "--series-prefixes",
        type=str,
        default=None,
        help="Comma-separated list of time-series prefixes to load",
    )
    common_parser.add_argument(
        "--raw-series",
        action="store_true",
        help="Use the default raw coordinate prefixes (eye/proboscis channels)",
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
        help="Model to train/evaluate ('all' runs every supported model; 'both' keeps LDA+logreg)",
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
    prefixes = _select_trace_prefixes(args)
    dataset = load_and_merge(
        args.data_csv,
        args.labels_csv,
        logger_name="prepare",
        trace_prefixes=prefixes,
    )
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
    prefixes = _select_trace_prefixes(args)
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
        trace_prefixes=prefixes,
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
    config_prefixes: Sequence[str] | None = None
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = PipelineConfig.from_json(config_path)
        if config.trace_series_prefixes:
            config_prefixes = config.trace_series_prefixes
    prefixes = _select_trace_prefixes(args, fallback=config_prefixes)
    dataset = load_and_merge(
        args.data_csv,
        args.labels_csv,
        logger_name="eval",
        trace_prefixes=prefixes,
    )
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
    config_prefixes: Sequence[str] | None = None
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = PipelineConfig.from_json(config_path)
        if config.trace_series_prefixes:
            config_prefixes = config.trace_series_prefixes
    prefixes = _select_trace_prefixes(args, fallback=config_prefixes)
    generate_visuals(
        data_csv=args.data_csv,
        labels_csv=args.labels_csv,
        run_dir=run_dir,
        seed=args.seed,
        output_dir=args.plots_dir,
        verbose=args.verbose,
        trace_prefixes=prefixes,
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
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if raw_args and raw_args[0] == "prepare-raw":
        command_args = raw_args[1:]
        try:
            prepare_raw_app(
                prog_name="flybehavior-response prepare-raw",
                args=command_args,
                standalone_mode=False,
            )
        except SystemExit as exc:  # pragma: no cover - delegated to Typer
            if exc.code:
                raise
        return

    parser = _configure_parser()
    args = parser.parse_args(raw_args)
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

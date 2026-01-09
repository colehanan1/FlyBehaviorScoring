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
    MERGE_KEYS,
    aggregate_trials,
    RAW_TRACE_PREFIXES,
    TRACE_PATTERN,
    load_geom_frames,
    load_geometry_dataset,
    load_and_merge,
    load_dataset,
    write_parquet,
)
from .logging_utils import get_logger, set_global_logging
from .modeling import supported_models, normalise_mlp_params, MODEL_MLP
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


def _parse_column_list(raw: str | None) -> List[str] | None:
    if raw is None:
        return None
    columns = [item.strip() for item in raw.split(",") if item.strip()]
    return columns or None


def _parse_feature_selection(raw: str | None) -> List[str] | None:
    if raw is None:
        return None
    token = raw.strip()
    if not token:
        return None
    if token.startswith("@"):
        path = Path(token[1:]).expanduser()
        if not path.exists():
            raise ValueError(f"Geometry feature list file not found: {path}")
        features: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            features.append(entry)
        return features or None
    return [item.strip() for item in token.split(",") if item.strip()]


def _parse_stats_list(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _add_geometry_cli_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--geometry-frames",
        type=Path,
        help="Path to per-frame geometry data (CSV or parquet)",
    )
    parser.add_argument(
        "--geometry-trials",
        type=Path,
        help="Optional per-trial geometry summary CSV to merge with streamed aggregates",
    )
    parser.add_argument(
        "--geom-cache-parquet",
        type=Path,
        help="Optional parquet cache destination for streamed geometry data",
    )
    parser.add_argument(
        "--geom-use-cache",
        action="store_true",
        help="Load geometry frames from --geom-cache-parquet when it already exists",
    )
    parser.add_argument(
        "--geom-chunk-size",
        type=int,
        default=50000,
        help="Chunk size to use when streaming geometry inputs",
    )
    parser.add_argument(
        "--geom-columns",
        type=str,
        help="Comma-separated list of geometry columns to retain while streaming",
    )
    parser.add_argument(
        "--geom-feature-columns",
        type=str,
        help=(
            "Comma-separated geometry feature columns to retain for modeling. "
            "Prefix with '@' to load newline-delimited names from a file."
        ),
    )
    parser.add_argument(
        "--geom-frame-column",
        type=str,
        default="frame_idx",
        help="Frame index column used to validate geometry contiguity",
    )
    parser.add_argument(
        "--geom-granularity",
        choices=["trial", "frame"],
        default="trial",
        help="Granularity for geometry-derived datasets (trial aggregates or frame-level rows)",
    )
    parser.add_argument(
        "--geom-stats",
        type=str,
        default="mean,min,max",
        help="Comma-separated aggregation statistics when --geom-granularity=trial",
    )
    parser.add_argument(
        "--geom-normalize",
        choices=["none", "zscore", "minmax"],
        default="none",
        help="Normalization mode applied to geometry columns prior to modeling",
    )
    parser.set_defaults(geom_downcast=True, geom_drop_missing_labels=True)
    parser.add_argument(
        "--no-geom-downcast",
        dest="geom_downcast",
        action="store_false",
        help="Disable float32 downcasting for geometry columns",
    )
    parser.add_argument(
        "--geom-keep-missing-labels",
        dest="geom_drop_missing_labels",
        action="store_false",
        help="Retain rows without matching labels when streaming geometry data",
    )
    parser.add_argument(
        "--geom-drop-missing-labels",
        dest="geom_drop_missing_labels",
        action="store_true",
        help=argparse.SUPPRESS,
    )


def _extract_geometry_kwargs(args: argparse.Namespace) -> dict[str, object]:
    columns = _parse_column_list(getattr(args, "geom_columns", None))
    stats = _parse_stats_list(getattr(args, "geom_stats", None))
    feature_columns = _parse_feature_selection(getattr(args, "geom_feature_columns", None))
    return {
        "geometry_source": getattr(args, "geometry_frames", None),
        "geom_chunk_size": getattr(args, "geom_chunk_size", 100_000),
        "geom_columns": columns,
        "geom_cache_parquet": getattr(args, "geom_cache_parquet", None),
        "geom_use_cache": bool(getattr(args, "geom_use_cache", False)),
        "geom_frame_column": getattr(args, "geom_frame_column", "frame_idx"),
        "geom_stats": stats or None,
        "geom_granularity": getattr(args, "geom_granularity", "trial"),
        "geom_normalization": getattr(args, "geom_normalize", "none"),
        "geom_drop_missing_labels": getattr(args, "geom_drop_missing_labels", True),
        "geom_downcast": getattr(args, "geom_downcast", True),
        "geom_trial_summary": getattr(args, "geometry_trials", None),
        "geom_feature_columns": feature_columns,
    }


def _select_trace_prefixes(
    args: argparse.Namespace,
    *,
    fallback: Sequence[str] | None = None,
    force_no_raw: bool | None = None,
) -> List[str] | None:
    no_raw_flag = getattr(args, "no_raw", False)
    if force_no_raw is not None:
        no_raw_flag = force_no_raw
    if no_raw_flag:
        return []
    if getattr(args, "raw_series", False):
        return list(RAW_DEFAULT_PREFIXES)
    if args.series_prefixes is not None:
        return _parse_series_prefixes(args.series_prefixes)
    if fallback is not None:
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
        "--no-raw",
        action="store_true",
        help="Exclude raw trace columns from downstream modeling workflows",
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
    model_choices = list(supported_models()) + ["both", "all"]
    common_parser.add_argument(
        "--model",
        type=str,
        choices=model_choices,
        default="all",
        help=(
            "Model to train/evaluate ('all' runs every supported model; 'both' keeps LDA+logreg; "
            "'fp_optimized_mlp' minimises false positives with class-weighted training)"
        ),
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

    prepare_parser = subparsers.add_parser(
        "prepare",
        parents=[common_parser],
        help="Validate inputs and create merged parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prepare_parser.add_argument(
        "--cache-parquet",
        type=Path,
        help="Optional parquet cache destination for streamed geometry data",
    )
    prepare_parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load geometry frames from --cache-parquet if it already exists",
    )
    prepare_parser.add_argument(
        "--geom-chunk-size",
        type=int,
        default=50000,
        help="Chunk size to use when streaming geometry CSV inputs",
    )
    prepare_parser.add_argument(
        "--geom-columns",
        type=str,
        help="Comma-separated list of geometry columns to keep while streaming",
    )
    prepare_parser.add_argument(
        "--frame-column",
        type=str,
        default="frame_idx",
        help="Frame index column used to validate chunk contiguity",
    )
    prepare_parser.add_argument(
        "--aggregate-geometry",
        action="store_true",
        help="Aggregate streamed geometry into per-trial summaries",
    )
    prepare_parser.add_argument(
        "--aggregate-stats",
        type=str,
        default="mean,min,max",
        help="Comma-separated aggregation statistics for numeric geometry columns",
    )
    prepare_parser.add_argument(
        "--aggregate-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="File format for aggregated geometry outputs",
    )
    prepare_parser.set_defaults(drop_missing_labels=True)
    prepare_parser.add_argument(
        "--keep-missing-labels",
        dest="drop_missing_labels",
        action="store_false",
        help="Retain frames without matching labels and fail when the merge encounters them",
    )
    prepare_parser.add_argument(
        "--drop-missing-labels",
        dest="drop_missing_labels",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    train_parser = subparsers.add_parser(
        "train",
        parents=[common_parser],
        help="Train model pipelines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_geometry_cli_options(train_parser)
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
    train_parser.add_argument(
        "--group-column",
        type=str,
        default="fly",
        help="Column used for group-aware splits (default ensures fly-level leakage prevention)",
    )
    train_parser.add_argument(
        "--group-override",
        type=str,
        help="Override the group column; specify 'none' to disable group-based splitting",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for the held-out test split",
    )
    train_parser.add_argument(
        "--classification-mode",
        type=str,
        choices=["binary", "multiclass", "threshold-1", "threshold-2"],
        default="binary",
        help=(
            "Classification mode: 'binary' (0 vs 1-5, default), "
            "'multiclass' (preserve all 6 classes: 0-5), "
            "'threshold-1' (0-1 vs 2-5), "
            "'threshold-2' (0-2 vs 3-5)"
        ),
    )
    train_parser.add_argument(
        "--best-params-json",
        type=Path,
        help=(
            "Path to Optuna best_params.json generated by optuna_mlp_tuning.py. "
            "When provided, the hyperparameters are reused for the MLP model."
        ),
    )
    train_parser.add_argument(
        "--class-weights",
        type=str,
        help=(
            "Custom class weights for MLP models (e.g., '0:2.0,1:1.0'). "
            "Higher weight on class 0 reduces false positives. "
            "Default for fp_optimized_mlp is '0:1.0,1:2.0'."
        ),
    )
    train_parser.add_argument(
        "--logreg-class-weights",
        type=str,
        help=(
            "Custom class weights for logistic regression to reduce false negatives. "
            "Format: '0:1.0,1:2.0' or use 'balanced' for sklearn auto-balancing. "
            "Higher weight on class 1 increases sensitivity to positive class (responders). "
            "Example: '0:1.0,1:3.0' gives responders 3x weight to reduce false negatives."
        ),
    )
    train_parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=100,
        help="Number of trees in the Random Forest (default: 100)",
    )
    train_parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Maximum depth of trees in Random Forest (default: None = unlimited)",
    )
    train_parser.add_argument(
        "--rf-class-weights",
        type=str,
        help=(
            "Custom class weights for Random Forest to reduce false negatives. "
            "Format: '0:1.0,1:2.0' or use 'balanced' for sklearn auto-balancing. "
            "Higher weight on class 1 increases sensitivity to positive class (responders)."
        ),
    )
    eval_parser = subparsers.add_parser(
        "eval",
        parents=[common_parser],
        help="Evaluate trained models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_geometry_cli_options(eval_parser)
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
    _add_geometry_cli_options(predict_parser)
    predict_parser.add_argument("--model-path", type=Path, required=True, help="Path to trained model joblib")
    predict_parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR / "predictions.csv",
        help="Path to write predictions CSV",
    )
    predict_parser.add_argument("--fly", type=str, help="Filter predictions to a specific fly identifier")
    predict_parser.add_argument(
        "--fly-number",
        type=int,
        help="Filter predictions to a specific numeric fly identifier",
    )
    predict_parser.add_argument(
        "--trial-label",
        type=str,
        help="Filter predictions to a specific trial label (aliases legacy testing_trial)",
    )
    predict_parser.add_argument(
        "--testing-trial",
        type=str,
        help="Legacy alias for --trial-label when datasets expose a testing_trial column",
    )
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary classification (default: 0.5). Use higher values (e.g., 0.65) to reduce false positives.",
    )

    return parser


def _handle_prepare(args: argparse.Namespace) -> None:
    if not args.data_csv or not args.labels_csv:
        raise ValueError("--data-csv and --labels-csv are required for prepare")
    logger = get_logger("prepare", verbose=args.verbose)
    geometry_mode = bool(
        args.cache_parquet
        or args.use_cache
        or args.geom_columns
        or args.aggregate_geometry
    )

    if geometry_mode:
        columns = _parse_column_list(args.geom_columns)
        cache_path = args.cache_parquet
        use_cache_flag = bool(args.use_cache and cache_path and cache_path.exists())
        if args.dry_run and cache_path and not use_cache_flag:
            logger.info(
                "Dry run enabled; parquet cache will not be written to %s.",
                cache_path,
            )
            cache_path = None
        stats_tokens = [item.strip() for item in (args.aggregate_stats or "").split(",") if item.strip()]
        if args.aggregate_geometry and not stats_tokens:
            stats_tokens = ["mean", "min", "max"]

        stream = load_geom_frames(
            args.data_csv,
            chunk_size=args.geom_chunk_size,
            columns=columns,
            cache_parquet=cache_path,
            use_cache=use_cache_flag,
            labels_csv=args.labels_csv,
            frame_column=args.frame_column,
            drop_missing_labels=args.drop_missing_labels,
            logger=logger,
        )

        if args.aggregate_geometry:
            aggregates = aggregate_trials(
                stream,
                key_columns=MERGE_KEYS,
                frame_column=args.frame_column,
                stats=stats_tokens,
            )
            logger.info(
                "Aggregated %d trials from streamed geometry frames.",
                len(aggregates),
            )
            if args.dry_run:
                logger.info("Dry run enabled; not writing aggregated artifact.")
                return
            args.artifacts_dir.mkdir(parents=True, exist_ok=True)
            if args.aggregate_format == "parquet":
                out_path = args.artifacts_dir / "geometry_aggregates.parquet"
                try:
                    aggregates.to_parquet(out_path, compression="zstd", index=False)
                except ImportError as exc:  # pragma: no cover - depends on optional deps
                    raise RuntimeError(
                        "Parquet support requires the 'pyarrow' or 'fastparquet' optional dependency. "
                        "Re-run with --aggregate-format csv or install one of the engines."
                    ) from exc
            else:
                out_path = args.artifacts_dir / "geometry_aggregates.csv"
                aggregates.to_csv(out_path, index=False)
            logger.info("Wrote aggregated geometry %s to %s", args.aggregate_format, out_path)
            return

        # Exhaust the iterator to materialise cache/statistics without aggregation.
        for _ in stream:
            pass
        if args.dry_run:
            logger.info("Dry run complete; no artifacts were written.")
        elif cache_path:
            logger.info("Geometry parquet cache available at %s", cache_path)
        return

    prefixes = _select_trace_prefixes(args)
    include_traces = not getattr(args, "no_raw", False)
    dataset = load_and_merge(
        args.data_csv,
        args.labels_csv,
        logger_name="prepare",
        trace_prefixes=prefixes,
        include_trace_columns=include_traces,
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
    if not args.labels_csv:
        raise ValueError("--labels-csv is required for train")
    if args.data_csv is None and getattr(args, "geometry_frames", None) is None:
        raise ValueError("Provide --data-csv or --geometry-frames for train")
    if args.data_csv is not None and getattr(args, "geometry_frames", None) is not None:
        raise ValueError("Specify either --data-csv or --geometry-frames, not both")
    if getattr(args, "geometry_frames", None) is not None and args.geom_use_cache and not args.geom_cache_parquet:
        raise ValueError("--geom-use-cache requires --geom-cache-parquet when using geometry inputs")
    if getattr(args, "geometry_trials", None) is not None and getattr(args, "geometry_frames", None) is None:
        raise ValueError("--geometry-trials requires --geometry-frames")
    if getattr(args, "geom_feature_columns", None) and getattr(args, "geometry_frames", None) is None:
        raise ValueError("--geom-feature-columns requires --geometry-frames")

    logger = get_logger("train", verbose=args.verbose)
    geom_kwargs = _extract_geometry_kwargs(args)
    features = parse_feature_list(args.features, args.include_auc_before)
    prefixes = _select_trace_prefixes(args)
    include_traces = not getattr(args, "no_raw", False)
    geometry_source = geom_kwargs.pop("geometry_source")
    models_to_train = _parse_models(args.model)

    mlp_params = None
    use_raw_pca = args.use_raw_pca
    n_pcs = args.n_pcs

    if args.best_params_json is not None:
        best_params_path = args.best_params_json.expanduser()
        if not best_params_path.exists():
            raise FileNotFoundError(f"Best-params JSON not found: {best_params_path}")
        raw_params = json.loads(best_params_path.read_text(encoding="utf-8"))
        mlp_params = normalise_mlp_params(raw_params)
        if MODEL_MLP not in models_to_train:
            logger.warning(
                "Ignoring --best-params-json because the MLP model is not selected for training."
            )
            mlp_params = None
        else:
            if not use_raw_pca:
                logger.info(
                    "Enabling PCA preprocessing to honour Optuna-derived n_components=%d.",
                    int(mlp_params["n_components"]),
                )
                use_raw_pca = True
            if n_pcs != int(mlp_params["n_components"]):
                logger.info(
                    "Overriding --n-pcs=%d with Optuna configuration n_components=%d.",
                    n_pcs,
                    int(mlp_params["n_components"]),
                )
            n_pcs = int(mlp_params["n_components"])

    # Parse custom class weights if provided
    if args.class_weights is not None:
        class_weight_dict = {}
        for pair in args.class_weights.split(","):
            pair = pair.strip()
            if ":" not in pair:
                raise ValueError(
                    f"Invalid class weight pair '{pair}'. Expected format: '0:2.0,1:1.0'"
                )
            class_str, weight_str = pair.split(":", 1)
            class_label = int(class_str.strip())
            weight_value = float(weight_str.strip())
            class_weight_dict[class_label] = weight_value

        logger.info("Using custom class weights: %s", class_weight_dict)

        # Add to mlp_params or create it
        if mlp_params is None:
            mlp_params = {}
        mlp_params["class_weight"] = class_weight_dict

    # Parse logistic regression class weights if provided
    logreg_class_weight = None
    if hasattr(args, "logreg_class_weights") and args.logreg_class_weights is not None:
        if args.logreg_class_weights.strip().lower() == "balanced":
            logreg_class_weight = "balanced"
            logger.info("Using balanced class weights for logistic regression")
        else:
            logreg_cw_dict = {}
            for pair in args.logreg_class_weights.split(","):
                pair = pair.strip()
                if ":" not in pair:
                    raise ValueError(
                        f"Invalid logreg class weight pair '{pair}'. Expected format: '0:1.0,1:3.0' or 'balanced'"
                    )
                class_str, weight_str = pair.split(":", 1)
                class_label = int(class_str.strip())
                weight_value = float(weight_str.strip())
                logreg_cw_dict[class_label] = weight_value
            logreg_class_weight = logreg_cw_dict
            logger.info("Using custom class weights for logistic regression: %s", logreg_class_weight)

    # Parse Random Forest class weights if provided
    rf_class_weight = None
    if hasattr(args, "rf_class_weights") and args.rf_class_weights is not None:
        if args.rf_class_weights.strip().lower() == "balanced":
            rf_class_weight = "balanced"
            logger.info("Using balanced class weights for Random Forest")
        else:
            rf_cw_dict = {}
            for pair in args.rf_class_weights.split(","):
                pair = pair.strip()
                if ":" not in pair:
                    raise ValueError(
                        f"Invalid RF class weight pair '{pair}'. Expected format: '0:1.0,1:3.0' or 'balanced'"
                    )
                class_str, weight_str = pair.split(":", 1)
                class_label = int(class_str.strip())
                weight_value = float(weight_str.strip())
                rf_cw_dict[class_label] = weight_value
            rf_class_weight = rf_cw_dict
            logger.info("Using custom class weights for Random Forest: %s", rf_class_weight)

    metrics = train_models(
        data_csv=args.data_csv,
        labels_csv=args.labels_csv,
        features=features,
        use_raw_pca=use_raw_pca,
        n_pcs=n_pcs,
        models=models_to_train,
        artifacts_dir=args.artifacts_dir,
        cv=args.cv,
        seed=args.seed,
        verbose=args.verbose,
        dry_run=args.dry_run,
        logreg_solver=args.logreg_solver,
        logreg_max_iter=args.logreg_max_iter,
        logreg_class_weight=logreg_class_weight,
        rf_n_estimators=getattr(args, "rf_n_estimators", 100),
        rf_max_depth=getattr(args, "rf_max_depth", None),
        rf_class_weight=rf_class_weight,
        trace_prefixes=prefixes,
        include_traces=include_traces,
        geometry_source=geometry_source,
        **geom_kwargs,
        group_column=args.group_column,
        group_override=args.group_override,
        test_size=args.test_size,
        mlp_params=mlp_params,
        classification_mode=args.classification_mode,
    )
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
    if not args.labels_csv:
        raise ValueError("--labels-csv is required for eval")
    if args.data_csv is None and getattr(args, "geometry_frames", None) is None:
        raise ValueError("Provide --data-csv or --geometry-frames for eval")
    if args.data_csv is not None and getattr(args, "geometry_frames", None) is not None:
        raise ValueError("Specify either --data-csv or --geometry-frames, not both")
    if getattr(args, "geometry_frames", None) is not None and args.geom_use_cache and not args.geom_cache_parquet:
        raise ValueError("--geom-use-cache requires --geom-cache-parquet when using geometry inputs")
    if getattr(args, "geometry_trials", None) is not None and getattr(args, "geometry_frames", None) is None:
        raise ValueError("--geometry-trials requires --geometry-frames")
    if getattr(args, "geom_feature_columns", None) and getattr(args, "geometry_frames", None) is None:
        raise ValueError("--geom-feature-columns requires --geometry-frames")
    run_dir = _resolve_run_dir(args.artifacts_dir, args.run_dir)
    logger = get_logger("eval", verbose=args.verbose)
    logger.info("Using run directory: %s", run_dir)
    config_prefixes: Sequence[str] | None = None
    config_use_traces: bool | None = None
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = PipelineConfig.from_json(config_path)
        config_use_traces = getattr(config, "use_trace_series", True)
        if config.trace_series_prefixes:
            config_prefixes = config.trace_series_prefixes
        elif config_use_traces is False:
            config_prefixes = []
    prefer_config_no_raw = (
        config_use_traces is False
        and args.series_prefixes is None
        and not getattr(args, "raw_series", False)
    )
    effective_no_raw = getattr(args, "no_raw", False) or prefer_config_no_raw
    prefixes = _select_trace_prefixes(
        args,
        fallback=config_prefixes,
        force_no_raw=effective_no_raw,
    )
    geom_kwargs = _extract_geometry_kwargs(args)
    geometry_source = geom_kwargs.pop("geometry_source")
    dataset = load_dataset(
        data_csv=args.data_csv,
        labels_csv=args.labels_csv,
        logger_name="eval",
        trace_prefixes=prefixes,
        include_traces=not effective_no_raw,
        geometry_source=geometry_source,
        **geom_kwargs,
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
    config_use_traces: bool | None = None
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = PipelineConfig.from_json(config_path)
        config_use_traces = getattr(config, "use_trace_series", True)
        if config.trace_series_prefixes:
            config_prefixes = config.trace_series_prefixes
        elif config_use_traces is False:
            config_prefixes = []
    prefer_config_no_raw = (
        config_use_traces is False
        and args.series_prefixes is None
        and not getattr(args, "raw_series", False)
    )
    effective_no_raw = getattr(args, "no_raw", False) or prefer_config_no_raw
    prefixes = _select_trace_prefixes(
        args,
        fallback=config_prefixes,
        force_no_raw=effective_no_raw,
    )
    generate_visuals(
        data_csv=args.data_csv,
        labels_csv=args.labels_csv,
        run_dir=run_dir,
        seed=args.seed,
        output_dir=args.plots_dir,
        verbose=args.verbose,
        trace_prefixes=prefixes,
        include_traces=not effective_no_raw,
    )


def _handle_predict(args: argparse.Namespace) -> None:
    geometry_path = getattr(args, "geometry_frames", None)
    if args.data_csv is None and geometry_path is None:
        raise ValueError("Provide --data-csv or --geometry-frames for predict")
    if args.data_csv is not None and geometry_path is not None:
        raise ValueError("Specify either --data-csv or --geometry-frames, not both")
    if geometry_path is not None and args.geom_use_cache and not args.geom_cache_parquet:
        raise ValueError("--geom-use-cache requires --geom-cache-parquet when using geometry inputs")
    if getattr(args, "geometry_trials", None) is not None and geometry_path is None:
        raise ValueError("--geometry-trials requires --geometry-frames")
    if getattr(args, "geom_feature_columns", None) and geometry_path is None:
        raise ValueError("--geom-feature-columns requires --geometry-frames")
    logger = get_logger("predict", verbose=args.verbose)
    model = load_pipeline(args.model_path)

    if geometry_path is not None:
        logger.info("Loading prediction geometry: %s", geometry_path)
        geom_kwargs = _extract_geometry_kwargs(args)
        geometry_source = geom_kwargs.pop("geometry_source")
        dataset = load_geometry_dataset(
            geometry_source,
            labels_csv=args.labels_csv,
            logger_name="predict",
            chunk_size=geom_kwargs["geom_chunk_size"],
            columns=geom_kwargs["geom_columns"],
            cache_parquet=geom_kwargs["geom_cache_parquet"],
            use_cache=geom_kwargs["geom_use_cache"],
            frame_column=geom_kwargs["geom_frame_column"],
            stats=geom_kwargs["geom_stats"],
            granularity=geom_kwargs["geom_granularity"],
            normalization=geom_kwargs["geom_normalization"],
            drop_missing_labels=geom_kwargs["geom_drop_missing_labels"],
            downcast=geom_kwargs["geom_downcast"],
            trial_summary=geom_kwargs["geom_trial_summary"],
            feature_columns=geom_kwargs["geom_feature_columns"],
            include_traces=not getattr(args, "no_raw", False),
        )
        data_df = dataset.frame.copy()
        logger.debug("Prediction dataset shape after geometry processing: %s", data_df.shape)
        if getattr(args, "no_raw", False) and dataset.trace_columns:
            drop_columns = [col for col in dataset.trace_columns if col in data_df.columns]
            if drop_columns:
                data_df = data_df.drop(columns=drop_columns)
                logger.info(
                    "Excluded %d raw trace columns from geometry predictions.",
                    len(drop_columns),
                )
    else:
        logger.info("Loading prediction data: %s", args.data_csv)
        data_df = pd.read_csv(args.data_csv)
        logger.debug("Prediction dataset shape: %s", data_df.shape)
        if getattr(args, "no_raw", False):
            drop_prefixes = list(dict.fromkeys([*DEFAULT_TRACE_PREFIXES, *RAW_TRACE_PREFIXES]))
            drop_columns = [
                col
                for col in data_df.columns
                if TRACE_PATTERN.match(col)
                or any(col.startswith(prefix) for prefix in drop_prefixes)
            ]
            if drop_columns:
                data_df = data_df.drop(columns=drop_columns)
                logger.info(
                    "Excluded %d raw trace columns from prediction inputs.",
                    len(drop_columns),
                )

    original_columns = set(data_df.columns)
    had_testing_trial_column = "testing_trial" in original_columns
    had_trial_label_column = "trial_label" in original_columns
    if not had_trial_label_column and had_testing_trial_column:
        logger.info("Detected legacy 'testing_trial' column; treating it as 'trial_label'.")
        data_df = data_df.rename(columns={"testing_trial": "trial_label"})
        had_trial_label_column = True

    filtered_df = data_df.copy()
    applied_filters: list[str] = []

    if args.fly is not None:
        if "fly" not in filtered_df.columns:
            raise ValueError("Column 'fly' missing from prediction CSV; cannot filter by fly.")
        filtered_df = filtered_df.loc[filtered_df["fly"].astype(str) == args.fly]
        applied_filters.append(f"fly={args.fly}")

    if args.fly_number is not None:
        if "fly_number" not in filtered_df.columns:
            raise ValueError(
                "Column 'fly_number' missing from prediction CSV; cannot filter by fly number."
            )
        numeric_fly_numbers = pd.to_numeric(filtered_df["fly_number"], errors="coerce")
        filtered_df = filtered_df.loc[numeric_fly_numbers == args.fly_number]
        applied_filters.append(f"fly_number={args.fly_number}")

    trial_filter_value = args.trial_label if args.trial_label is not None else args.testing_trial
    if trial_filter_value is not None:
        if "trial_label" not in filtered_df.columns:
            missing_column = "trial_label" if had_trial_label_column else "testing_trial"
            raise ValueError(
                f"Column '{missing_column}' missing from prediction CSV; cannot filter by trial."
            )
        filtered_df = filtered_df.loc[
            filtered_df["trial_label"].astype(str) == str(trial_filter_value)
        ]
        applied_filters.append(f"trial_label={trial_filter_value}")

    if filtered_df.empty:
        criteria = ", ".join(applied_filters) if applied_filters else "provided dataset"
        raise ValueError(f"No rows matched the prediction filters ({criteria}).")

    if applied_filters and len(filtered_df) > 1:
        raise ValueError(
            "Prediction filters %s matched %d rows; refine selection with more specific values."
            % (applied_filters, len(filtered_df))
        )

    filtered_df = filtered_df.copy()
    if had_testing_trial_column and "testing_trial" not in filtered_df.columns:
        filtered_df["testing_trial"] = filtered_df.get("trial_label", pd.NA)

    feature_df = filtered_df.drop(columns=[LABEL_COLUMN, LABEL_INTENSITY_COLUMN], errors="ignore")

    logger.info(
        "Scoring %d row(s) with model %s", len(filtered_df), args.model_path.name
    )

    # Use custom threshold if model supports probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feature_df)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            proba_positive = proba[:, 1]
            predictions = (proba_positive >= args.threshold).astype(int)
            if args.threshold != 0.5:
                logger.info("Applying custom decision threshold: %.2f", args.threshold)
        else:
            predictions = model.predict(feature_df)
            proba_positive = None
    else:
        predictions = model.predict(feature_df)
        proba_positive = None

    output_columns = [
        col
        for col in ["dataset", "fly", "fly_number", "trial_label", "testing_trial"]
        if col in filtered_df.columns
    ]
    output = filtered_df[output_columns].copy() if output_columns else pd.DataFrame()
    output["prediction"] = predictions.astype(int)

    if proba_positive is not None:
        output["probability"] = proba_positive

    if args.dry_run:
        logger.info("Dry run enabled; predictions not written")
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
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

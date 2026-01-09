"""Command-line interface for flypca."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import typer
import yaml

from .cluster import cluster_features, evaluate_with_labels
from .features import compute_feature_table
from .io import load_trials
from .lagpca import LagPCAResult, fit_lag_pca_for_trials, project_trial
from .paths import default_config_path, outputs_dir
from .viz import feature_violin, pc_scatter, pc_trajectories_plot, pc_loadings_plot, scree_plot

app = typer.Typer(add_completion=False)

DEFAULT_REPORT_DIR = outputs_dir() / "artifacts"


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_config_or_default(
    config_path: Optional[Path],
    default_path: Optional[Path] = None,
) -> Dict:
    """Load a YAML config, falling back to the repo default when available."""

    if config_path is not None:
        return _load_config(config_path)
    resolved = default_path or default_config_path()
    if resolved and resolved.exists():
        logging.info("No config supplied; defaulting to %s", resolved)
        return _load_config(resolved)
    logging.debug("No configuration provided and no default config found.")
    return {}


@app.callback()
def main(
    ctx: typer.Context,
    log_level: str = typer.Option("INFO", help="Logging level."),
    seed: int = typer.Option(0, help="Random seed."),
) -> None:
    np.random.seed(seed)
    _configure_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["seed"] = seed


@app.command("fit-lag-pca")
def fit_lag_pca(
    data: Path = typer.Option(..., exists=True, help="Path to data CSV or directory."),
    config: Path = typer.Option(..., exists=True, help="YAML configuration path."),
    out: Path = typer.Option(..., help="Output path for joblib model."),
    incremental: bool = typer.Option(False, help="Use IncrementalPCA."),
) -> None:
    cfg = _load_config(config)
    trials = load_trials(data, cfg)
    logging.info("Loaded %d trials for lag PCA fitting", len(trials))
    result = fit_lag_pca_for_trials(trials, cfg, incremental=incremental, model_path=out)
    logging.info("Explained variance ratio: %s", result.explained_variance_ratio_)


def _load_model(path: Path) -> LagPCAResult:
    return LagPCAResult.load(path)


def _format_trial_id_from_labels(row: pd.Series, template: Optional[str]) -> str:
    """Derive a trial identifier for an external label row."""

    if "trial_id" in row and pd.notna(row["trial_id"]):
        return str(row["trial_id"])
    if template:
        try:
            return str(template.format(**row))
        except KeyError:
            logging.debug(
                "Label row missing keys for template %s; falling back to defaults.", template
            )
    fly_value = row.get("fly") or row.get("fly_id")
    trial_label = row.get("trial_label") or row.get("trial") or row.get("trial_name")
    if pd.notna(fly_value) and pd.notna(trial_label):
        return f"{fly_value}_{trial_label}"
    raise ValueError(
        "Unable to derive trial_id for label row; include 'trial_id' or both 'fly' and "
        "'trial_label'."
    )


def _load_labels_table(
    labels_path: Path,
    config: Dict[str, object],
    label_column: str,
) -> pd.DataFrame:
    """Load external labels and align them by trial identifier."""

    labels_df = pd.read_csv(labels_path)
    if label_column not in labels_df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in {labels_path}. "
            "Ensure the CSV includes the requested column."
        )
    io_cfg = config.get("io", {}) if isinstance(config, dict) else {}
    wide_cfg = io_cfg.get("wide", {}) if isinstance(io_cfg, dict) else {}
    template = wide_cfg.get("trial_id_template") if isinstance(wide_cfg, dict) else None
    labels_df = labels_df.copy()
    labels_df["trial_id"] = labels_df.apply(
        lambda row: _format_trial_id_from_labels(row, template), axis=1
    )
    labels_df["trial_id"] = labels_df["trial_id"].astype(str)
    if labels_df.duplicated("trial_id").any():
        logging.warning(
            "Duplicate trial_ids detected in labels file %s; keeping last occurrence.",
            labels_path,
        )
        labels_df = labels_df.drop_duplicates("trial_id", keep="last")
    return labels_df[["trial_id", label_column]].copy()


def _load_projection_directory(path: Path) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    projections: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for npz_path in path.glob("*.npz"):
        data = np.load(npz_path)
        projections[npz_path.stem] = (data["time"], data["pcs"])
    return projections


def _build_projection_matrix(
    projections: Dict[str, tuple[np.ndarray, np.ndarray]],
    trial_ids: Sequence[str],
    n_components: Optional[int] = None,
    max_timepoints: Optional[int] = None,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    expected_shape: Optional[Tuple[int, int]] = None
    for trial_id in trial_ids:
        if trial_id not in projections:
            raise ValueError(f"Missing projection for trial {trial_id}")
        _, pcs = projections[trial_id]
        pcs_array = np.asarray(pcs)
        if pcs_array.ndim != 2:
            raise ValueError(f"Projection for {trial_id} has unexpected shape {pcs_array.shape}")
        comp = n_components if n_components is not None else pcs_array.shape[1]
        comp = min(comp, pcs_array.shape[1])
        trimmed = pcs_array[:, :comp]
        if max_timepoints is not None:
            trimmed = trimmed[:max_timepoints, :]
        if expected_shape is None:
            expected_shape = trimmed.shape
        elif trimmed.shape != expected_shape:
            raise ValueError(
                "Projection shapes are inconsistent: expected %s got %s for %s"
                % (expected_shape, trimmed.shape, trial_id)
            )
        rows.append(trimmed.reshape(-1))
    if not rows:
        raise ValueError("No projections available for clustering")
    return np.vstack(rows)


@app.command("project")
def project(
    model: Path = typer.Option(..., exists=True),
    data: Path = typer.Option(..., exists=True),
    config: Optional[Path] = typer.Option(None, help="Optional config for loading data."),
    out: Path = typer.Option(..., help="Output directory for projections."),
) -> None:
    cfg = _load_config_or_default(config)
    trials = load_trials(data, cfg)
    logging.info("Loaded %d trials for projection", len(trials))
    result = _load_model(model)
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for trial in trials:
        time, pcs = project_trial(trial, result)
        np.savez(out / f"{trial.trial_id}.npz", time=time, pcs=pcs)
        manifest_rows.append({"trial_id": trial.trial_id, "fly_id": trial.fly_id, "file": f"{trial.trial_id}.npz"})
    pd.DataFrame(manifest_rows).to_csv(out / "manifest.csv", index=False)
    logging.info("Saved projections for %d trials to %s", len(manifest_rows), out)


@app.command("features")
def features(
    data: Path = typer.Option(..., exists=True),
    config: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(..., help="Output parquet file for features."),
    model: Optional[Path] = typer.Option(None, exists=True, help="Optional model for PC features."),
    projections: Optional[Path] = typer.Option(None, exists=True, help="Optional directory of projections."),
) -> None:
    cfg = _load_config(config)
    trials = load_trials(data, cfg)
    logging.info("Loaded %d trials for feature extraction", len(trials))
    lag_result = _load_model(model) if model else None
    projection_map = _load_projection_directory(projections) if projections else None
    table = compute_feature_table(trials, cfg, result=lag_result, projections=projection_map)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    logging.info("Saved feature table with %d rows to %s", len(table), out)


@app.command("cluster")
def cluster(
    features_path: Path = typer.Option(..., "--features-path", "--features", exists=True),
    out: Path = typer.Option(..., help="Output CSV for cluster assignments."),
    method: str = typer.Option("gmm", help="Clustering method."),
    n_components: int = typer.Option(2, help="Number of clusters for GMM."),
    labels_path: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="Optional CSV containing ground-truth labels (e.g. user_score_odor).",
    ),
    labels_column_name: str = typer.Option(
        "user_score_odor",
        help="Column name in the labels CSV providing numeric annotations.",
    ),
    label_column: Optional[str] = typer.Option(None, help="Optional label column name."),
    config: Optional[Path] = typer.Option(None, exists=True, help="Optional YAML config."),
    projections_dir: Optional[Path] = typer.Option(None, exists=True, help="Optional directory of projections."),
    datasets: Optional[List[str]] = typer.Option(
        None,
        "--dataset",
        "--datasets",
        help="Filter clustering to specific dataset names.",
    ),
) -> None:
    cfg = _load_config_or_default(config)
    table = pd.read_parquet(features_path) if features_path.suffix == ".parquet" else pd.read_csv(features_path)
    cluster_cfg = cfg.get("clustering", {})
    dataset_filter_cfg = datasets if datasets else cluster_cfg.get("datasets")
    if dataset_filter_cfg:
        dataset_values = (
            list(dataset_filter_cfg)
            if isinstance(dataset_filter_cfg, (list, tuple, set))
            else [dataset_filter_cfg]
        )
        dataset_values = [str(v) for v in dataset_values if v is not None]
        if dataset_values:
            if "dataset" not in table.columns:
                logging.warning(
                    "Dataset filter requested but 'dataset' column missing; skipping filter."
                )
            else:
                before = len(table)
                table = table[table["dataset"].astype(str).isin(dataset_values)].copy()
                if table.empty:
                    raise ValueError(
                        "No rows remain after applying dataset filter: %s"
                        % ", ".join(dataset_values)
                    )
                logging.info(
                    "Filtered datasets %s: %d -> %d rows",
                    ", ".join(dataset_values),
                    before,
                    len(table),
                )
    if labels_path is not None:
        labels_table = _load_labels_table(labels_path, cfg, labels_column_name)
        before_merge = len(table)
        table = table.merge(labels_table, on="trial_id", how="left")
        missing_mask = table[labels_column_name].isna()
        if missing_mask.any():
            sample_missing = ", ".join(
                table.loc[missing_mask, "trial_id"].astype(str).head(5)
            )
            logging.warning(
                "Labels missing for %d of %d trials (e.g. %s).",
                int(missing_mask.sum()),
                before_merge,
                sample_missing or "n/a",
            )
        else:
            logging.info("Merged labels for all %d trials.", before_merge)
        if label_column is None:
            label_column = labels_column_name
    feature_columns = cluster_cfg.get("feature_columns")
    min_variance = float(cluster_cfg.get("min_variance", 1e-6))
    standardize = bool(cluster_cfg.get("standardize", True))
    component_range_cfg = cluster_cfg.get("component_range")
    if component_range_cfg is None:
        component_values = None
    else:
        if isinstance(component_range_cfg, dict) and {"min", "max"} <= component_range_cfg.keys():
            component_values = range(
                int(component_range_cfg["min"]),
                int(component_range_cfg["max"]) + 1,
            )
        else:
            values = list(component_range_cfg) if isinstance(component_range_cfg, Sequence) else [component_range_cfg]
            component_values = [int(v) for v in values]
    covariance_types = cluster_cfg.get("covariance_types")
    if covariance_types is not None:
        covariance_types = [str(cov) for cov in covariance_types]
    use_projections_cfg = cluster_cfg.get("use_projections", "auto")
    if isinstance(use_projections_cfg, str) and use_projections_cfg.lower() == "auto":
        use_projections = projections_dir is not None
    else:
        use_projections = bool(use_projections_cfg)
    if projections_dir is not None:
        use_projections = True
    combine_cfg = cluster_cfg.get("combine_with_features", "auto")
    if isinstance(combine_cfg, str) and combine_cfg.lower() == "auto":
        combine_projection = use_projections
    else:
        combine_projection = bool(combine_cfg)
    projection_components = cluster_cfg.get("projection_components")
    projection_timepoints = cluster_cfg.get("projection_timepoints")
    projection_matrix = None
    if projections_dir is not None or use_projections:
        if projections_dir is None:
            raise ValueError("Projections directory required when projections are enabled")
        projections = _load_projection_directory(projections_dir)
        trial_ids = table["trial_id"].astype(str).tolist()
        projection_matrix = _build_projection_matrix(
            projections,
            trial_ids,
            n_components=int(projection_components) if projection_components is not None else None,
            max_timepoints=int(projection_timepoints) if projection_timepoints is not None else None,
        )
        use_projection_only = use_projections and not combine_projection
        if use_projection_only:
            logging.info("Clustering using projection trajectories only")
        elif combine_projection:
            logging.info("Clustering using combined feature and projection spaces")
        else:
            logging.info("Clustering using projection trajectories (default behaviour)")
    result = cluster_features(
        table,
        method=method,
        n_components=n_components,
        random_state=cfg.get("seed", 0),
        feature_columns=feature_columns,
        min_variance=min_variance,
        standardize=standardize,
        component_range=component_values,
        covariance_types=covariance_types,
        projection_matrix=projection_matrix,
        combine_projection=combine_projection,
    )
    df_out = table[["trial_id", "fly_id"]].copy()
    df_out["cluster"] = result.assignments
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    metrics_path = out.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result.metrics, f, indent=2)
    logging.info("Saved clusters to %s", out)
    if label_column and label_column in table.columns:
        metrics_supervised = evaluate_with_labels(table.drop(columns=[label_column]), table[label_column], table["fly_id"])
        logging.info("Supervised metrics: %s", metrics_supervised)
        with (out.with_suffix(".supervised.json")).open("w", encoding="utf-8") as f:
            json.dump(metrics_supervised, f, indent=2)


@app.command("report")
def report(
    features_path: Path = typer.Option(..., "--features-path", "--features", exists=True),
    clusters_path: Path = typer.Option(..., "--clusters-path", "--clusters", exists=True),
    model: Optional[Path] = typer.Option(None, exists=True),
    projections_dir: Optional[Path] = typer.Option(None, exists=True),
    out_dir: Path = typer.Option(DEFAULT_REPORT_DIR, help="Output directory for report."),
) -> None:
    features_df = pd.read_parquet(features_path) if features_path.suffix == ".parquet" else pd.read_csv(features_path)
    clusters_df = pd.read_csv(clusters_path)
    clusters_df["trial_id"] = clusters_df["trial_id"].astype(str)
    if clusters_df.duplicated("trial_id").any():
        logging.warning(
            "Duplicate trial_ids detected in clusters file %s; keeping last occurrence.",
            clusters_path,
        )
        clusters_df = clusters_df.drop_duplicates("trial_id", keep="last")
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    feature_trial_ids = features_df["trial_id"].astype(str)
    cluster_series = (
        clusters_df.set_index("trial_id")["cluster"].reindex(feature_trial_ids)
    )
    features_df = features_df.copy()
    features_df["cluster"] = cluster_series.to_numpy()
    missing_assignments = features_df["cluster"].isna()
    if missing_assignments.any():
        missing_trials = feature_trial_ids[missing_assignments.to_numpy()]
        sample = ", ".join(map(str, missing_trials[:10])) or "n/a"
        logging.warning(
            "Dropping %d feature rows without cluster assignments (e.g. %s).",
            int(missing_assignments.sum()),
            sample,
        )
        features_df = features_df.loc[~missing_assignments.to_numpy()].reset_index(drop=True)
        cluster_series = features_df["cluster"].reset_index(drop=True)
        feature_trial_ids = features_df["trial_id"].astype(str)
        if features_df.empty:
            raise ValueError(
                "No overlapping trials between features table and cluster assignments."
            )
    assignments = features_df.pop("cluster").to_numpy()
    projections = _load_projection_directory(projections_dir) if projections_dir else None
    result = _load_model(model) if model else None
    if result:
        scree_plot(result, figures_dir)
        pc_loadings_plot(result, figures_dir)
    if projections:
        traj_data = [projections[trial] for trial in list(projections.keys())[:5]]
        pc_trajectories_plot(traj_data, figures_dir)
    pc_scatter(features_df, assignments, figures_dir)
    feature_violin(features_df, assignments, ["latency", "peak_value", "snr"], figures_dir)
    report_path = out_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# FlyPCA Report\n\n")
        f.write("## Cluster Metrics\n\n")
        if (clusters_path.with_suffix(".metrics.json")).exists():
            metrics_data = json.loads((clusters_path.with_suffix(".metrics.json")).read_text())
            for key, value in metrics_data.items():
                f.write(f"- {key}: {value:.3f}\n")
        f.write("\nFigures saved to `figures/`.\n")
    logging.info("Report written to %s", report_path)


if __name__ == "__main__":
    app()

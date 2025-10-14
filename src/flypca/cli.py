"""Command-line interface for flypca."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import typer
import yaml

from .cluster import cluster_features, evaluate_with_labels
from .features import compute_feature_table
from .io import load_trials
from .lagpca import LagPCAResult, fit_lag_pca_for_trials, project_trial
from .viz import feature_violin, pc_scatter, pc_trajectories_plot, pc_loadings_plot, scree_plot

app = typer.Typer(add_completion=False)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    result = fit_lag_pca_for_trials(trials, cfg, incremental=incremental, model_path=out)
    logging.info("Explained variance ratio: %s", result.explained_variance_ratio_)


def _load_model(path: Path) -> LagPCAResult:
    return LagPCAResult.load(path)


def _load_projection_directory(path: Path) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    projections: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for npz_path in path.glob("*.npz"):
        data = np.load(npz_path)
        projections[npz_path.stem] = (data["time"], data["pcs"])
    return projections


@app.command("project")
def project(
    model: Path = typer.Option(..., exists=True),
    data: Path = typer.Option(..., exists=True),
    config: Optional[Path] = typer.Option(None, help="Optional config for loading data."),
    out: Path = typer.Option(..., help="Output directory for projections."),
) -> None:
    cfg = _load_config(config) if config else {}
    trials = load_trials(data, cfg)
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
    lag_result = _load_model(model) if model else None
    projection_map = _load_projection_directory(projections) if projections else None
    table = compute_feature_table(trials, cfg, result=lag_result, projections=projection_map)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    logging.info("Saved feature table with %d rows to %s", len(table), out)


@app.command("cluster")
def cluster(
    features_path: Path = typer.Option(..., exists=True),
    out: Path = typer.Option(..., help="Output CSV for cluster assignments."),
    method: str = typer.Option("gmm", help="Clustering method."),
    n_components: int = typer.Option(2, help="Number of clusters for GMM."),
    label_column: Optional[str] = typer.Option(None, help="Optional label column name."),
) -> None:
    table = pd.read_parquet(features_path) if features_path.suffix == ".parquet" else pd.read_csv(features_path)
    result = cluster_features(table, method=method, n_components=n_components)
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
    features_path: Path = typer.Option(..., exists=True),
    clusters_path: Path = typer.Option(..., exists=True),
    model: Optional[Path] = typer.Option(None, exists=True),
    projections_dir: Optional[Path] = typer.Option(None, exists=True),
    out_dir: Path = typer.Option(Path("artifacts"), help="Output directory for report."),
) -> None:
    features_df = pd.read_parquet(features_path) if features_path.suffix == ".parquet" else pd.read_csv(features_path)
    clusters_df = pd.read_csv(clusters_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    assignments = clusters_df.set_index("trial_id").loc[features_df["trial_id"], "cluster"].to_numpy()
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

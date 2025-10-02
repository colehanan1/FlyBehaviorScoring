"""Command line entry point for unsupervised clustering workflows."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict
import warnings

import numpy as np

from .data_prep import prepare_data
from .io_utils import (
    ensure_output_dir,
    write_clusters,
    write_report,
    write_time_importance,
    write_components,
)
from .pca_core import compute_pca, compute_time_importance
from .plots import (
    plot_embedding,
    plot_time_importance,
    plot_variance,
    plot_components,
    plot_cluster_traces,
)
from .models import pca_gmm, pca_hdbscan, pca_kmeans, reaction_profiles


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unsupervised clustering pipelines.")
    parser.add_argument("--npy", type=Path, required=True, help="Path to the trial matrix npy file.")
    parser.add_argument("--meta", type=Path, required=True, help="Path to the metadata JSON file.")
    parser.add_argument("--out", type=Path, default=Path("outputs/unsup"), help="Output directory base path.")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="Minimum cluster size for HDBSCAN/DBSCAN.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--max-pcs", type=int, default=10, help="Maximum number of principal components to retain.")
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=2,
        help="Minimum number of clusters to evaluate for k-means/GMM selection.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=10,
        help="Maximum number of clusters to evaluate for k-means/GMM selection.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=("EB", "3-octonol"),
        help="Dataset names to include in the analysis.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print verbose diagnostics during data preparation and modeling.",
    )
    return parser.parse_args()


def _collect_base_metrics(prepared, pca_results) -> Dict[str, int | float]:
    return {
        "n_trials": prepared.n_trials,
        "n_time": prepared.n_timepoints,
        "PCs_80pct": pca_results.pcs_80pct,
        "PCs_90pct": pca_results.pcs_90pct,
    }


def _extract_labels(metadata) -> np.ndarray | None:
    if "dataset_name" not in metadata.columns:
        return None
    labels = metadata["dataset_name"].to_numpy()
    unique = np.unique(labels)
    if unique.size != 2:
        return None
    mapping = {name: idx for idx, name in enumerate(sorted(unique))}
    return np.vectorize(mapping.get)(labels)


def main() -> None:
    args = _parse_args()

    if args.min_clusters < 1:
        raise ValueError("--min-clusters must be at least 1.")
    if args.max_clusters < args.min_clusters:
        raise ValueError("--max-clusters must be greater than or equal to --min-clusters.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out / timestamp
    artifacts = ensure_output_dir(run_dir)

    # Suppress the scikit-learn deprecation warning that advises renaming the
    # ``force_all_finite`` keyword argument to ``ensure_all_finite``. The
    # warning originates from internal cross-validation helpers invoked by the
    # clustering models we use, so filtering it here keeps the CLI output
    # focused on actionable diagnostics for end users.
    warnings.filterwarnings(
        "ignore",
        message="'force_all_finite' was renamed to 'ensure_all_finite'",
        category=FutureWarning,
        module=r"sklearn\..*",
    )

    if args.debug:
        print(f"[run_all] Writing artifacts to: {run_dir}")

    prepared = prepare_data(
        args.npy,
        args.meta,
        target_datasets=args.datasets,
        debug=args.debug,
    )
    if args.debug:
        print(
            "[run_all] Prepared traces:",
            f"n_trials={prepared.n_trials}",
            f"n_timepoints={prepared.n_timepoints}",
        )

    pca_results = compute_pca(prepared.traces, max_pcs=args.max_pcs, random_state=args.seed)
    if args.debug:
        print(
            "[run_all] PCA complete:",
            f"pcs_80pct={pca_results.pcs_80pct}",
            f"pcs_90pct={pca_results.pcs_90pct}",
            f"explained_var={np.round(pca_results.explained_variance_ratio, 4)}",
        )
    importance_df = compute_time_importance(pca_results, prepared.time_columns)
    write_time_importance(run_dir / "timepoint_importance.csv", importance_df)

    time_points = np.arange(1, prepared.n_timepoints + 1)

    for model_name in (
        "simple",
        "flexible",
        "noise_robust",
        "reaction_motifs",
        "reaction_clusters",
    ):
        plot_variance(pca_results, str(artifacts.variance_plot(model_name)))
        plot_time_importance(
            importance_df, str(artifacts.time_importance_plot(model_name))
        )

    labels_true = _extract_labels(prepared.metadata)

    base_metrics = _collect_base_metrics(prepared, pca_results)

    model_metrics: Dict[str, Dict[str, float | int | None]] = {}

    # Simple model: PCA + KMeans
    if args.debug:
        print("[run_all] Running PCA+k-means model...")
    simple_outputs = pca_kmeans.run_model(
        pca_results,
        dataset_labels=labels_true,
        seed=args.seed,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
    )
    embedding_simple = pca_results.scores[:, : max(2, pca_results.pcs_80pct or 2)]
    plot_embedding(
        embedding_simple[:, :2],
        simple_outputs.labels,
        str(artifacts.embedding_plot("simple")),
    )
    plot_cluster_traces(
        time_points,
        prepared.traces,
        simple_outputs.labels,
        str(artifacts.average_trace_plot("simple")),
    )

    metrics_simple = {
        **base_metrics,
        "algo": "PCA+k-means",
        **simple_outputs.metrics,
    }
    write_report(artifacts.report_path("simple"), metrics_simple)
    write_clusters(artifacts.cluster_path("simple"), prepared.metadata, simple_outputs.labels)
    model_metrics["simple"] = simple_outputs.metrics

    # Flexible model: PCA + GMM
    if args.debug:
        print("[run_all] Running PCA+GMM model...")
    flexible_outputs = pca_gmm.run_model(
        pca_results,
        dataset_labels=labels_true,
        seed=args.seed,
        min_components=args.min_clusters,
        max_components=args.max_clusters,
    )
    embedding_flexible = pca_results.scores[:, : max(2, pca_results.pcs_80pct or 2)]
    plot_embedding(
        embedding_flexible[:, :2],
        flexible_outputs.labels,
        str(artifacts.embedding_plot("flexible")),
    )
    plot_cluster_traces(
        time_points,
        prepared.traces,
        flexible_outputs.labels,
        str(artifacts.average_trace_plot("flexible")),
    )

    metrics_flexible = {
        **base_metrics,
        "algo": "PCA+GMM",
        **flexible_outputs.metrics,
    }
    write_report(artifacts.report_path("flexible"), metrics_flexible)
    write_clusters(artifacts.cluster_path("flexible"), prepared.metadata, flexible_outputs.labels)
    model_metrics["flexible"] = flexible_outputs.metrics

    # Noise-robust model: PCA + HDBSCAN/DBSCAN
    if args.debug:
        print("[run_all] Running PCA+HDBSCAN/DBSCAN model...")
    noise_outputs = pca_hdbscan.run_model(
        pca_results,
        dataset_labels=labels_true,
        min_cluster_size=args.min_cluster_size,
        seed=args.seed,
    )
    embedding_noise = pca_results.scores[:, : max(2, pca_results.pcs_80pct or 2)]
    plot_embedding(
        embedding_noise[:, :2],
        noise_outputs.labels,
        str(artifacts.embedding_plot("noise_robust")),
    )
    plot_cluster_traces(
        time_points,
        prepared.traces,
        noise_outputs.labels,
        str(artifacts.average_trace_plot("noise_robust")),
    )

    algo_name = (
        "PCA+HDBSCAN"
        if getattr(pca_hdbscan, "hdbscan", None) is not None
        else "PCA+DBSCAN"
    )
    metrics_noise = {
        **base_metrics,
        "algo": algo_name,
        **noise_outputs.metrics,
    }
    write_report(artifacts.report_path("noise_robust"), metrics_noise)
    write_clusters(artifacts.cluster_path("noise_robust"), prepared.metadata, noise_outputs.labels)
    model_metrics["noise_robust"] = noise_outputs.metrics

    # Odor reaction motif model
    if args.debug:
        print("[run_all] Running odor reaction motif model...")
    reaction_motif_outputs = reaction_profiles.run_model_with_motifs(
        prepared.traces,
        dataset_labels=labels_true,
        seed=args.seed,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
    )
    plot_embedding(
        reaction_motif_outputs.embedding,
        reaction_motif_outputs.labels,
        str(artifacts.embedding_plot("reaction_motifs")),
    )
    plot_cluster_traces(
        time_points,
        prepared.traces,
        reaction_motif_outputs.labels,
        str(artifacts.average_trace_plot("reaction_motifs")),
    )
    metrics_motifs = {
        **base_metrics,
        "algo": "Odor reaction motifs (NMF+k-means)",
        **reaction_motif_outputs.metrics,
    }
    write_report(artifacts.report_path("reaction_motifs"), metrics_motifs)
    write_clusters(
        artifacts.cluster_path("reaction_motifs"),
        prepared.metadata,
        reaction_motif_outputs.labels,
    )
    model_metrics["reaction_motifs"] = reaction_motif_outputs.metrics
    if (
        reaction_motif_outputs.components is not None
        and reaction_motif_outputs.component_time is not None
    ):
        write_components(
            artifacts.component_csv("reaction_motifs"),
            reaction_motif_outputs.component_time,
            reaction_motif_outputs.components,
        )
        plot_components(
            reaction_motif_outputs.component_time,
            reaction_motif_outputs.components,
            str(artifacts.component_plot("reaction_motifs")),
        )

    # Odor reaction cluster-only model
    if args.debug:
        print("[run_all] Running odor reaction cluster model...")
    reaction_cluster_outputs = reaction_profiles.run_model_clusters_only(
        prepared.traces,
        dataset_labels=labels_true,
        seed=args.seed,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
    )
    plot_embedding(
        reaction_cluster_outputs.embedding,
        reaction_cluster_outputs.labels,
        str(artifacts.embedding_plot("reaction_clusters")),
    )
    plot_cluster_traces(
        time_points,
        prepared.traces,
        reaction_cluster_outputs.labels,
        str(artifacts.average_trace_plot("reaction_clusters")),
    )
    metrics_reaction = {
        **base_metrics,
        "algo": "Odor reaction features (k-means)",
        **reaction_cluster_outputs.metrics,
    }
    write_report(artifacts.report_path("reaction_clusters"), metrics_reaction)
    write_clusters(
        artifacts.cluster_path("reaction_clusters"),
        prepared.metadata,
        reaction_cluster_outputs.labels,
    )
    model_metrics["reaction_clusters"] = reaction_cluster_outputs.metrics

    if args.debug:
        print("[run_all] Model metrics summary:", model_metrics)


if __name__ == "__main__":
    main()

"""Command line entry point for unsupervised clustering workflows."""
from __future__ import annotations

import argparse
from datetime import datetime
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_prep import prepare_data
from .io_utils import (
    ensure_output_dir,
    write_clusters,
    write_report,
    write_time_importance,
    write_components,
)
from .pca_core import compute_pca, compute_time_importance
from .odor_response import evaluate_odor_response, run_response_pca
from .plots import (
    ODOR_OFF_FRAME,
    ODOR_ON_FRAME,
    plot_auc_ranking,
    plot_cluster_odor_embedding,
    plot_cluster_traces,
    plot_components,
    plot_embedding,
    plot_embedding_by_auc,
    plot_pca_eigenvectors,
    plot_time_importance,
    plot_variance,
)
from .models import pca_gmm, pca_hdbscan, pca_kmeans, reaction_profiles
from .subcluster import run_subclustering


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
    parser.add_argument(
        "--skip-subclustering",
        action="store_true",
        help="Disable the automated second-stage subclustering step.",
    )
    parser.add_argument(
        "--subcluster-targets",
        nargs="+",
        default=("0", "1"),
        help="Parent cluster labels to pass into the subclustering workflow.",
    )
    parser.add_argument(
        "--subcluster-algos",
        nargs="+",
        default=("gmm", "hdbscan", "ward"),
        choices=("gmm", "hdbscan", "ward"),
        help=(
            "Algorithms to execute during subclustering (default: gmm hdbscan ward)."
        ),
    )
    parser.add_argument(
        "--subcluster-k-min",
        type=int,
        default=2,
        help="Minimum number of GMM components to evaluate for subclustering.",
    )
    parser.add_argument(
        "--subcluster-k-max",
        type=int,
        default=8,
        help="Maximum number of GMM components to evaluate for subclustering.",
    )
    parser.add_argument(
        "--subcluster-ward-k",
        type=int,
        default=4,
        help="Number of Ward linkage clusters to compute during subclustering.",
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


def _infer_testing_codes(metadata: pd.DataFrame) -> Tuple[pd.Series | None, str | None]:
    """Extract testing odor codes (1-10) from metadata if present."""

    pattern = re.compile(r"(?:^|\b)test(?:ing)?[^0-9]*([0-9]+)", re.IGNORECASE)

    for column in metadata.columns:
        series = metadata[column]
        if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
            continue
        str_series = series.astype(str)
        extracted = str_series.str.extract(pattern, expand=False)
        if extracted.notna().any():
            codes = pd.to_numeric(extracted, errors="coerce").astype("Int64")
            if codes.notna().any():
                return codes, column

    for column in metadata.columns:
        normalized = column.lower()
        if not any(keyword in normalized for keyword in ("test", "odor", "stim")):
            continue
        series = metadata[column]
        if not pd.api.types.is_numeric_dtype(series):
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            continue
        unique_values = {int(value) for value in valid.unique() if float(value).is_integer()}
        if not unique_values:
            continue
        if max(unique_values) <= 15 and min(unique_values) >= 0 and len(unique_values) >= 2:
            return numeric.astype("Int64"), column

    return None, None


def main() -> None:
    args = _parse_args()

    if args.min_clusters < 1:
        raise ValueError("--min-clusters must be at least 1.")
    if args.max_clusters < args.min_clusters:
        raise ValueError("--max-clusters must be greater than or equal to --min-clusters.")
    if args.subcluster_k_min < 2:
        raise ValueError("--subcluster-k-min must be at least 2 for GMM.")
    if args.subcluster_k_max < args.subcluster_k_min:
        raise ValueError("--subcluster-k-max must be >= --subcluster-k-min.")
    if args.subcluster_ward_k < 2:
        raise ValueError("--subcluster-ward-k must be at least 2.")

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

    pc_columns = [f"PC{i+1}" for i in range(pca_results.scores.shape[1])]
    pca_features = pd.DataFrame(
        pca_results.scores,
        index=prepared.metadata.index,
        columns=pc_columns,
    )

    time_points = np.arange(1, prepared.n_timepoints + 1)

    cluster_outputs: Dict[str, Path] = {}

    for model_name in (
        "simple",
        "flexible",
        "noise_robust",
        "reaction_motifs",
        "reaction_clusters",
    ):
        plot_variance(pca_results, str(artifacts.variance_plot(model_name)))
        plot_pca_eigenvectors(
            pca_results,
            time_points,
            str(artifacts.eigenvector_plot(model_name)),
            title=f"PCA eigenvectors ({model_name})",
        )
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

    try:
        odor_results = evaluate_odor_response(
            prepared.traces,
            simple_outputs.labels,
            prepared.metadata.index,
            time_points,
            odor_on=ODOR_ON_FRAME,
            odor_off=ODOR_OFF_FRAME,
            target_clusters=(0, 1),
        )
    except ValueError as exc:
        if args.debug:
            print("[run_all] Odor response evaluation failed:", exc)
        odor_results = None
    else:
        odor_results.metrics.to_csv(
            artifacts.response_auc_csv("simple"), index=False
        )
        odor_results.cluster_summary.to_csv(
            artifacts.response_auc_summary_csv("simple"), index=False
        )
        plot_auc_ranking(
            odor_results.metrics,
            str(artifacts.response_auc_ranking_plot("simple")),
        )

    response_embedding_data: np.ndarray | None = None

    if odor_results and odor_results.feature_matrix is not None:
        response_pca = run_response_pca(
            odor_results.feature_matrix,
            max_pcs=args.max_pcs,
            random_state=args.seed,
        )
        if response_pca is not None:
            plot_variance(
                response_pca,
                str(artifacts.response_variance_plot("simple")),
            )
            plot_pca_eigenvectors(
                response_pca,
                odor_results.feature_timepoints
                if odor_results.feature_timepoints is not None
                else np.arange(1, response_pca.components.shape[1] + 1),
                str(artifacts.response_eigenvector_plot("simple")),
                title="Odor response PCA eigenvectors",
            )
            response_embedding_data = response_pca.scores[:, :2]

            response_scores = pd.DataFrame(
                response_pca.scores,
                columns=[
                    f"PC{idx+1}" for idx in range(response_pca.scores.shape[1])
                ],
            )
            response_scores.insert(0, "cluster_label", odor_results.trial_labels)
            response_scores.insert(0, "trial_index", odor_results.trial_ids)
            response_scores.to_csv(
                artifacts.response_scores_csv("simple"), index=False
            )
        elif args.debug:
            print(
                "[run_all] Skipping odor response PCA due to insufficient variance."
            )
    elif args.debug and odor_results is not None:
        print("[run_all] Odor response evaluation returned no target trials.")

    if (
        odor_results
        and odor_results.auc_ratios.size > 0
        and odor_results.trial_labels.size == odor_results.auc_ratios.size
    ):
        if response_embedding_data is None:
            response_embedding_data = pca_results.scores[odor_results.trial_indices][:, :2]

        plot_embedding_by_auc(
            response_embedding_data,
            odor_results.trial_labels,
            odor_results.auc_ratios,
            str(artifacts.response_embedding_plot("simple")),
            cluster_colors={0: "#b2182b", 1: "#2166ac"},
        )

    testing_codes, testing_column = _infer_testing_codes(prepared.metadata)
    if testing_codes is not None:
        odor_color_map = {
            2: "red",
            4: "red",
            5: "red",
            1: "pink",
            3: "pink",
            6: "green",
            7: "blue",
            8: "black",
            9: "gray",
            10: "brown",
        }
        if args.debug:
            print(
                "[run_all] Using metadata column for tested odor:",
                testing_column,
            )
        aligned_codes = testing_codes.reindex(prepared.metadata.index)
        for cluster_label in (1, 0):
            plot_path = artifacts.base_dir / (
                f"embedding_simple_cluster{cluster_label}_tested_odor.png"
            )
            plot_cluster_odor_embedding(
                pca_features,
                simple_outputs.labels,
                aligned_codes,
                cluster_label,
                str(plot_path),
                color_map=odor_color_map,
                default_color="lightgrey",
            )
    elif args.debug:
        print(
            "[run_all] Unable to infer testing odor column for color-coded cluster plots.",
            "Available columns:",
            list(prepared.metadata.columns),
        )

    metrics_simple = {
        **base_metrics,
        "algo": "PCA+k-means",
        **simple_outputs.metrics,
    }
    write_report(artifacts.report_path("simple"), metrics_simple)
    simple_cluster_path = artifacts.cluster_path("simple")
    write_clusters(
        simple_cluster_path,
        prepared.metadata,
        simple_outputs.labels,
        features=pca_features,
    )
    cluster_outputs["simple"] = simple_cluster_path
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
    flexible_cluster_path = artifacts.cluster_path("flexible")
    write_clusters(
        flexible_cluster_path,
        prepared.metadata,
        flexible_outputs.labels,
        features=pca_features,
    )
    cluster_outputs["flexible"] = flexible_cluster_path
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
    noise_cluster_path = artifacts.cluster_path("noise_robust")
    write_clusters(
        noise_cluster_path,
        prepared.metadata,
        noise_outputs.labels,
        features=pca_features,
    )
    cluster_outputs["noise_robust"] = noise_cluster_path
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
    motifs_cluster_path = artifacts.cluster_path("reaction_motifs")
    write_clusters(
        motifs_cluster_path,
        prepared.metadata,
        reaction_motif_outputs.labels,
        features=pca_features,
    )
    cluster_outputs["reaction_motifs"] = motifs_cluster_path
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
    reaction_cluster_path = artifacts.cluster_path("reaction_clusters")
    write_clusters(
        reaction_cluster_path,
        prepared.metadata,
        reaction_cluster_outputs.labels,
        features=pca_features,
    )
    cluster_outputs["reaction_clusters"] = reaction_cluster_path
    model_metrics["reaction_clusters"] = reaction_cluster_outputs.metrics

    if not args.skip_subclustering:
        for model_name, cluster_csv in cluster_outputs.items():
            if not cluster_csv.exists():
                if args.debug:
                    print(
                        "[run_all] Skipping subclustering because cluster CSV is missing:",
                        model_name,
                        cluster_csv,
                    )
                continue

            subcluster_dir = run_dir / f"subclusters_{model_name}"
            if args.debug:
                print(
                    "[run_all] Running subclustering workflow on:",
                    cluster_csv,
                    "->",
                    subcluster_dir,
                )

            try:
                result_paths = run_subclustering(
                    cluster_csv,
                    cluster_col="cluster_label",
                    parent_targets=args.subcluster_targets,
                    algos=args.subcluster_algos,
                    k_min=args.subcluster_k_min,
                    k_max=args.subcluster_k_max,
                    ward_k=args.subcluster_ward_k,
                    outdir=subcluster_dir,
                )
                augmented_path = result_paths.get("augmented_csv")
                if isinstance(augmented_path, str):
                    augmented_path = Path(augmented_path)
                metrics_path = result_paths.get("metrics_csv")
                if isinstance(metrics_path, str):
                    metrics_path = Path(metrics_path)

                if args.debug:
                    print(
                        "[run_all] Subclustering outputs (model=", model_name, "):",
                        "augmented=", augmented_path,
                        "metrics=", metrics_path,
                    )

                if augmented_path is not None and augmented_path.exists():
                    augmented_df = pd.read_csv(augmented_path)
                    trial_subcluster_path = (
                        subcluster_dir / f"trial_clusters_{model_name}_subclusters.csv"
                    )
                    augmented_df.to_csv(trial_subcluster_path, index=False)

                    summary_records: List[Dict[str, object]] = []

                    if "source_row_index" not in augmented_df.columns:
                        if args.debug:
                            print(
                                "[run_all] Subcluster outputs missing source_row_index;"
                                " skipping average trace plots."
                            )
                    else:
                        for parent in args.subcluster_targets:
                            parent_mask = (
                                augmented_df["cluster_label"].astype(str).str.strip()
                                == str(parent)
                            )
                            if not parent_mask.any():
                                continue

                            parent_df = augmented_df.loc[parent_mask].copy()
                            parent_df.sort_values("source_row_index", inplace=True)

                            for algo in args.subcluster_algos:
                                if algo == "gmm":
                                    label_column = f"sub_gmm_label_parent{parent}"
                                elif algo == "hdbscan":
                                    label_column = f"sub_hdbscan_label_parent{parent}"
                                elif algo == "ward":
                                    label_column = f"sub_ward_label_parent{parent}"
                                else:
                                    continue

                                if label_column not in parent_df.columns:
                                    continue

                                labels_series = parent_df[label_column].dropna()
                                if labels_series.empty:
                                    continue

                                valid_indices = labels_series.index
                                labels = labels_series.to_numpy()
                                try:
                                    labels = labels.astype(int, copy=False)
                                except (ValueError, TypeError):
                                    labels = labels.astype(object)

                                plot_indices = (
                                    parent_df.loc[valid_indices, "source_row_index"]
                                    .astype(int)
                                    .to_numpy()
                                )
                                plot_traces = prepared.traces[plot_indices]

                                plot_path = (
                                    subcluster_dir
                                    / f"cluster_average_trace_{model_name}_parent{parent}_{algo}.png"
                                )
                                plot_cluster_traces(
                                    time_points,
                                    plot_traces,
                                    labels,
                                    str(plot_path),
                                )

                                try:
                                    counts_series = labels_series.astype(int)
                                except (ValueError, TypeError):
                                    counts_series = labels_series
                                counts = counts_series.value_counts()
                                for sub_label, count in counts.items():
                                    summary_records.append(
                                        {
                                            "parent_cluster": parent,
                                            "algorithm": algo,
                                            "subcluster_label": sub_label,
                                            "n_trials": int(count),
                                        }
                                    )

                    if summary_records:
                        summary_df = pd.DataFrame(summary_records)
                        summary_df["parent_cluster"] = summary_df["parent_cluster"].astype(str)
                        summary_df["subcluster_label"] = summary_df["subcluster_label"].astype(str)
                        summary_path = (
                            subcluster_dir / f"subcluster_membership_{model_name}.csv"
                        )
                        summary_df.sort_values(
                            ["algorithm", "parent_cluster", "subcluster_label"],
                            inplace=True,
                        )
                        summary_df.to_csv(summary_path, index=False)
            except Exception as exc:  # pragma: no cover - defensive guard
                warnings.warn(f"Subclustering step failed for {model_name}: {exc}")
                if args.debug:
                    print("[run_all] Subclustering error (model=", model_name, "):", exc)

    if args.debug:
        print("[run_all] Model metrics summary:", model_metrics)


if __name__ == "__main__":
    main()

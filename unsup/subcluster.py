"""Utilities for second-stage subclustering of coarse trial clusters."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Guard the optional dependency so the script still executes without HDBSCAN.
try:  # pragma: no cover - optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None


ID_LIKE_PATTERN = re.compile(
    r"(id|trial|fly|file|path|name|uid|timestamp|time|date|cluster|label|grp)",
    re.IGNORECASE,
)

PC_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"^(pc|pca|principal[_-]?component)[ _-]*(\d+)$", re.IGNORECASE),
    re.compile(r"^pc(\d+)$", re.IGNORECASE),
)

ALLOWED_ALGOS = ("gmm", "hdbscan", "ward")


@dataclass
class SubclusterResult:
    """Container for all outputs generated for a parent cluster."""

    parent_value: str
    features_used: str
    used_index: pd.Index
    gmm_summary: pd.DataFrame | None
    gmm_labels: np.ndarray | None
    gmm_k: int | None
    hdbscan_labels: np.ndarray | None
    ward_labels: np.ndarray | None


def _normalize_parent_values(values: Iterable[str | int | float]) -> List[str]:
    normalized: List[str] = []
    for value in values:
        normalized.append(str(value).strip())
    return normalized


def _sanitize_algorithms(algorithms: Sequence[str]) -> List[str]:
    sanitized: List[str] = []
    for algo in algorithms:
        normalized = str(algo).strip().lower()
        if normalized not in ALLOWED_ALGOS:
            raise ValueError(
                f"Unsupported algorithm {algo!r}. Expected one of: {ALLOWED_ALGOS}"
            )
        if normalized not in sanitized:
            sanitized.append(normalized)
    return sanitized


def detect_pca_columns(columns: Sequence[str]) -> List[str]:
    """Return columns that already represent principal components."""

    detected: List[str] = []
    for column in columns:
        stripped = column.strip()
        for pattern in PC_PATTERNS:
            if pattern.match(stripped):
                detected.append(column)
                break
        else:
            # Fall back to inspecting the first whitespace-delimited token.
            token = stripped.split()[0]
            if any(pattern.match(token) for pattern in PC_PATTERNS):
                detected.append(column)

    def sort_key(name: str) -> tuple[int, str]:
        match = re.search(r"(\d+)", name)
        if match:
            return (int(match.group(1)), name)
        return (10_000, name)

    detected = sorted(set(detected), key=sort_key)
    return detected


def _select_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_columns: List[str] = []
    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if ID_LIKE_PATTERN.search(column):
            continue
        numeric_columns.append(column)
    return numeric_columns


def build_feature_matrix(
    df: pd.DataFrame,
    cluster_column: str,
    parent_value: str,
    *,
    min_numeric_features: int = 3,
    max_pca_components: int = 10,
) -> tuple[pd.DataFrame, pd.Index, str]:
    """Return standardized feature matrix, selected index, and description."""

    mask = df[cluster_column].astype(str).str.strip() == parent_value
    if not mask.any():
        # Attempt a numeric comparison if string coercion fails.
        numeric_series = pd.to_numeric(df[cluster_column], errors="coerce")
        target_value = pd.to_numeric(pd.Series([parent_value]), errors="coerce").iloc[0]
        mask = numeric_series == target_value

    if not mask.any():
        raise ValueError(f"No rows found for parent cluster {parent_value!r}.")

    subset = df.loc[mask].copy()
    candidate_pcs = detect_pca_columns(subset.columns)

    if len(candidate_pcs) >= 2:
        numeric = subset[candidate_pcs].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
        feature_matrix = numeric.astype(float)
        description = f"provided PCs ({len(candidate_pcs)})"
    else:
        numeric_columns = _select_numeric_columns(subset)
        if len(numeric_columns) < min_numeric_features:
            raise ValueError(
                "Insufficient numeric features to compute PCA. "
                f"Found {len(numeric_columns)} columns: {numeric_columns}"
            )

        numeric = (
            subset[numeric_columns]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .astype(float)
        )
        scaler = StandardScaler()
        standardized = scaler.fit_transform(numeric.values)
        n_components = min(max_pca_components, standardized.shape[1])
        pca = PCA(n_components=n_components, random_state=17)
        projected = pca.fit_transform(standardized)
        feature_matrix = pd.DataFrame(
            projected,
            index=numeric.index,
            columns=[f"PC{i+1}" for i in range(projected.shape[1])],
        )
        description = (
            f"PCA({projected.shape[1]}) from {len(numeric_columns)} numeric cols"
        )

    return feature_matrix, feature_matrix.index, description


def _standardize(values: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(values.values)


def fit_gmm(
    X: pd.DataFrame,
    *,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 17,
) -> tuple[pd.DataFrame, int, np.ndarray]:
    """Fit GMM models and select the best component count via BIC."""

    if k_min < 2:
        raise ValueError("GaussianMixture requires at least two components.")
    if k_max < k_min:
        raise ValueError("k_max must be >= k_min")

    standardized = _standardize(X)
    records: List[dict] = []
    labels_by_k: dict[int, np.ndarray] = {}

    for k in range(k_min, k_max + 1):
        model = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=random_state,
            n_init=5,
            max_iter=1000,
        )
        model.fit(standardized)
        bic = model.bic(standardized)
        aic = model.aic(standardized)
        predicted = model.predict(standardized)
        if len(np.unique(predicted)) > 1:
            silhouette = silhouette_score(standardized, predicted)
        else:
            silhouette = np.nan
        labels_by_k[k] = predicted
        records.append(
            {
                "K": k,
                "BIC": float(bic),
                "AIC": float(aic),
                "Silhouette": float(silhouette) if np.isfinite(silhouette) else np.nan,
            }
        )

    summary = pd.DataFrame.from_records(records)
    bic_min = summary["BIC"].min()
    best_k = int(summary.loc[summary["BIC"].idxmin(), "K"])

    # Break near-ties using the silhouette score whenever possible.
    near_ties = summary[summary["BIC"] <= bic_min * 1.02]
    if len(near_ties) > 1:
        silhouettes = near_ties["Silhouette"].fillna(-np.inf)
        if not silhouettes.isna().all():
            best_k = int(near_ties.loc[silhouettes.idxmax(), "K"])

    return summary, best_k, labels_by_k[best_k]


def fit_hdbscan(X: pd.DataFrame) -> tuple[dict, np.ndarray] | tuple[None, None]:
    if hdbscan is None:
        return None, None

    standardized = _standardize(X)
    min_cluster_size = max(5, int(0.01 * len(standardized)))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        cluster_selection_epsilon=0.0,
    )
    labels = clusterer.fit_predict(standardized)
    n_clusters = int((labels >= 0).sum() and (labels[labels >= 0].max() + 1))
    meta = {"algo": "hdbscan", "n_clusters": n_clusters}
    return meta, labels


def fit_ward(X: pd.DataFrame, *, k: int = 4) -> tuple[dict, np.ndarray]:
    if k < 2:
        raise ValueError("Ward clustering requires k >= 2")

    standardized = _standardize(X)
    link = linkage(standardized, method="ward")
    labels = fcluster(link, t=k, criterion="maxclust") - 1
    return {"algo": "ward", "k": k}, labels


def _plot_gmm_metrics(
    summary: pd.DataFrame,
    *,
    parent_value: str,
    outdir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary["K"], summary["BIC"], marker="o", label="BIC")
    ax.plot(summary["K"], summary["AIC"], marker="s", label="AIC", alpha=0.7)
    ax.set_xlabel("Components (K)")
    ax.set_ylabel("Score")
    ax.set_title(f"GMM model selection for parent {parent_value}")
    ax.legend()
    fig.tight_layout()
    figure_path = outdir / f"sub_gmm_selection_parent{parent_value}_bic.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def _plot_pc_scatter(
    X: pd.DataFrame,
    labels: np.ndarray,
    *,
    parent_value: str,
    outdir: Path,
) -> None:
    if X.shape[1] < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=labels,
        cmap="tab10",
        s=18,
        alpha=0.8,
    )
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    ax.set_title(f"Parent {parent_value} GMM subclusters")
    legend1 = ax.legend(*scatter.legend_elements(), title="Subcluster")
    ax.add_artist(legend1)
    fig.tight_layout()
    figure_path = outdir / f"sub_gmm_parent{parent_value}_pc_scatter.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def process_parent_cluster(
    df: pd.DataFrame,
    cluster_column: str,
    parent_value: str,
    *,
    outdir: Path,
    algos: Sequence[str],
    k_min: int,
    k_max: int,
    ward_k: int,
) -> SubclusterResult:
    features, index, description = build_feature_matrix(df, cluster_column, parent_value)

    gmm_summary: pd.DataFrame | None = None
    gmm_labels: np.ndarray | None = None
    gmm_k: int | None = None
    hdbscan_labels: np.ndarray | None = None
    ward_labels: np.ndarray | None = None

    if "gmm" in algos:
        gmm_summary, gmm_k, gmm_labels = fit_gmm(
            features, k_min=k_min, k_max=k_max
        )
        gmm_summary.assign(
            parent_cluster=parent_value,
            features_used=description,
        ).to_csv(
            outdir / f"sub_gmm_selection_parent{parent_value}.csv",
            index=False,
        )
        _plot_gmm_metrics(gmm_summary, parent_value=parent_value, outdir=outdir)
        if gmm_labels is not None:
            _plot_pc_scatter(
                features.iloc[:, :2],
                gmm_labels,
                parent_value=parent_value,
                outdir=outdir,
            )

    if "hdbscan" in algos:
        hdbscan_meta, hdbscan_labels = fit_hdbscan(features)
        if hdbscan_meta is not None and hdbscan_labels is not None:
            hdbscan_labels = hdbscan_labels.astype(int)
        else:
            hdbscan_labels = None

    if "ward" in algos:
        ward_meta, ward_labels = fit_ward(features, k=ward_k)
        ward_labels = ward_labels.astype(int)

    return SubclusterResult(
        parent_value=parent_value,
        features_used=description,
        used_index=index,
        gmm_summary=gmm_summary,
        gmm_labels=gmm_labels,
        gmm_k=gmm_k,
        hdbscan_labels=hdbscan_labels,
        ward_labels=ward_labels,
    )


def run_subclustering(
    input_csv: Path | str,
    *,
    cluster_col: str,
    parent_targets: Sequence[str | int | float] = ("0", "1"),
    algos: Sequence[str] = ("gmm",),
    k_min: int = 2,
    k_max: int = 8,
    ward_k: int = 4,
    outdir: Path | str = "outputs/unsup/subclusters_c01",
) -> Dict[str, Path | List[Path] | None]:
    """Execute the subclustering workflow for the provided parent clusters."""

    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if cluster_col not in df.columns:
        raise ValueError(
            f"Cluster column {cluster_col!r} not present in input file."
        )

    parent_values = _normalize_parent_values(parent_targets)
    selected_algos = _sanitize_algorithms(algos)

    results: List[SubclusterResult] = []
    meta_files: List[Path] = []

    for parent in parent_values:
        result = process_parent_cluster(
            df,
            cluster_col,
            parent,
            outdir=output_dir,
            algos=selected_algos,
            k_min=k_min,
            k_max=k_max,
            ward_k=ward_k,
        )
        results.append(result)

        meta_path = output_dir / f"sub_meta_parent{parent}.json"
        meta_payload = {
            "parent_cluster": parent,
            "features_used": result.features_used,
            "algorithms": selected_algos,
            "gmm_k": result.gmm_k,
            "hdbscan_available": result.hdbscan_labels is not None,
            "ward_available": result.ward_labels is not None,
        }
        meta_path.write_text(json.dumps(meta_payload, indent=2))
        meta_files.append(meta_path)

    augmented_rows: List[pd.DataFrame] = []
    metrics_records: List[dict] = []

    for result in results:
        subset = df.loc[result.used_index].copy()

        if result.gmm_k is not None and result.gmm_labels is not None:
            subset[f"sub_gmm_k_parent{result.parent_value}"] = result.gmm_k
            subset[f"sub_gmm_label_parent{result.parent_value}"] = result.gmm_labels
            metrics_records.append(
                {
                    "parent": result.parent_value,
                    "algorithm": "gmm",
                    "k": result.gmm_k,
                    "features": result.features_used,
                }
            )

        if result.hdbscan_labels is not None:
            subset[f"sub_hdbscan_label_parent{result.parent_value}"] = (
                result.hdbscan_labels
            )
            positive = result.hdbscan_labels[result.hdbscan_labels >= 0]
            n_clusters = int(len(positive) and (positive.max() + 1))
            metrics_records.append(
                {
                    "parent": result.parent_value,
                    "algorithm": "hdbscan",
                    "k": n_clusters,
                    "features": result.features_used,
                }
            )

        if result.ward_labels is not None:
            subset[f"sub_ward_k_parent{result.parent_value}"] = ward_k
            subset[f"sub_ward_label_parent{result.parent_value}"] = result.ward_labels
            metrics_records.append(
                {
                    "parent": result.parent_value,
                    "algorithm": "ward",
                    "k": ward_k,
                    "features": result.features_used,
                }
            )

        augmented_rows.append(subset)

    augmented_path: Path | None = None
    metrics_path: Path | None = None

    if augmented_rows:
        augmented = pd.concat(augmented_rows, axis=0).sort_index()
        augmented_path = output_dir / "subclusters_parent_c01.csv"
        augmented.to_csv(augmented_path, index=False)

    if metrics_records:
        metrics = pd.DataFrame.from_records(metrics_records)
        metrics_path = output_dir / "subclusters_metrics_summary.csv"
        metrics.to_csv(metrics_path, index=False)

    return {
        "output_dir": output_dir,
        "augmented_csv": augmented_path,
        "metrics_csv": metrics_path,
        "meta_files": meta_files,
    }


def run(args: argparse.Namespace) -> None:
    run_subclustering(
        args.input_csv,
        cluster_col=args.cluster_col,
        parent_targets=args.parent_targets,
        algos=args.algos,
        k_min=args.k_min,
        k_max=args.k_max,
        ward_k=args.ward_k,
        outdir=args.outdir,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run secondary subclustering on selected parent clusters "
            "from the first-stage trial table."
        )
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to the trial-level CSV containing parent cluster labels.",
    )
    parser.add_argument(
        "--cluster-col",
        required=True,
        help="Column name holding the parent cluster assignments.",
    )
    parser.add_argument(
        "--parent-targets",
        nargs="+",
        default=["0", "1"],
        help="Parent cluster values to analyze (default: 0 and 1).",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=["gmm"],
        choices=["gmm", "hdbscan", "ward"],
        help="Algorithms to run (one or more of: gmm, hdbscan, ward).",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        default=2,
        help="Minimum number of GMM components to evaluate (default: 2).",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=8,
        help="Maximum number of GMM components to evaluate (default: 8).",
    )
    parser.add_argument(
        "--ward-k",
        type=int,
        default=4,
        help="Number of clusters to extract from Ward linkage (default: 4).",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/unsup/subclusters_c01",
        help="Directory where results will be written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()


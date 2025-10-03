"""Helper utilities for writing artifacts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class ArtifactPaths:
    base_dir: Path

    def report_path(self, model_name: str) -> Path:
        return self.base_dir / f"report_{model_name}.csv"

    def cluster_path(self, model_name: str) -> Path:
        return self.base_dir / f"trial_clusters_{model_name}.csv"

    def variance_plot(self, model_name: str) -> Path:
        return self.base_dir / f"pca_variance_{model_name}.png"

    def time_importance_plot(self, model_name: str) -> Path:
        return self.base_dir / f"time_importance_{model_name}.png"

    def eigenvector_plot(self, model_name: str) -> Path:
        return self.base_dir / f"pca_eigenvectors_{model_name}.png"

    def embedding_plot(self, model_name: str) -> Path:
        return self.base_dir / f"embedding_{model_name}.png"

    def component_csv(self, model_name: str) -> Path:
        return self.base_dir / f"motifs_{model_name}.csv"

    def component_plot(self, model_name: str) -> Path:
        return self.base_dir / f"motifs_{model_name}.png"

    def average_trace_plot(self, model_name: str) -> Path:
        return self.base_dir / f"cluster_average_trace_{model_name}.png"

    def response_auc_csv(self, model_name: str) -> Path:
        return self.base_dir / f"odor_response_auc_{model_name}.csv"

    def response_auc_summary_csv(self, model_name: str) -> Path:
        return self.base_dir / f"odor_response_cluster_summary_{model_name}.csv"

    def response_variance_plot(self, model_name: str) -> Path:
        return self.base_dir / f"odor_response_variance_{model_name}.png"

    def response_eigenvector_plot(self, model_name: str) -> Path:
        return self.base_dir / f"odor_response_eigenvectors_{model_name}.png"

    def response_embedding_plot(self, model_name: str) -> Path:
        return self.base_dir / f"odor_response_embedding_{model_name}.png"

    def response_scores_csv(self, model_name: str) -> Path:
        return self.base_dir / f"odor_response_pca_scores_{model_name}.csv"

    def response_auc_ranking_plot(self, model_name: str) -> Path:
        return self.base_dir / f"odor_response_auc_ranking_{model_name}.png"

    def response_auc_rankings_csv(self, model_name: str) -> Path:
        return self.base_dir / f"odor_response_auc_rankings_{model_name}.csv"


def ensure_output_dir(base: Path) -> ArtifactPaths:
    base.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(base_dir=base)


def write_report(path: Path, metrics: Dict[str, object]) -> None:
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)


def write_clusters(
    path: Path,
    metadata: pd.DataFrame,
    labels: Iterable[int],
    *,
    features: pd.DataFrame | None = None,
) -> None:
    out_df = metadata.copy()
    out_df["cluster_label"] = list(labels)

    if features is not None:
        # Align on index to avoid accidental row reordering and only keep
        # the overlapping entries to preserve downstream joins.
        aligned = features.reindex(out_df.index)
        for column in aligned.columns:
            out_df[column] = aligned[column]

    out_df.to_csv(path, index=False)


def write_time_importance(path: Path, importance_df: pd.DataFrame) -> None:
    importance_df.sort_values("time_index", inplace=True)
    importance_df.to_csv(path, index=False)


def write_components(path: Path, time_points: Iterable[float], components: np.ndarray) -> None:
    """Persist temporal motifs to CSV."""

    df = pd.DataFrame(
        components.T,
        columns=[f"component_{idx}" for idx in range(1, components.shape[0] + 1)],
    )
    df.insert(0, "time_index", list(time_points))
    df.to_csv(path, index=False)


__all__ = [
    "ArtifactPaths",
    "ensure_output_dir",
    "write_report",
    "write_clusters",
    "write_time_importance",
    "write_components",
]

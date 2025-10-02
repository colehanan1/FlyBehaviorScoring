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

    def embedding_plot(self, model_name: str) -> Path:
        return self.base_dir / f"embedding_{model_name}.png"

    def component_csv(self, model_name: str) -> Path:
        return self.base_dir / f"motifs_{model_name}.csv"

    def component_plot(self, model_name: str) -> Path:
        return self.base_dir / f"motifs_{model_name}.png"

    def average_trace_plot(self, model_name: str) -> Path:
        return self.base_dir / f"cluster_average_trace_{model_name}.png"


def ensure_output_dir(base: Path) -> ArtifactPaths:
    base.mkdir(parents=True, exist_ok=True)
    return ArtifactPaths(base_dir=base)


def write_report(path: Path, metrics: Dict[str, object]) -> None:
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)


def write_clusters(path: Path, metadata: pd.DataFrame, labels: Iterable[int]) -> None:
    out_df = metadata.copy()
    out_df["cluster_label"] = list(labels)
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

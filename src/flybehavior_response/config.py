"""Configuration and artifact management for flybehavior_response."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ISO_FMT = "%Y-%m-%dT%H-%M-%SZ"


@dataclass(slots=True)
class PipelineConfig:
    """Configuration describing a training run."""

    features: List[str]
    n_pcs: int
    use_raw_pca: bool
    seed: int
    models: List[str]
    trace_column_range: tuple[int, int]
    data_csv: str
    labels_csv: str
    file_hashes: Dict[str, str]
    class_balance: Dict[str, float]
    logreg_solver: str
    logreg_max_iter: int
    label_intensity_counts: Dict[str, int]
    label_weight_summary: Dict[str, float]
    label_weight_strategy: str
    trace_series_prefixes: List[str] = field(default_factory=list)
    use_trace_series: bool = True
    data_format: str = "tabular"
    dataset_granularity: str = "trial"
    group_column: str | None = None
    geometry_aggregations: List[str] = field(default_factory=list)
    geometry_normalization: str = "none"
    geometry_trial_summary: str | None = None
    geometry_feature_columns: List[str] = field(default_factory=list)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8")

    
    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        data["trace_column_range"] = tuple(data["trace_column_range"])
        data.setdefault("label_intensity_counts", {})
        data.setdefault("label_weight_summary", {})
        data.setdefault("label_weight_strategy", "proportional_intensity")
        data.setdefault("trace_series_prefixes", [])
        data.setdefault("use_trace_series", True)
        data.setdefault("data_format", "tabular")
        data.setdefault("dataset_granularity", "trial")
        data.setdefault("group_column", None)
        data.setdefault("geometry_aggregations", [])
        data.setdefault("geometry_normalization", "none")
        data.setdefault("geometry_trial_summary", None)
        data.setdefault("geometry_feature_columns", [])
        data["label_intensity_counts"] = {
            str(k): int(v) for k, v in data["label_intensity_counts"].items()
        }
        data["label_weight_summary"] = {
            str(k): float(v) for k, v in data["label_weight_summary"].items()
        }
        data["trace_series_prefixes"] = list(data["trace_series_prefixes"])
        data["geometry_aggregations"] = list(data["geometry_aggregations"])
        data["geometry_feature_columns"] = list(data["geometry_feature_columns"])
        if data["group_column"] is not None:
            data["group_column"] = str(data["group_column"])
        return cls(**data)


@dataclass(slots=True)
class RunArtifacts:
    """Metadata for paths generated in a run."""

    run_dir: Path
    config_path: Path
    metrics_path: Path
    models: Dict[str, Path] = field(default_factory=dict)
    plots: Dict[str, Path] = field(default_factory=dict)


def make_run_artifacts(base_dir: Path) -> RunArtifacts:
    timestamp = datetime.now(tz=timezone.utc).strftime(ISO_FMT)
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        metrics_path=run_dir / "metrics.json",
    )


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_class_balance(labels: List[int]) -> Dict[str, float]:
    total = len(labels)
    balance: Dict[str, float] = {}
    if total == 0:
        return balance
    ones = sum(labels)
    zeros = total - ones
    balance["0"] = zeros / total
    balance["1"] = ones / total
    return balance

"""Tests for CLI utility helpers."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("typer")

from flypca.cli import _load_config_or_default, report


def test_load_config_or_default_prefers_explicit(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text("foo: 1\n", encoding="utf-8")
    default = tmp_path / "default.yaml"
    default.write_text("foo: 2\n", encoding="utf-8")

    cfg = _load_config_or_default(explicit, default_path=default)
    assert cfg["foo"] == 1


def test_load_config_or_default_falls_back(tmp_path: Path) -> None:
    default = tmp_path / "default.yaml"
    default.write_text("bar: 3\n", encoding="utf-8")

    cfg = _load_config_or_default(None, default_path=default)
    assert cfg["bar"] == 3


def test_load_config_or_default_handles_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    cfg = _load_config_or_default(None, default_path=missing)
    assert cfg == {}


def test_report_deduplicates_clusters(tmp_path: Path) -> None:
    features = pd.DataFrame(
        {
            "trial_id": ["t1", "t2", "t3"],
            "fly_id": ["f1", "f2", "f3"],
            "pc1": [0.1, 0.2, 0.3],
            "pc2": [0.4, 0.5, 0.6],
            "latency": [0.5, 0.7, 0.6],
            "peak_value": [1.0, 0.9, 0.8],
            "snr": [2.0, 1.5, 1.8],
        }
    )
    features_path = tmp_path / "features.csv"
    features.to_csv(features_path, index=False)

    clusters = pd.DataFrame(
        {
            "trial_id": ["t1", "t1", "t2", "t4"],
            "fly_id": ["f1", "f1", "f2", "f4"],
            "cluster": [0, 1, 0, 1],
        }
    )
    clusters_path = tmp_path / "clusters.csv"
    clusters.to_csv(clusters_path, index=False)

    out_dir = tmp_path / "out"
    report(
        features_path=features_path,
        clusters_path=clusters_path,
        model=None,
        projections_dir=None,
        out_dir=out_dir,
    )

    scatter_path = out_dir / "figures" / "pc_scatter.png"
    assert scatter_path.exists()

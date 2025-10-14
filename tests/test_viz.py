"""Tests for visualization helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flypca import viz


def test_pc_scatter_validates_assignment_length(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "trial_id": ["t1", "t2"],
            "pc1": [0.1, 0.2],
            "pc2": [0.3, 0.4],
        }
    )

    # Works with aligned series
    series_assignments = pd.Series([0, 1], index=df["trial_id"])
    path = viz.pc_scatter(df, series_assignments, tmp_path)
    assert path.exists()

    # Raises a clear error on mismatch
    with pytest.raises(ValueError):
        viz.pc_scatter(df, np.array([0, 1, 2]), tmp_path)


def test_feature_violin_handles_sequence_assignments(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "trial_id": ["t1", "t2", "t3"],
            "latency": [0.5, 0.7, 0.6],
            "peak_value": [1.0, 0.8, 0.9],
            "snr": [2.0, 1.5, 1.8],
        }
    )
    assignments = [0, 0, 1]
    path = viz.feature_violin(df, assignments, ["latency"], tmp_path)
    assert path.exists()

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flybehavior_response.metrics import attach_metrics
from flybehavior_response.synthetic_fly import SyntheticConfig, SyntheticFlyGenerator


def _build_training_frame() -> tuple[pd.DataFrame, list[str]]:
    trace_columns = [f"dir_val_{idx}" for idx in range(3600)]
    rows = []
    for fly in ("flyA", "flyB"):
        for trial in range(2):
            label = trial % 2
            intensity = 2 if label == 1 else 0
            trace = np.zeros(3600, dtype=float)
            trace[:1260] = 0.5
            if label == 1:
                trace[1260:2460] = 5.0
                trace[2000] = 6.0
            else:
                trace[1260:2460] = 0.2
            trace[2460:] = 0.3
            row = {
                "dataset": "demo",
                "fly": fly,
                "fly_number": 1 if fly == "flyA" else 2,
                "trial_label": f"trial_{fly}_{trial}",
                "user_score_odor": label,
                "user_score_odor_intensity": intensity,
            }
            row.update({col: float(trace[idx]) for idx, col in enumerate(trace_columns)})
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame = attach_metrics(frame, trace_columns=trace_columns, fly_column="fly")
    return frame, trace_columns


def test_synthetic_generation_thresholds_match_spec(tmp_path):
    frame, trace_columns = _build_training_frame()
    config = SyntheticConfig(
        use_synthetics=True,
        synthetic_fly_ratio=1.0,
        synthetic_ops=("scale",),
        preview_synthetics=0,
        auto_filter_threshold=0.0,
        save_synthetics_dir=tmp_path / "artifacts",
        seed=123,
    )
    generator = SyntheticFlyGenerator(config=config, logger=logging.getLogger("test"))
    result = generator.generate(frame, trace_columns=trace_columns, fly_column="fly")

    assert not result.dataframe.empty
    assert (result.dataframe["is_synthetic"] == 1).all()
    assert all(col in result.dataframe.columns for col in trace_columns)

    thresholds = result.dataframe["threshold_trial"].to_numpy()
    mean_before = result.dataframe["mean_before_fly"].to_numpy()
    std_before = result.dataframe["std_before_trial"].to_numpy()
    assert np.allclose(thresholds, mean_before + 4.0 * std_before, atol=1e-6)

    assert result.manifest["synthetic_fly_id"].str.startswith("syn_").all()
    assert set(result.manifest["parent_fly_id"].unique()) <= {"flyA", "flyB"}
    assert "was_previewed" in result.manifest.columns


def test_synthetic_generation_keeps_training_isolation(tmp_path):
    frame, trace_columns = _build_training_frame()
    config = SyntheticConfig(
        use_synthetics=True,
        synthetic_fly_ratio=0.5,
        synthetic_ops=("jitter",),
        preview_synthetics=0,
        auto_filter_threshold=0.0,
        save_synthetics_dir=tmp_path,
        seed=999,
    )
    generator = SyntheticFlyGenerator(config=config, logger=logging.getLogger("test_isolation"))
    result = generator.generate(frame, trace_columns=trace_columns, fly_column="fly")
    kept_ids = {row for row in result.dataframe["synthetic_trial_id"]}
    original_ids = set(frame["trial_label"])
    assert kept_ids.isdisjoint(original_ids)

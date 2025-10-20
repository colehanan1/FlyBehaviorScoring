from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flybehavior_response.metrics import (
    attach_metrics,
    compute_metrics,
    detect_fly_column,
    discover_trace_columns,
    validate_metric_parity,
)


def _build_frame() -> pd.DataFrame:
    trace_columns = [f"dir_val_{idx}" for idx in range(3600)]
    rows = []
    for fly in ("flyA", "flyB"):
        for trial in range(2):
            base = np.zeros(3600, dtype=float)
            base[:1260] = 0.5 + 0.05 * trial
            if trial % 2 == 0:
                base[1260:2460] = 4.0
                label = 1
                intensity = 2
            else:
                base[1260:2460] = 0.2
                label = 0
                intensity = 0
            base[2460:] = 0.3
            row = {
                "dataset": "demo",
                "fly": fly,
                "fly_number": 1 if fly == "flyA" else 2,
                "trial_label": f"trial_{fly}_{trial}",
                "user_score_odor": label,
                "user_score_odor_intensity": intensity,
            }
            row.update({col: float(base[idx]) for idx, col in enumerate(trace_columns)})
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame = attach_metrics(frame, trace_columns=trace_columns, fly_column="fly")
    return frame


def test_discover_trace_columns_orders_numerically():
    df = pd.DataFrame({
        "dir_val_10": [0.0],
        "dir_val_2": [0.0],
        "dir_val_1": [0.0],
    })
    columns = discover_trace_columns(df)
    assert columns == ["dir_val_1", "dir_val_2", "dir_val_10"]


def test_validate_metric_parity_matches_original_values():
    frame = _build_frame()
    trace_columns = discover_trace_columns(frame)
    fly_column = detect_fly_column(frame)
    recomputed = validate_metric_parity(
        frame,
        trace_columns=trace_columns,
        fly_column=fly_column,
    )
    for column in ["AUC-Before", "AUC-During", "AUC-After", "TimeToPeak-During", "Peak-Value"]:
        assert np.allclose(frame[column].to_numpy(), recomputed[column].to_numpy(), atol=1e-6)


def test_compute_metrics_respects_missing_policy_nan():
    frame = _build_frame()
    trace_columns = discover_trace_columns(frame)
    fly_column = detect_fly_column(frame)
    metrics_nan = compute_metrics(
        frame,
        trace_columns=trace_columns,
        fly_column=fly_column,
        missing_ttpd_policy="nan",
    )
    assert metrics_nan["TimeToPeak-During"].isna().any()

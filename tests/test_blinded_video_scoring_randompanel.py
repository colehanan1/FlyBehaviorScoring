"""Tests for RandomPanel support in the blinded video scorer.

RandomPanel trials are labelled ``training_<N>_<odor>`` (e.g. ``training_5_hexanol``)
rather than the opto datasets' ``testing_<N>``. The scorer must surface those
odour trials, skip light-only ``training_<N>`` trials that carry no odour suffix,
optionally restrict to a single dataset, and still resolve videos whose filenames
use the odour-free ``training_<N>`` stem — all without regressing the existing
``testing_<N>`` behaviour.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# The labeler script lives in scripts/label, which is not on the src path.
LABEL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "label"
if str(LABEL_DIR) not in sys.path:
    sys.path.insert(0, str(LABEL_DIR))

import blinded_video_scoring as bvs  # noqa: E402

RANDOMPANEL_PATTERN = r"training_\d+_"
TESTING_PATTERN = r"testing_\d+"

_COLS = ["dataset", "fly", "fly_number", "trial_type", "trial_label"]


def _df(rows: list[list]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_COLS)


def test_randompanel_pattern_keeps_odor_trials_and_drops_light_only():
    df = _df([
        ["RandomPanel-24-1", "july_04_batch_1", 1, "testing", "training_5_hexanol"],
        ["RandomPanel-24-1", "july_04_batch_1", 1, "testing", "training_2_3-octonol"],
        ["RandomPanel-24-1", "july_04_batch_1", 1, "testing", "training_15"],  # light-only: no odour suffix
    ])
    out = bvs.filter_scorable_trials(df, label_pattern=RANDOMPANEL_PATTERN)
    assert set(out["trial_label"]) == {"training_5_hexanol", "training_2_3-octonol"}


def test_dataset_filter_restricts_to_one_dataset():
    df = _df([
        ["RandomPanel-24-1", "fA", 1, "testing", "training_5_hexanol"],
        ["RandomPanel-24-0.1", "fB", 1, "testing", "training_5_hexanol"],
    ])
    out = bvs.filter_scorable_trials(
        df, label_pattern=RANDOMPANEL_PATTERN, dataset_filter="RandomPanel-24-1"
    )
    assert set(out["dataset"]) == {"RandomPanel-24-1"}
    assert len(out) == 1


def test_default_testing_pattern_is_unchanged():
    """Regression guard: the opto datasets' behaviour must not change."""
    df = _df([
        ["Hex-Control-24-0.01", "fA", 1, "testing", "testing_1"],
        ["Hex-Control-24-0.01", "fA", 1, "testing", "testing_11"],        # excluded testing_11
        ["Hex-Control-24-0.01", "fA", 1, "testing", "testing_2_light"],   # excluded light
        ["RandomPanel-24-1", "fB", 1, "testing", "training_5_hexanol"],   # dropped by testing pattern
        ["Hex-Control-24-0.01", "fA", 1, "training", "testing_3"],        # dropped: not a testing trial_type
    ])
    out = bvs.filter_scorable_trials(df, label_pattern=TESTING_PATTERN)
    assert set(out["trial_label"]) == {"testing_1"}


@pytest.mark.parametrize(
    "label,expected",
    [
        ("training_5_hexanol", "training_5"),
        ("training_2_3-octonol", "training_2"),
        ("testing_1_fly1_angle_distance_rms_envelope", "testing_1"),
        ("training_15", "training_15"),
        ("testing_11", "testing_11"),
    ],
)
def test_video_core_label_strips_odor_suffix(label, expected):
    assert bvs._video_core_label(label) == expected


def test_randompanel_preset_registered_and_scoped():
    """The per-dataset presets exist and carry the right filter + pattern."""
    assert "randompanel" in bvs.DATASETS
    assert "RandomPanel-24-1" in bvs.DATASETS
    rp = bvs.DATASETS["RandomPanel-24-1"]
    assert rp["dataset_filter"] == "RandomPanel-24-1"
    assert rp["label_pattern"] == RANDOMPANEL_PATTERN
    assert bvs.DATASETS["randompanel"]["dataset_filter"] is None

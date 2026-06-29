"""Regression tests for signal-feature computation on degenerate slices.

Per-trial odor timing + ``n_frames`` truncation can collapse the ``before`` /
``during`` / ``after`` windows down to zero or one frame (e.g. a trial whose
real duration is shorter than its configured odor-off time). The Welch PSD must
degrade gracefully instead of raising ``noverlap must be less than nperseg``.
"""

from __future__ import annotations

import math

import numpy as np

from flybehavior_response.xgb_ordinal import SIGNAL_FEATURES, compute_signal_features


def _assert_complete(out: dict) -> None:
    assert set(SIGNAL_FEATURES).issubset(out.keys())
    for k in SIGNAL_FEATURES:
        v = out[k]
        assert isinstance(v, float)
        assert not math.isinf(v), f"{k} is inf"


def test_during_collapsed_to_single_frame_does_not_crash() -> None:
    # n_frames truncates below odor_off so `during` collapses to 1 frame and
    # `after` is empty -- the exact production crash.
    trace = np.linspace(0.0, 1.0, 3600)
    out = compute_signal_features(
        trace, fps=40.0, odor_on_idx=10, odor_off_idx=2480, n_frames=11
    )
    _assert_complete(out)


def test_empty_before_slice_does_not_crash() -> None:
    # odor_on at frame 0 -> `before` is empty.
    trace = np.linspace(0.0, 1.0, 200)
    out = compute_signal_features(
        trace, fps=40.0, odor_on_idx=0, odor_off_idx=120, n_frames=200
    )
    _assert_complete(out)


def test_very_short_trace_does_not_crash() -> None:
    out = compute_signal_features(
        np.array([0.3, 0.7]), fps=40.0, odor_on_idx=1, odor_off_idx=2, n_frames=2
    )
    _assert_complete(out)


def test_normal_trace_still_produces_features() -> None:
    rng = np.random.default_rng(0)
    trace = rng.standard_normal(3600)
    out = compute_signal_features(
        trace, fps=40.0, odor_on_idx=1280, odor_off_idx=2480, n_frames=3600
    )
    _assert_complete(out)

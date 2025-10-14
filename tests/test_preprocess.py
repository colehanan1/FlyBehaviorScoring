"""Tests for preprocessing utilities."""

from __future__ import annotations

import numpy as np

from flypca.preprocess import PreprocessConfig, SmoothingConfig, preprocess_trial


def test_baseline_zscore_centers_to_zero(synthetic_dataset) -> None:
    trials, _ = synthetic_dataset
    trial = trials[0]
    config = PreprocessConfig(fps=trial.fps, pre_s=2.0, post_s=2.0, smoothing=SmoothingConfig())
    time, zscored, mean, std = preprocess_trial(trial, config)
    baseline_mask = time < 0
    assert np.isclose(np.mean(zscored[baseline_mask]), 0, atol=0.05)
    assert np.isclose(np.std(zscored[baseline_mask]), 1, atol=0.1)
    assert np.isclose(mean, 0, atol=0.2)
    assert std > 0

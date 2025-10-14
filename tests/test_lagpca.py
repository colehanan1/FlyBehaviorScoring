"""Tests for lag-PCA fitting."""

from __future__ import annotations

import numpy as np

from flypca.lagpca import fit_lag_pca_for_trials, hankel_embed


def test_hankel_embed_shape() -> None:
    data = np.arange(10, dtype=float)
    embedded = hankel_embed(data, lag=4)
    assert embedded.shape == (7, 4)


def test_lag_pca_explained_variance(synthetic_dataset) -> None:
    trials, meta = synthetic_dataset
    reactor_ids = set(meta.loc[meta["reaction"] == 1, "trial_id"][:40])
    reactor_trials = [trial for trial in trials if trial.trial_id in reactor_ids]
    config = {
        "pre_s": 2.0,
        "post_s": 2.0,
        "lag_ms": 250,
        "n_components": 5,
        "smoothing": {"enable": True, "savgol_window_ms": 151, "savgol_poly": 3},
    }
    result = fit_lag_pca_for_trials(reactor_trials, config)
    assert result.model.components_.shape[0] == result.model.n_components_
    assert result.explained_variance_ratio_[0] > 0.5
    assert result.explained_variance_ratio_[:2].sum() > 0.7

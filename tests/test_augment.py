"""Unit tests for augmentation utilities."""

from __future__ import annotations

import numpy as np

from flybehavior_response import augment
from flybehavior_response.synthetic import SyntheticConfig, make_synthetics


def _sample_series(n_samples: int = 20, time_steps: int = 50, channels: int = 3) -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.normal(size=(n_samples, time_steps, channels)).astype(np.float32)


def test_augmentations_preserve_shape() -> None:
    series = _sample_series(1)
    x = series[0]
    for fn in (augment.jitter, augment.scale, augment.time_shift, augment.crop_resize):
        result = fn(x, rng=np.random.default_rng(42))
        assert result.shape == x.shape


def test_augmentations_deterministic_with_seed() -> None:
    x = _sample_series(1)[0]
    seed = 99
    for fn in (augment.jitter, augment.scale, augment.time_shift, augment.crop_resize):
        rng_a = np.random.default_rng(seed)
        rng_b = np.random.default_rng(seed)
        np.testing.assert_allclose(fn(x, rng=rng_a), fn(x, rng=rng_b))


def test_mixup_finite_output() -> None:
    x_a, x_b = _sample_series(2)
    result = augment.mixup_same_class(x_a, x_b, rng=np.random.default_rng(7))
    assert result.shape == x_a.shape
    assert np.isfinite(result).all()


def test_make_synthetics_ratio_target() -> None:
    base = _sample_series(40)
    labels = np.array([0] * 20 + [1] * 20, dtype=int)
    ids = [f"sample_{idx}" for idx in range(len(base))]
    config = SyntheticConfig(synthetic_ratio=0.5, synthetic_ops=("jitter",), seed=202)
    X_syn, y_syn, parents, ops, seeds = make_synthetics(base, labels, ids, [0, 1], config)
    assert X_syn.shape[1:] == base.shape[1:]
    ratio = len(X_syn) / len(base)
    assert abs(ratio - config.synthetic_ratio) <= 0.02
    assert set(y_syn.tolist()) <= {0, 1}
    assert len(parents) == len(X_syn) == len(ops) == len(seeds)
    # Deterministic with same seed
    X_syn_b, y_syn_b, parents_b, ops_b, seeds_b = make_synthetics(base, labels, ids, [0, 1], config)
    np.testing.assert_allclose(X_syn, X_syn_b)
    np.testing.assert_array_equal(y_syn, y_syn_b)
    assert parents == parents_b
    np.testing.assert_array_equal(seeds, seeds_b)

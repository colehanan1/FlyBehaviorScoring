"""Augmentation utilities for synthetic time-series generation."""

from __future__ import annotations

import numpy as np


def jitter(x: np.ndarray, sigma: float = 0.02, rng: np.random.Generator | None = None) -> np.ndarray:
    """Additive noise scaled to the per-channel peak-to-peak range."""

    rng = np.random.default_rng() if rng is None else rng
    scale = sigma * np.ptp(x, axis=0, keepdims=True)
    noise = rng.normal(0.0, scale, size=x.shape)
    return x + noise


def scale(
    x: np.ndarray, low: float = 0.9, high: float = 1.1, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Apply per-channel multiplicative scaling."""

    rng = np.random.default_rng() if rng is None else rng
    factors = rng.uniform(low, high, size=(1, x.shape[1]))
    return x * factors


def time_shift(
    x: np.ndarray, max_frac: float = 0.02, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Circularly shift the sequence along the time axis."""

    rng = np.random.default_rng() if rng is None else rng
    T = x.shape[0]
    shift = int(rng.uniform(-max_frac, max_frac) * T)
    if shift == 0:
        return x
    return np.roll(x, shift, axis=0)


def crop_resize(
    x: np.ndarray, keep: float = 0.95, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Randomly crop a segment and resize back to the original length."""

    rng = np.random.default_rng() if rng is None else rng
    T, D = x.shape
    keep = float(np.clip(keep, 0.0, 1.0))
    crop_len = max(1, int(T * keep))
    start = int(rng.integers(0, max(1, T - crop_len + 1)))
    segment = x[start : start + crop_len]
    base_index = np.arange(crop_len)
    target_index = np.linspace(0, crop_len - 1, num=T)
    resized = np.stack(
        [np.interp(target_index, base_index, segment[:, d]) for d in range(D)],
        axis=1,
    )
    return resized


def mixup_same_class(
    x_a: np.ndarray,
    x_b: np.ndarray,
    alpha: float = 0.2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Perform mixup between two samples of the same class."""

    rng = np.random.default_rng() if rng is None else rng
    lam = float(rng.beta(alpha, alpha))
    return lam * x_a + (1.0 - lam) * x_b


__all__ = [
    "jitter",
    "scale",
    "time_shift",
    "crop_resize",
    "mixup_same_class",
]


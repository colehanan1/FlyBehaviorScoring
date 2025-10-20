"""NumPy-based augmentation operators for synthetic fly generation."""

from __future__ import annotations

from typing import Callable, Dict, Iterable

import numpy as np


def _ensure_2d(trace: np.ndarray) -> np.ndarray:
    if trace.ndim == 1:
        return trace[:, None]
    return trace


def jitter(trace: np.ndarray, rng: np.random.Generator, *, scale: float = 0.02) -> np.ndarray:
    base = _ensure_2d(trace)
    std = np.nanstd(base)
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    noise = rng.normal(loc=0.0, scale=scale * std, size=base.shape)
    return (base + noise).astype(float)


def scale(trace: np.ndarray, rng: np.random.Generator, *, min_scale: float = 0.9, max_scale: float = 1.1) -> np.ndarray:
    factor = rng.uniform(min_scale, max_scale)
    base = _ensure_2d(trace)
    return (base * factor).astype(float)


def time_shift(trace: np.ndarray, rng: np.random.Generator, *, max_frames: int | None = None) -> np.ndarray:
    base = _ensure_2d(trace)
    total_frames = base.shape[0]
    if max_frames is None:
        max_frames = max(1, total_frames // 12)
    shift = rng.integers(-max_frames, max_frames + 1)
    if shift == 0:
        return base.astype(float)
    rolled = np.roll(base, shift=shift, axis=0)
    if shift > 0:
        rolled[:shift] = base[:1]
    else:
        rolled[shift:] = base[-1:]
    return rolled.astype(float)


def crop_resize(trace: np.ndarray, rng: np.random.Generator, *, min_ratio: float = 0.7) -> np.ndarray:
    base = _ensure_2d(trace)
    total_frames = base.shape[0]
    min_frames = max(int(total_frames * min_ratio), 2)
    start = rng.integers(0, total_frames - min_frames + 1)
    end = rng.integers(start + min_frames, total_frames + 1)
    segment = base[start:end]
    target_indices = np.linspace(0, segment.shape[0] - 1, total_frames)
    source_indices = np.arange(segment.shape[0])
    resized = np.empty((total_frames, base.shape[1]), dtype=float)
    for dim in range(base.shape[1]):
        resized[:, dim] = np.interp(target_indices, source_indices, segment[:, dim])
    return resized


def mixup_same_class(
    trace_a: np.ndarray,
    trace_b: np.ndarray,
    rng: np.random.Generator,
    *,
    alpha: float = 0.2,
) -> np.ndarray:
    if alpha <= 0:
        raise ValueError("MixUp alpha must be positive")
    lambda_sample = float(rng.beta(alpha, alpha))
    a = _ensure_2d(trace_a)
    b = _ensure_2d(trace_b)
    return (lambda_sample * a + (1.0 - lambda_sample) * b).astype(float)


OPERATIONS: Dict[str, Callable[..., np.ndarray]] = {
    "jitter": jitter,
    "scale": scale,
    "time_shift": time_shift,
    "crop_resize": crop_resize,
    "mixup_same_class": mixup_same_class,
}


def available_ops() -> Iterable[str]:
    return tuple(OPERATIONS.keys())


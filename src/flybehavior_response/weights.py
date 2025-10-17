"""Utilities for handling sample-weight strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ensure_same_length(
    data: pd.DataFrame, labels: pd.Series, weights: pd.Series
) -> None:
    if not (len(data) == len(labels) == len(weights)):
        raise ValueError("Data, labels, and weights must share the same length for expansion.")


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    if np.any(weights < 0):
        raise ValueError("Sample weights must be non-negative.")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Sample weights must sum to a positive finite value.")
    return weights / total


def expand_samples_by_weight(
    data: pd.DataFrame,
    labels: pd.Series,
    weights: pd.Series,
    *,
    target_size: int | None = None,
    max_multiplier: float = 6.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Replicate samples so unweighted estimators approximate sample weights.

    Parameters
    ----------
    data, labels, weights:
        Training inputs with aligned indices.
    target_size:
        Desired number of rows after expansion. Defaults to ``len(data)``. The
        value is automatically clipped so the expanded data never exceeds
        ``len(data) * max_multiplier``.
    max_multiplier:
        Hard cap on the ratio between expanded and original sample counts. This
        avoids exploding dataset sizes when weights are very large.
    """

    _ensure_same_length(data, labels, weights)
    if len(data) == 0:
        return data.copy(), labels.copy()

    base_size = len(data)
    max_size = int(np.ceil(base_size * max_multiplier))
    desired = int(target_size) if target_size is not None else base_size
    desired = max(base_size, desired)
    desired = min(desired, max_size)

    weight_array = weights.to_numpy(dtype=float)
    normalized = _normalize_weights(weight_array)
    raw_counts = normalized * desired
    counts = np.floor(raw_counts).astype(int)
    deficit = desired - counts.sum()
    if deficit > 0:
        fractional = raw_counts - counts
        order = np.argsort(fractional)[::-1]
        for idx in order[:deficit]:
            counts[idx] += 1

    positive_mask = weight_array > 0
    zero_positive = np.where((counts == 0) & positive_mask)[0]
    if zero_positive.size > 0:
        counts[zero_positive] = 1
        overflow = counts.sum() - desired
        if overflow > 0:
            order = np.argsort(raw_counts)[::-1]
            for idx in order:
                if overflow == 0:
                    break
                if counts[idx] <= 1 and positive_mask[idx]:
                    continue
                removable = min(counts[idx] - (1 if positive_mask[idx] else 0), overflow)
                if removable > 0:
                    counts[idx] -= removable
                    overflow -= removable

    total = counts.sum()
    if total > desired:
        overflow = total - desired
        order = np.argsort(counts)[::-1]
        for idx in order:
            if overflow == 0:
                break
            min_allowed = 1 if positive_mask[idx] else 0
            removable = min(counts[idx] - min_allowed, overflow)
            if removable > 0:
                counts[idx] -= removable
                overflow -= removable

    expanded_index = np.repeat(np.arange(base_size), counts)

    expanded_data = data.iloc[expanded_index].reset_index(drop=True)
    expanded_labels = labels.iloc[expanded_index].reset_index(drop=True)
    return expanded_data, expanded_labels

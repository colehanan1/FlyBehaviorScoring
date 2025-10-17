"""Utilities for handling sample-weight strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def expand_samples_by_weight(
    data: pd.DataFrame, labels: pd.Series, weights: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Replicate samples according to integer weights."""

    if not (len(data) == len(labels) == len(weights)):
        raise ValueError("Data, labels, and weights must share the same length for expansion.")

    integer_weights = np.maximum(1, np.rint(weights.to_numpy()).astype(int))
    expanded_index = np.repeat(np.arange(len(integer_weights)), integer_weights)

    expanded_data = data.iloc[expanded_index].reset_index(drop=True)
    expanded_labels = labels.iloc[expanded_index].reset_index(drop=True)
    return expanded_data, expanded_labels

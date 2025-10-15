"""Helpers for working with wide time-series tables."""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

import pandas as pd

__all__ = ["find_series_columns"]


def find_series_columns(df: pd.DataFrame, prefixes: Sequence[str]) -> Dict[str, List[str]]:
    """Locate contiguous 0-based time-series columns for each prefix.

    Parameters
    ----------
    df:
        Input dataframe containing time-series columns.
    prefixes:
        Prefix strings (including trailing separators, e.g. ``"dir_val_"`` or
        ``"eye_x_f"``) that identify per-frame series.

    Returns
    -------
    dict
        Mapping of prefix -> ordered list of column names matching that prefix.

    Raises
    ------
    ValueError
        If any prefix is missing, has non-contiguous indices, or indices do not
        start at zero.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "dir_val_0": [0.0],
    ...     "dir_val_1": [0.1],
    ...     "alt_0": [1.0],
    ...     "alt_1": [1.1],
    ... })
    >>> find_series_columns(df, ["dir_val_", "alt_"])
    {'dir_val_': ['dir_val_0', 'dir_val_1'], 'alt_': ['alt_0', 'alt_1']}
    """

    if not prefixes:
        raise ValueError("At least one prefix must be provided for detection.")

    result: Dict[str, List[str]] = {}
    for prefix in prefixes:
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        matches: List[tuple[int, str]] = []
        for column in df.columns:
            match = pattern.match(column)
            if match:
                matches.append((int(match.group(1)), column))
        if not matches:
            raise ValueError(f"No columns found for prefix '{prefix}'.")
        matches.sort(key=lambda pair: pair[0])
        indices = [idx for idx, _ in matches]
        expected = list(range(len(indices)))
        if indices != expected:
            raise ValueError(
                "Prefix '{prefix}' columns must provide contiguous indices starting at 0. "
                "Found indices {indices}.".format(prefix=prefix, indices=indices)
            )
        result[prefix] = [name for _, name in matches]

    lengths = {prefix: len(columns) for prefix, columns in result.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            "All prefixes must share the same frame count. Got lengths: {lengths}.".format(
                lengths=lengths
            )
        )
    return result

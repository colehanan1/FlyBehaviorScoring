"""Dataframe parsing helpers."""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

__all__ = [
    "choose_signal_column",
    "extract_time_seconds",
    "parse_fly_trial",
]


def choose_signal_column(df: pd.DataFrame) -> np.ndarray:
    """Pick a likely signal column from a dataframe."""
    preferred = ['RMS', 'rms', 'envelope', 'Envelope', 'distance', 'Eye_Prob_Dist']
    for p in preferred:
        if p in df.columns:
            return df[p].values
    candidates = [
        c
        for c in df.columns
        if c.lower() not in ('time', 'timestamp', 'frame', 'frames', 't', 'ms', 'seconds')
    ]
    if candidates:
        return df[candidates[0]].values
    return df.iloc[:, -1].values


def extract_time_seconds(df: pd.DataFrame, fps: float) -> np.ndarray:
    """Derive a time axis in seconds from dataframe columns or FPS."""
    time_like: list[np.ndarray] = []
    for col in df.columns:
        lower = col.lower()
        if any(token in lower for token in ('time', 'second', 'timestamp')):
            series = pd.to_numeric(df[col], errors='coerce')
            if series.notna().sum() > 0:
                time_like.append(series.to_numpy(dtype=float))
    if time_like:
        return time_like[0]
    n = len(df)
    fps_val = fps if fps and fps > 0 else 0.0
    if fps_val > 0:
        return np.arange(n, dtype=float) / float(fps_val)
    return np.arange(n, dtype=float)


def parse_fly_trial(path: str) -> tuple[str | None, str | None]:
    """Try to parse fly_id and trial_id from path/filename digits."""
    fname = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))

    fly_id: str | None = None
    trial_id: str | None = None

    if parent.lower().startswith('fly'):
        digits = ''.join(filter(str.isdigit, parent))
        if digits:
            fly_id = digits
    elif parent.isdigit():
        fly_id = parent

    nums = re.findall(r'\d+', fname)
    if nums:
        if len(nums) >= 2:
            fly_id = fly_id or nums[0]
            trial_id = nums[1]
        else:
            if fly_id is None:
                fly_id = nums[0]
            else:
                trial_id = nums[0]
    return fly_id, trial_id

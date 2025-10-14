"""Synthetic data generation for demos and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from .io import TrialTimeseries


@dataclass
class SyntheticTrialMetadata:
    trial_id: str
    fly_id: str
    reaction: int
    amplitude: float
    latency: float


def _generate_trace(
    n_samples: int,
    fps: float,
    odor_on_idx: int,
    reaction: bool,
    amplitude: float,
    latency: float,
    rng: np.random.Generator,
) -> np.ndarray:
    time = np.arange(n_samples) / fps
    noise = rng.normal(scale=0.1, size=n_samples)
    baseline = noise
    if reaction:
        t = time - (odor_on_idx / fps + latency)
        sigmoid = 1 / (1 + np.exp(-10 * t))
        decay = np.exp(-t.clip(min=0) / 1.0)
        response = amplitude * sigmoid * decay
        baseline += response
    return baseline


def generate_synthetic_trials(
    n_flies: int = 4,
    trials_per_fly: int = 12,
    fps: float = 40.0,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    seed: int = 123,
) -> Tuple[List[TrialTimeseries], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    odor_on_idx = int(pre_s * fps)
    n_samples = int((pre_s + post_s + 1.0) * fps)
    trials: List[TrialTimeseries] = []
    meta_rows = []
    for fly_idx in range(n_flies):
        fly_id = f"fly_{fly_idx}"
        for trial_idx in range(trials_per_fly):
            trial_id = f"trial_{fly_idx}_{trial_idx}"
            reaction = rng.random() < 0.5
            amplitude = rng.uniform(1.5, 2.5) if reaction else rng.uniform(0.2, 0.5)
            latency = rng.uniform(0.2, 0.6) if reaction else rng.uniform(0.6, 1.0)
            trace = _generate_trace(
                n_samples,
                fps,
                odor_on_idx,
                reaction,
                amplitude,
                latency,
                rng,
            )
            time = np.arange(n_samples) / fps
            trial = TrialTimeseries(
                trial_id=trial_id,
                fly_id=fly_id,
                fps=fps,
                odor_on_idx=odor_on_idx,
                odor_off_idx=odor_on_idx + int(1.0 * fps),
                time=time,
                distance=trace,
            )
            trials.append(trial)
            meta_rows.append(
                {
                    "trial_id": trial_id,
                    "fly_id": fly_id,
                    "reaction": int(reaction),
                    "amplitude": float(amplitude),
                    "latency": float(latency),
                }
            )
    meta = pd.DataFrame(meta_rows)
    return trials, meta

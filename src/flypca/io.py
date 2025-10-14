"""Data loading utilities for flypca."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrialTimeseries:
    """Container for a single trial time series."""

    trial_id: str
    fly_id: str
    fps: float
    odor_on_idx: int
    odor_off_idx: Optional[int]
    time: np.ndarray
    distance: np.ndarray

    def validate(self) -> None:
        """Validate the trial structure."""
        if self.fps <= 0:
            raise ValueError(f"FPS must be positive for trial {self.trial_id}.")
        if self.odor_on_idx < 0:
            raise ValueError(f"odor_on_idx must be non-negative for trial {self.trial_id}.")
        if self.odor_off_idx is not None and self.odor_off_idx <= self.odor_on_idx:
            raise ValueError(
                f"odor_off_idx must be greater than odor_on_idx for trial {self.trial_id}."
            )
        if self.time.ndim != 1 or self.distance.ndim != 1:
            raise ValueError("time and distance must be one-dimensional arrays.")
        if len(self.time) != len(self.distance):
            raise ValueError("time and distance must have equal length.")
        if not np.all(np.diff(self.time) >= 0):
            raise ValueError(f"Time must be monotonic for trial {self.trial_id}.")
        if self.odor_on_idx >= len(self.distance):
            raise ValueError(
                f"odor_on_idx {self.odor_on_idx} out of bounds for trial {self.trial_id}."
            )
        if self.odor_off_idx is not None and self.odor_off_idx > len(self.distance):
            raise ValueError(
                f"odor_off_idx {self.odor_off_idx} out of bounds for trial {self.trial_id}."
            )


def _coerce_column(df: pd.DataFrame, names: Sequence[str]) -> Optional[pd.Series]:
    for name in names:
        if name in df.columns:
            return df[name]
    return None


def _load_manifest(directory: Path) -> pd.DataFrame:
    manifest_path = directory / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Directory {directory} must contain manifest.csv with trial metadata."
        )
    manifest = pd.read_csv(manifest_path)
    required = {"path", "trial_id", "fly_id", "odor_on_idx"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    return manifest


def _resolve_fps(row: pd.Series, default_fps: Optional[float]) -> float:
    if "fps" in row and not np.isnan(row["fps"]):
        return float(row["fps"])
    if default_fps is None:
        raise ValueError(
            "FPS not provided in manifest or config; specify fps in configs/default.yaml."
        )
    return float(default_fps)


def _load_trial_from_df(df: pd.DataFrame, fps: float) -> TrialTimeseries:
    trial_id = str(df["trial_id"].iloc[0])
    fly_id = str(df["fly_id"].iloc[0])
    odor_on_idx = int(df["odor_on_idx"].iloc[0])
    odor_off_idx = int(df["odor_off_idx"].iloc[0]) if "odor_off_idx" in df else None
    distance = df["distance"].to_numpy(dtype=float)
    time_series = _coerce_column(df, ["time", "t", "timestamp"])
    if time_series is not None:
        time = time_series.to_numpy(dtype=float)
    else:
        time = np.arange(len(distance), dtype=float) / fps
    trial = TrialTimeseries(
        trial_id=trial_id,
        fly_id=fly_id,
        fps=fps,
        odor_on_idx=odor_on_idx,
        odor_off_idx=odor_off_idx,
        time=time,
        distance=distance,
    )
    trial.validate()
    return trial


def _load_from_manifest(directory: Path, default_fps: Optional[float]) -> List[TrialTimeseries]:
    manifest = _load_manifest(directory)
    trials: List[TrialTimeseries] = []
    for _, row in manifest.iterrows():
        trial_path = directory / row["path"]
        if not trial_path.exists():
            raise FileNotFoundError(f"Missing trial CSV: {trial_path}")
        df = pd.read_csv(trial_path)
        for required in ("distance",):
            if required not in df.columns:
                raise ValueError(f"Trial file {trial_path} missing column {required}")
        df["trial_id"] = row["trial_id"]
        df["fly_id"] = row["fly_id"]
        df["odor_on_idx"] = row["odor_on_idx"]
        if "odor_off_idx" in row and not pd.isna(row["odor_off_idx"]):
            df["odor_off_idx"] = int(row["odor_off_idx"])
        fps = _resolve_fps(row, default_fps)
        trials.append(_load_trial_from_df(df, fps=fps))
    LOGGER.info("Loaded %d trials from manifest %s", len(trials), directory)
    return trials


def _load_stacked_csv(path: Path, default_fps: Optional[float]) -> List[TrialTimeseries]:
    df = pd.read_csv(path)
    required = {"trial_id", "fly_id", "distance", "odor_on_idx"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Stacked CSV missing required columns: {missing}")
    grouped = df.groupby("trial_id", sort=False)
    trials: List[TrialTimeseries] = []
    for trial_id, group in grouped:
        fps = (
            float(group["fps"].iloc[0])
            if "fps" in group.columns and not pd.isna(group["fps"].iloc[0])
            else (default_fps if default_fps is not None else None)
        )
        if fps is None:
            raise ValueError(f"FPS missing for trial {trial_id} and no default provided.")
        trials.append(_load_trial_from_df(group, fps=fps))
    LOGGER.info("Loaded %d trials from stacked CSV %s", len(trials), path)
    return trials


def load_trials(path: str | Path, config: Optional[dict[str, object]] = None) -> List[TrialTimeseries]:
    """Load trials from a CSV file or directory."""

    config = config or {}
    default_fps = float(config.get("fps", 0.0)) if "fps" in config else None
    path = Path(path)
    if path.is_dir():
        return _load_from_manifest(path, default_fps=default_fps)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            manifest_info = json.load(f)
        if "trials" not in manifest_info:
            raise ValueError("JSON manifest must contain 'trials' key.")
        trials: List[TrialTimeseries] = []
        for item in manifest_info["trials"]:
            trial_path = Path(item["path"])
            df = pd.read_csv(trial_path)
            df["trial_id"] = item["trial_id"]
            df["fly_id"] = item["fly_id"]
            df["odor_on_idx"] = item["odor_on_idx"]
            if "odor_off_idx" in item:
                df["odor_off_idx"] = item["odor_off_idx"]
            fps = float(item.get("fps", default_fps)) if item.get("fps") or default_fps else None
            if fps is None:
                raise ValueError(f"FPS missing for trial {item['trial_id']}")
            trials.append(_load_trial_from_df(df, fps=fps))
        return trials
    if path.is_file():
        return _load_stacked_csv(path, default_fps=default_fps)
    raise FileNotFoundError(f"Could not locate data path: {path}")

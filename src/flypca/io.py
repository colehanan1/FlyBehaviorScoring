"""Data loading utilities for flypca."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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


def _read_csv(path: Path, read_cfg: Optional[Dict[str, object]]) -> pd.DataFrame:
    kwargs: Dict[str, object] = {"low_memory": False}
    if read_cfg:
        kwargs.update(read_cfg)
    LOGGER.debug("Reading CSV %s with options %s", path, kwargs)
    return pd.read_csv(path, **kwargs)


def _rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    missing: List[str] = []
    for src, dst in mapping.items():
        if src in df.columns:
            if src != dst:
                rename_map[src] = dst
        elif dst in df.columns:
            continue
        else:
            missing.append(src)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.rename(columns=rename_map)


def _load_stacked_csv(
    path: Path,
    default_fps: Optional[float],
    config: Optional[Dict[str, object]],
    read_cfg: Optional[Dict[str, object]],
) -> List[TrialTimeseries]:
    df = _read_csv(path, read_cfg)
    config = config or {}
    mapping = {
        str(config.get("trial_id_column", "trial_id")): "trial_id",
        str(config.get("fly_id_column", "fly_id")): "fly_id",
        str(config.get("distance_column", "distance")): "distance",
        str(config.get("odor_on_column", "odor_on_idx")): "odor_on_idx",
    }
    optional_mapping = {
        str(config.get("time_column")): "time",
        str(config.get("odor_off_column")): "odor_off_idx",
        str(config.get("fps_column", "fps")): "fps",
    }
    mapping.update({k: v for k, v in optional_mapping.items() if k and k in df.columns})
    df = _rename_columns(df, mapping)
    required = {"trial_id", "fly_id", "distance", "odor_on_idx"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Stacked CSV missing required columns: "
            f"{missing}. Configure io.stacked.* in configs/default.yaml to map columns."
        )
    grouped = df.groupby("trial_id", sort=False)
    trials: List[TrialTimeseries] = []
    for trial_id, group in grouped:
        fps = None
        if "fps" in group.columns and not pd.isna(group["fps"].iloc[0]):
            fps = float(group["fps"].iloc[0])
        elif default_fps is not None:
            fps = float(default_fps)
        if fps is None:
            raise ValueError(f"FPS missing for trial {trial_id} and no default provided.")
        trials.append(_load_trial_from_df(group, fps=fps))
    LOGGER.info("Loaded %d trials from stacked CSV %s", len(trials), path)
    return trials


def _extract_time_series_columns(df: pd.DataFrame, time_cfg: Dict[str, object]) -> List[str]:
    if "columns" in time_cfg:
        columns = [str(c) for c in time_cfg["columns"]]
    elif "prefix" in time_cfg:
        prefix = str(time_cfg["prefix"])
        columns = [c for c in df.columns if c.startswith(prefix)]
        if not columns:
            raise ValueError(f"No columns starting with prefix '{prefix}' found for wide format.")
        # sort by numeric suffix when available
        def _key(name: str) -> float:
            suffix = name[len(prefix) :]
            try:
                return float(suffix)
            except ValueError:
                return float("inf")

        columns = sorted(columns, key=_key)
    else:
        raise ValueError("time_columns must specify either 'columns' or 'prefix'.")
    start_index = int(time_cfg.get("start_index", 0))
    if start_index < 0:
        raise ValueError("time_columns.start_index must be non-negative.")
    if start_index:
        columns = columns[start_index:]
    max_count = time_cfg.get("max_count")
    if max_count is not None:
        max_count_int = int(max_count)
        if max_count_int <= 0:
            raise ValueError("time_columns.max_count must be positive when provided.")
        columns = columns[:max_count_int]
    if not columns:
        raise ValueError("No time columns remain after applying start_index/max_count constraints.")
    LOGGER.debug("Identified %d time columns for wide format.", len(columns))
    return columns


def _load_wide_csv(
    path: Path,
    default_fps: Optional[float],
    config: Dict[str, object],
    read_cfg: Optional[Dict[str, object]],
) -> List[TrialTimeseries]:
    df = _read_csv(path, read_cfg)
    trial_col = str(config.get("trial_id_column", "trial_id"))
    fly_col = str(config.get("fly_id_column", "fly_id"))
    odor_on_col = config.get("odor_on_column")
    odor_off_col = config.get("odor_off_column")
    odor_on_value = config.get("odor_on_value")
    odor_off_value = config.get("odor_off_value")
    fps_col = config.get("fps_column")
    trial_template = config.get("trial_id_template")
    fly_template = config.get("fly_id_template")
    if trial_col not in df.columns:
        LOGGER.warning(
            "Trial identifier column '%s' not found; using row index as trial_id.", trial_col
        )
        df[trial_col] = [f"trial_{i}" for i in range(len(df))]
    if fly_col not in df.columns:
        LOGGER.warning("Fly identifier column '%s' not found; using 'unknown'.", fly_col)
        df[fly_col] = "unknown"
    if odor_on_col and odor_on_col not in df.columns and odor_on_value is None:
        raise ValueError(
            f"Wide CSV missing odor onset column '{odor_on_col}'."
            " Provide io.wide.odor_on_value in the config to use a constant value."
        )
    time_cfg = config.get("time_columns")
    if not isinstance(time_cfg, dict):
        raise ValueError("io.wide.time_columns must be provided for wide CSV loading.")
    time_columns = _extract_time_series_columns(df, time_cfg)
    trials: List[TrialTimeseries] = []
    for _, row in df.iterrows():
        context = {col: row[col] for col in df.columns}
        if trial_template:
            try:
                trial_id = trial_template.format(**context)
            except KeyError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"trial_id_template references missing column {exc.args[0]!r}."
                ) from exc
        else:
            trial_id = str(row[trial_col])
        if fly_template:
            try:
                fly_id = fly_template.format(**context)
            except KeyError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"fly_id_template references missing column {exc.args[0]!r}."
                ) from exc
        else:
            fly_id = str(row[fly_col])
        if odor_on_col and odor_on_col in df.columns and not pd.isna(row[odor_on_col]):
            odor_on_idx = int(row[odor_on_col])
        elif odor_on_value is not None:
            odor_on_idx = int(odor_on_value)
        else:
            raise ValueError(
                f"Odor onset missing for trial {trial_id}; specify io.wide.odor_on_column or odor_on_value."
            )
        if odor_off_col and odor_off_col in df.columns and not pd.isna(row[odor_off_col]):
            odor_off_idx = int(row[odor_off_col])
        elif odor_off_value is not None:
            odor_off_idx = int(odor_off_value)
        else:
            odor_off_idx = None
        if fps_col and fps_col in df.columns and not pd.isna(row[fps_col]):
            fps = float(row[fps_col])
        elif default_fps is not None:
            fps = float(default_fps)
        else:
            raise ValueError(f"FPS missing for trial {trial_id} and no default provided.")
        series = pd.to_numeric(row[time_columns], errors="coerce")
        values = series.to_numpy(dtype=float)
        valid_mask = ~np.isnan(values)
        if not valid_mask.any():
            raise ValueError(f"Trial {trial_id} has no numeric distance samples.")
        distance = values[valid_mask]
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
        trials.append(trial)
    LOGGER.info("Loaded %d trials from wide CSV %s", len(trials), path)
    return trials


def load_trials(path: str | Path, config: Optional[dict[str, object]] = None) -> List[TrialTimeseries]:
    """Load trials from a CSV file or directory."""

    config = config or {}
    default_fps = float(config.get("fps", 0.0)) if "fps" in config else None
    path = Path(path)
    io_cfg = config.get("io", {}) if isinstance(config, dict) else {}
    read_cfg = io_cfg.get("read_csv") if isinstance(io_cfg, dict) else None
    stacked_cfg = io_cfg.get("stacked") if isinstance(io_cfg, dict) else None
    wide_cfg = io_cfg.get("wide") if isinstance(io_cfg, dict) else None
    fmt = io_cfg.get("format") if isinstance(io_cfg, dict) else "auto"
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
        if fmt == "wide":
            if wide_cfg is None:
                raise ValueError("io.wide configuration required for wide CSV loading.")
            return _load_wide_csv(path, default_fps=default_fps, config=wide_cfg, read_cfg=read_cfg)
        if fmt == "stacked":
            return _load_stacked_csv(path, default_fps=default_fps, config=stacked_cfg, read_cfg=read_cfg)
        # auto-detect stacked first, then wide
        try:
            return _load_stacked_csv(path, default_fps=default_fps, config=stacked_cfg, read_cfg=read_cfg)
        except ValueError as err:
            LOGGER.debug("Stacked CSV parsing failed: %s", err)
            if wide_cfg is None:
                raise
            LOGGER.info("Falling back to wide CSV parsing based on io.wide configuration.")
            return _load_wide_csv(path, default_fps=default_fps, config=wide_cfg, read_cfg=read_cfg)
    raise FileNotFoundError(f"Could not locate data path: {path}")

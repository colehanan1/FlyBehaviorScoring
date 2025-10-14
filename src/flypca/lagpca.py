"""Lag-embedded PCA utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import IncrementalPCA, PCA

from .io import TrialTimeseries
from .preprocess import PreprocessConfig, SmoothingConfig, preprocess_trial

LOGGER = logging.getLogger(__name__)


@dataclass
class LagPCAResult:
    """Result of fitting lag-embedded PCA."""

    model: PCA | IncrementalPCA
    lag_samples: int
    fps: float
    pre_s: float
    post_s: float
    smoothing: SmoothingConfig
    explained_variance_ratio_: np.ndarray
    feature_names_: List[str]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        LOGGER.info("Saved LagPCAResult to %s", path)

    @staticmethod
    def load(path: str | Path) -> "LagPCAResult":
        result: LagPCAResult = joblib.load(path)
        return result


def hankel_embed(y: np.ndarray, lag: int) -> np.ndarray:
    """Return Hankel embedding using sliding windows."""

    if y.ndim != 1:
        raise ValueError("Input must be one-dimensional.")
    if lag <= 1:
        raise ValueError("lag must be greater than 1")
    if y.size < lag:
        raise ValueError("Trace shorter than lag length.")
    windows = sliding_window_view(y, window_shape=lag)
    return windows


def fit_pca(X: np.ndarray, n_components: int, incremental: bool = False) -> PCA | IncrementalPCA:
    """Fit a PCA or IncrementalPCA model."""

    if incremental:
        model: PCA | IncrementalPCA = IncrementalPCA(n_components=n_components, batch_size=1024)
        model.fit(X)
    else:
        model = PCA(n_components=n_components, random_state=0)
        model.fit(X)
    return model


def _extract_trials(
    trials: Sequence[TrialTimeseries],
    config: PreprocessConfig,
    lag_samples: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    windows: list[np.ndarray] = []
    times: list[np.ndarray] = []
    for trial in trials:
        time, zscored, _, _ = preprocess_trial(trial, config)
        if zscored.size <= lag_samples:
            LOGGER.warning("Skipping trial %s due to short length", trial.trial_id)
            continue
        embedded = hankel_embed(zscored, lag_samples)
        windows.append(embedded)
        center_indices = np.arange(lag_samples // 2, lag_samples // 2 + embedded.shape[0])
        center_indices = np.clip(center_indices, 0, time.size - 1)
        times.append(time[center_indices])
    if not windows:
        raise ValueError("No trials available after preprocessing.")
    return windows, times


def fit_lag_pca_for_trials(
    trials: Sequence[TrialTimeseries],
    config: dict,
    incremental: bool = False,
    model_path: str | Path | None = None,
) -> LagPCAResult:
    """Fit lag-embedded PCA across trials."""

    if not trials:
        raise ValueError("No trials provided for PCA fitting.")
    fps_values = {round(trial.fps, 6) for trial in trials}
    if len(fps_values) != 1:
        raise ValueError("All trials must share the same FPS for lag-PCA fitting.")
    fps = trials[0].fps
    smoothing_cfg = config.get("smoothing", {})
    smoothing = SmoothingConfig(
        enable=bool(smoothing_cfg.get("enable", True)),
        savgol_window_ms=int(smoothing_cfg.get("savgol_window_ms", 151)),
        savgol_poly=int(smoothing_cfg.get("savgol_poly", 3)),
        lowpass_hz=smoothing_cfg.get("lowpass_hz"),
    )
    preproc_config = PreprocessConfig(
        fps=fps,
        pre_s=float(config.get("pre_s", 2.0)),
        post_s=float(config.get("post_s", 2.0)),
        smoothing=smoothing,
    )
    lag_ms = float(config.get("lag_ms", 250))
    lag_samples = max(int(round(lag_ms / 1000 * fps)), 2)
    windows, _ = _extract_trials(trials, preproc_config, lag_samples)
    X = np.concatenate(windows, axis=0)
    LOGGER.debug("Fitting PCA on matrix shape %s", X.shape)
    n_components = int(config.get("n_components", 5))
    model = fit_pca(X, n_components=n_components, incremental=incremental)
    feature_names = [f"PC{i+1}" for i in range(model.n_components_)]
    result = LagPCAResult(
        model=model,
        lag_samples=lag_samples,
        fps=fps,
        pre_s=preproc_config.pre_s,
        post_s=preproc_config.post_s,
        smoothing=smoothing,
        explained_variance_ratio_=np.asarray(model.explained_variance_ratio_),
        feature_names_=feature_names,
    )
    if model_path is not None:
        result.save(model_path)
    return result


def project_trial(trial: TrialTimeseries, result: LagPCAResult) -> tuple[np.ndarray, np.ndarray]:
    """Project a trial into the LagPCA latent space."""

    preproc_config = PreprocessConfig(
        fps=result.fps,
        pre_s=result.pre_s,
        post_s=result.post_s,
        smoothing=result.smoothing,
    )
    time, zscored, _, _ = preprocess_trial(trial, preproc_config)
    if zscored.size <= result.lag_samples:
        raise ValueError("Trial shorter than lag window for projection.")
    embedded = hankel_embed(zscored, result.lag_samples)
    pcs = result.model.transform(embedded)
    center_indices = np.arange(result.lag_samples // 2, result.lag_samples // 2 + pcs.shape[0])
    center_indices = np.clip(center_indices, 0, time.size - 1)
    return time[center_indices], pcs

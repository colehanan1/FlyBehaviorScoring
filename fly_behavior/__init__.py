"""Shared utilities for fly behavior analysis and labeling."""

from .envelope import compute_envelope, compute_threshold, DEFAULT_SMOOTHING_FPS
from .data import choose_signal_column, extract_time_seconds, parse_fly_trial
from .segments import (
    SegmentDefinition,
    SEGMENT_DEFINITIONS,
    SEGMENT_MAP,
    ODOR_LATENCY_CHOICES,
    BASELINE_SECONDS,
    analysis_duration_seconds,
    compute_segment_windows,
)
from .metrics import compute_metrics

__all__ = [
    "compute_envelope",
    "compute_threshold",
    "DEFAULT_SMOOTHING_FPS",
    "choose_signal_column",
    "extract_time_seconds",
    "parse_fly_trial",
    "SegmentDefinition",
    "SEGMENT_DEFINITIONS",
    "SEGMENT_MAP",
    "ODOR_LATENCY_CHOICES",
    "BASELINE_SECONDS",
    "analysis_duration_seconds",
    "compute_segment_windows",
    "compute_metrics",
]

"""Segment definitions and utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

BASELINE_SECONDS: float = 30.0
ODOR_SECONDS: float = 30.0
POST_SECONDS: float = 30.0

ODOR_LATENCY_CHOICES: Dict[str, float] = {
    "opto": 2.0,
    "manual": 3.0,
}


@dataclass(frozen=True)
class SegmentDefinition:
    key: str
    label: str
    duration_seconds: float
    rateable: bool
    role: str


SEGMENT_DEFINITIONS: List[SegmentDefinition] = [
    SegmentDefinition(
        key="baseline",
        label="Baseline (first 30 s)",
        duration_seconds=BASELINE_SECONDS ,
        rateable=False,
        role="baseline",
    ),
    SegmentDefinition(
        key="odor",
        label="During odor",
        duration_seconds=ODOR_SECONDS,
        rateable=True,
        role="during",
    ),
    SegmentDefinition(
        key="post",
        label="30 s after odor",
        duration_seconds=POST_SECONDS,
        rateable=True,
        role="after",
    ),
]

SEGMENT_MAP: Dict[str, SegmentDefinition] = {seg.key: seg for seg in SEGMENT_DEFINITIONS}


def analysis_duration_seconds(latency_seconds: float) -> float:
    """Total seconds analysed for metrics including latency gap."""
    latency = max(latency_seconds, 0.0)
    return BASELINE_SECONDS + latency + ODOR_SECONDS + POST_SECONDS


def compute_segment_windows(
    fps: float,
    latency_seconds: float,
    max_available_frames: int,
) -> Dict[str, Tuple[int, int]]:
    """Return start/end frame indices (end-exclusive) for each segment."""
    fps_val = float(fps) if fps and fps > 0 else 0.0
    if fps_val <= 0:
        return {seg.key: (0, 0) for seg in SEGMENT_DEFINITIONS}

    latency = max(latency_seconds, 0.0)
    limit = max(0, max_available_frames)

    def clamp(frame: int) -> int:
        return max(0, min(limit, frame))

    baseline_end = clamp(int(round(BASELINE_SECONDS * fps_val)))
    odor_start = clamp(int(round((BASELINE_SECONDS + latency) * fps_val)))
    if odor_start < baseline_end:
        odor_start = baseline_end
    odor_end = clamp(odor_start + int(round((ODOR_SECONDS+ latency) * fps_val)))
    post_start = odor_end
    post_end = clamp(post_start + int(round(POST_SECONDS * fps_val)))

    return {
        "baseline": (0, baseline_end),
        "odor": (odor_start, odor_end),
        "post": (post_start, post_end),
    }


__all__ = [
    "SegmentDefinition",
    "SEGMENT_DEFINITIONS",
    "SEGMENT_MAP",
    "ODOR_LATENCY_CHOICES",
    "BASELINE_SECONDS",
    "analysis_duration_seconds",
    "compute_segment_windows",
]

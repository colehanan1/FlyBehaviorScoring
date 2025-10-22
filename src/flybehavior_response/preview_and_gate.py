"""Interactive preview and gating utilities for synthetic trials."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import math
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import LABEL_COLUMN, LABEL_INTENSITY_COLUMN


class PreviewDecision(Enum):
    KEEP = "keep"
    DROP = "drop"
    RELABEL_TO_ZERO = "relabel0"
    RELABEL_TO_ONE = "relabel1"


def _apply_decision(label: int, decision: PreviewDecision) -> int:
    if decision == PreviewDecision.RELABEL_TO_ZERO:
        return 0
    if decision == PreviewDecision.RELABEL_TO_ONE:
        return 1
    return label


def _compute_conflict(row: pd.Series) -> bool:
    label = int(row[LABEL_COLUMN])
    auc_during = float(row.get("AUC-During", 0.0))
    peak_value = float(row.get("Peak-Value", 0.0))
    ttpd = float(row.get("TimeToPeak-During", 0.0))
    if label == 1:
        return math.isclose(auc_during, 0.0, abs_tol=1e-6) and math.isclose(peak_value, 0.0, abs_tol=1e-6)
    return (auc_during > 0) or (ttpd > 0)


def _auto_decision(
    *,
    label: int,
    conflict: bool,
    probability: float | None,
    margin: float,
) -> PreviewDecision:
    if conflict:
        return PreviewDecision.DROP
    if probability is None or margin <= 0:
        return PreviewDecision.KEEP
    target = float(label)
    delta = abs(target - probability)
    if delta < margin:
        return PreviewDecision.KEEP
    if label == 1 and probability < 0.5:
        return PreviewDecision.RELABEL_TO_ZERO
    if label == 0 and probability > 0.5:
        return PreviewDecision.RELABEL_TO_ONE
    return PreviewDecision.DROP


def _plot_trial(ax, row: pd.Series, trace_columns: Sequence[str], probability: float | None) -> None:
    trace = row[trace_columns].to_numpy(dtype=float)
    frames = np.arange(len(trace))
    ax.plot(frames, trace, color="tab:blue", linewidth=1.0)
    threshold = float(row.get("threshold_trial", np.nan))
    if np.isfinite(threshold):
        ax.axhline(threshold, color="tab:orange", linestyle="--", linewidth=1.0)
    title = [
        row.get("synthetic_trial_id", "synthetic"),
        f"label={int(row[LABEL_COLUMN])}",
        row.get("aug_op", "op"),
        f"parents={row.get('parent_trial_ids', '')}",
        f"seed={row.get('seed', '')}",
        f"AUCd={row.get('AUC-During', float('nan')):.1f}",
        f"Peak={row.get('Peak-Value', float('nan')):.1f}",
        f"TTPD={row.get('TimeToPeak-During', float('nan')):.2f}s",
    ]
    if probability is not None:
        title.append(f"p(reaction)={probability:.2f}")
    if bool(row.get("conflict_flag", False)):
        title.append("CONFLICT")
    ax.set_title(" | ".join(title), fontsize=8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("dir_val")


def _save_preview_grid(
    *,
    frame: pd.DataFrame,
    trace_columns: Sequence[str],
    probabilities: List[float | None],
    output_dir: Path,
) -> Path | None:
    if frame.empty:
        return None
    cols = min(4, len(frame))
    rows = math.ceil(len(frame) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), squeeze=False)
    for ax in axes.ravel()[len(frame) :]:
        ax.axis("off")
    for (idx, row), ax, prob in zip(frame.iterrows(), axes.ravel(), probabilities):
        _plot_trial(ax, row, trace_columns, prob)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "synthetic_preview.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _interactive_review(
    *,
    row: pd.Series,
    trace_columns: Sequence[str],
    probability: float | None,
) -> PreviewDecision:
    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_trial(ax, row, trace_columns, probability)
    decision: PreviewDecision | None = None

    def _on_key(event):
        nonlocal decision
        mapping = {
            "k": PreviewDecision.KEEP,
            "d": PreviewDecision.DROP,
            "0": PreviewDecision.RELABEL_TO_ZERO,
            "1": PreviewDecision.RELABEL_TO_ONE,
            "n": PreviewDecision.KEEP,
        }
        if event.key == "q":
            decision = PreviewDecision.DROP
            plt.close(fig)
            return
        if event.key in mapping:
            decision = mapping[event.key]
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", _on_key)
    plt.show(block=True)
    if decision is None:
        decision = PreviewDecision.KEEP
    return decision


@dataclass
class PreviewPayload:
    decisions: List[PreviewDecision]
    final_labels: List[int]
    predictions: List[float]
    conflicts: List[bool]
    preview_image: Path | None
    preview_indices: List[int]


def preview_and_gate(
    frame: pd.DataFrame,
    *,
    trace_columns: Sequence[str],
    config,
    logger: logging.Logger,
    pipeline=None,
) -> PreviewPayload:
    frame = frame.copy()
    decisions: List[PreviewDecision] = [PreviewDecision.KEEP] * len(frame)
    final_labels = frame[LABEL_COLUMN].astype(int).tolist()
    conflicts = [_compute_conflict(row) for _, row in frame.iterrows()]
    frame.loc[:, "conflict_flag"] = conflicts
    predictions: List[float | None] = [None] * len(frame)

    scoring_pipeline = pipeline
    feature_frame = frame.drop(columns=[LABEL_COLUMN], errors="ignore")
    feature_frame = feature_frame.drop(columns=[LABEL_INTENSITY_COLUMN], errors="ignore")
    if scoring_pipeline is not None and hasattr(scoring_pipeline, "predict_proba"):
        try:
            proba = scoring_pipeline.predict_proba(feature_frame)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to score synthetics for preview: %s", exc)
        else:
            if proba.ndim == 2 and proba.shape[1] >= 2:
                for idx in range(len(frame)):
                    predictions[idx] = float(proba[idx, 1])

    for idx, (label, conflict, prediction) in enumerate(zip(final_labels, conflicts, predictions)):
        decision = _auto_decision(
            label=label,
            conflict=conflict,
            probability=prediction,
            margin=float(config.auto_filter_threshold),
        )
        decisions[idx] = decision
        final_labels[idx] = _apply_decision(label, decision)

    rng = np.random.default_rng(config.seed)
    preview_indices: List[int] = []
    total_trials = len(frame)
    preview_target = int(config.preview_synthetics)
    if preview_target < 0:
        preview_indices = list(range(total_trials))
    elif preview_target > 0:
        preview_target = min(preview_target, total_trials)
        if preview_target == total_trials:
            preview_indices = list(range(total_trials))
        else:
            taken: set[int] = set()
            # ensure each class is represented when available
            for label_value in sorted(frame[LABEL_COLUMN].unique()):
                class_indices = [idx for idx, value in enumerate(final_labels) if value == label_value]
                if class_indices and len(taken) < preview_target:
                    taken.add(class_indices[0])
            remaining_pool = [idx for idx in range(total_trials) if idx not in taken]
            need = preview_target - len(taken)
            if need > 0 and remaining_pool:
                sampled = rng.choice(remaining_pool, size=min(need, len(remaining_pool)), replace=False)
                taken.update(int(idx) for idx in np.atleast_1d(sampled))
            preview_indices = sorted(taken)

    logger.info(
        "Previewing %d of %d synthetic trials (target=%d).",
        len(preview_indices),
        total_trials,
        preview_target,
    )

    if preview_indices and preview_target != 0:
        for idx in preview_indices:
            row = frame.iloc[idx]
            manual_decision = _interactive_review(
                row=row,
                trace_columns=trace_columns,
                probability=predictions[idx],
            )
            decisions[idx] = manual_decision
            final_labels[idx] = _apply_decision(int(row[LABEL_COLUMN]), manual_decision)

    preview_frame = frame.iloc[preview_indices] if preview_indices else frame.iloc[0:0]
    preview_probs = [predictions[idx] for idx in preview_indices]
    if preview_frame.empty and len(frame) > 0 and config.preview_synthetics == 0:
        sample_count = min(8, len(frame))
        sampled = list(rng.choice(len(frame), size=sample_count, replace=False))
        preview_frame = frame.iloc[sampled]
        preview_probs = [predictions[idx] for idx in sampled]
    preview_image = _save_preview_grid(
        frame=preview_frame,
        trace_columns=trace_columns,
        probabilities=preview_probs,
        output_dir=config.save_synthetics_dir,
    )

    logger.info(
        "Synthetic gating summary | keep=%d drop=%d relabel0=%d relabel1=%d",
        decisions.count(PreviewDecision.KEEP),
        decisions.count(PreviewDecision.DROP),
        decisions.count(PreviewDecision.RELABEL_TO_ZERO),
        decisions.count(PreviewDecision.RELABEL_TO_ONE),
    )

    for idx, decision in enumerate(decisions):
        if decision == PreviewDecision.RELABEL_TO_ZERO:
            predictions[idx] = predictions[idx] if predictions[idx] is not None else float("nan")
        elif decision == PreviewDecision.RELABEL_TO_ONE:
            predictions[idx] = predictions[idx] if predictions[idx] is not None else float("nan")
        else:
            predictions[idx] = predictions[idx] if predictions[idx] is not None else float("nan")

    cleaned_predictions = [float(value) if value is not None else float("nan") for value in predictions]

    return PreviewPayload(
        decisions=decisions,
        final_labels=final_labels,
        predictions=cleaned_predictions,
        conflicts=conflicts,
        preview_image=preview_image,
        preview_indices=preview_indices,
    )


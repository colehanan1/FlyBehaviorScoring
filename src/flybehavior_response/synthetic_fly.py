"""Synthetic fly generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .augment import OPERATIONS, mixup_same_class
from .io import LABEL_COLUMN, LABEL_INTENSITY_COLUMN
from .metrics import attach_metrics, detect_fly_column, discover_trace_columns
from .preview_and_gate import PreviewDecision, preview_and_gate


@dataclass(slots=True)
class SyntheticConfig:
    use_synthetics: bool = False
    synthetic_fly_ratio: float = 0.25
    synthetic_ops: Sequence[str] = field(
        default_factory=lambda: ("jitter", "scale", "time_shift", "crop_resize", "mixup_same_class")
    )
    mixup_alpha: float = 0.2
    preview_synthetics: int = 12
    preview_score_checkpoint: Path | None = None
    auto_filter_threshold: float = 0.0
    save_synthetics_dir: Path = Path("./synthetics")
    missing_ttpd_policy: str = "zero"
    seed: int = 42


@dataclass(slots=True)
class SyntheticTrial:
    synthetic_fly_id: str
    synthetic_trial_id: str
    parent_fly_id: str
    parent_trial_ids: Tuple[str, ...]
    trace: np.ndarray
    label: int
    intensity: float
    trial_label: str
    aug_op: str
    seed: int
    metadata: Dict[str, object]


@dataclass(slots=True)
class SyntheticResult:
    dataframe: pd.DataFrame
    manifest: pd.DataFrame
    kept_indices: List[int]
    preview_image: Path | None


class SyntheticFlyGenerator:
    """Generates synthetic flies for a training split."""

    def __init__(
        self,
        *,
        config: SyntheticConfig,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.logger = logger
        self.rng = np.random.default_rng(config.seed)

    def generate(
        self,
        frame: pd.DataFrame,
        *,
        trace_columns: Sequence[str] | None = None,
        fly_column: str | None = None,
        preview_pipeline=None,
    ) -> SyntheticResult:
        if not self.config.use_synthetics:
            empty = pd.DataFrame(columns=list(frame.columns))
            return SyntheticResult(dataframe=empty, manifest=pd.DataFrame(), kept_indices=[], preview_image=None)

        if trace_columns is None:
            trace_columns = discover_trace_columns(frame)
        if fly_column is None:
            fly_column = detect_fly_column(frame)

        trace_columns = list(trace_columns)
        available_ops = [op for op in self.config.synthetic_ops if op in OPERATIONS]
        if not available_ops:
            raise ValueError("No valid synthetic operations provided.")

        group_counts = frame[fly_column].value_counts().to_dict()
        if not group_counts:
            raise ValueError("No fly groups available for synthetic generation.")

        self.logger.info("Proposing synthetic flies using ops: %s", available_ops)

        synthetic_trials: List[SyntheticTrial] = []

        for fly_id, fly_df in frame.groupby(fly_column):
            synth_count = max(1, int(math.ceil(self.config.synthetic_fly_ratio)))
            for clone_idx in range(synth_count):
                synthetic_fly_id = f"syn_{fly_id}_{self.config.seed}_{clone_idx}"
                synthetic_trials.extend(
                    self._generate_for_fly(
                        fly_df,
                        trace_columns=trace_columns,
                        synthetic_fly_id=synthetic_fly_id,
                        preview_ops=available_ops,
                        fly_column=fly_column,
                    )
                )

        synthetic_df = self._assemble_dataframe(
            frame=frame,
            synthetic_trials=synthetic_trials,
            trace_columns=trace_columns,
        )

        previewed = preview_and_gate(
            synthetic_df,
            trace_columns=trace_columns,
            config=self.config,
            logger=self.logger,
            pipeline=preview_pipeline,
        )

        preview_flags = np.zeros(len(synthetic_df), dtype=bool)
        if previewed.preview_indices:
            preview_flags[np.asarray(previewed.preview_indices, dtype=int)] = True

        self.logger.info(
            "Synthetic trials generated=%d previewed=%d",
            len(synthetic_df),
            int(np.count_nonzero(preview_flags)),
        )

        synthetic_df["decision"] = [decision.value for decision in previewed.decisions]
        synthetic_df["final_label"] = previewed.final_labels
        synthetic_df["pred_prob"] = previewed.predictions
        synthetic_df["conflict_flag"] = previewed.conflicts
        synthetic_df[LABEL_COLUMN] = previewed.final_labels
        synthetic_df["was_previewed"] = preview_flags
        if LABEL_INTENSITY_COLUMN in synthetic_df.columns:
            intensities = synthetic_df[LABEL_INTENSITY_COLUMN].astype(float)
            intensities.loc[synthetic_df[LABEL_COLUMN] == 0] = 0.0
            intensities.loc[(synthetic_df[LABEL_COLUMN] == 1) & (intensities <= 0)] = 1.0
            synthetic_df[LABEL_INTENSITY_COLUMN] = intensities

        kept_mask = [decision in {PreviewDecision.KEEP, PreviewDecision.RELABEL_TO_ONE, PreviewDecision.RELABEL_TO_ZERO} for decision in previewed.decisions]
        kept_indices = [idx for idx, keep in enumerate(kept_mask) if keep]

        manifest = synthetic_df[[
            "synthetic_fly_id",
            "synthetic_trial_id",
            "parent_fly_id",
            "parent_trial_ids",
            "aug_op",
            "seed",
            "conflict_flag",
            "pred_prob",
            "decision",
            "final_label",
            "was_previewed",
        ]].copy()

        manifest.insert(0, "is_synthetic", 1)

        return SyntheticResult(
            dataframe=synthetic_df,
            manifest=manifest,
            kept_indices=kept_indices,
            preview_image=previewed.preview_image,
        )

    def _generate_for_fly(
        self,
        fly_df: pd.DataFrame,
        *,
        trace_columns: Sequence[str],
        synthetic_fly_id: str,
        preview_ops: Sequence[str],
        fly_column: str,
    ) -> List[SyntheticTrial]:
        trials: List[SyntheticTrial] = []
        traces = fly_df[trace_columns].to_numpy(dtype=float)
        trial_labels = fly_df[LABEL_COLUMN].astype(int).to_numpy()
        trial_names = fly_df.get("trial_label", fly_df.index.astype(str)).astype(str).to_numpy()
        metadata_columns = [col for col in fly_df.columns if col not in trace_columns]

        for trial_idx, row in enumerate(fly_df.itertuples(index=False)):
            label = int(trial_labels[trial_idx])
            op = self.rng.choice(preview_ops)
            trace = traces[trial_idx]
            parent_ids = (str(trial_names[trial_idx]),)
            if op == "mixup_same_class":
                same_label_indices = np.where(trial_labels == label)[0]
                if len(same_label_indices) < 2:
                    op = self.rng.choice([candidate for candidate in preview_ops if candidate != "mixup_same_class"])
                else:
                    other_idx = int(self.rng.choice(same_label_indices[same_label_indices != trial_idx]))
                    mixed = mixup_same_class(
                        traces[trial_idx],
                        traces[other_idx],
                        rng=self.rng,
                        alpha=self.config.mixup_alpha,
                    )
                    trace = mixed[:, 0]
                    parent_ids = (str(trial_names[trial_idx]), str(trial_names[other_idx]))
            if op != "mixup_same_class":
                trace = OPERATIONS[op](trace, rng=self.rng).squeeze()

            synthetic_trial_id = f"{synthetic_fly_id}_trial{len(trials)}"
            trial_label_value = getattr(row, "trial_label", synthetic_trial_id)
            parent_metadata = {
                column: getattr(row, column)
                for column in metadata_columns
                if hasattr(row, column)
            }
            parent_fly_identifier = parent_metadata.get(fly_column, getattr(row, fly_column, "unknown"))
            synthetic_intensity = 1.0 if label else 0.0
            trials.append(
                SyntheticTrial(
                    synthetic_fly_id=synthetic_fly_id,
                    synthetic_trial_id=synthetic_trial_id,
                    parent_fly_id=str(parent_fly_identifier),
                    parent_trial_ids=parent_ids,
                    trace=trace.astype(float),
                    label=label,
                    intensity=synthetic_intensity,
                    trial_label=str(trial_label_value),
                    aug_op=op,
                    seed=self.config.seed,
                    metadata=parent_metadata,
                )
            )
        return trials

    def _assemble_dataframe(
        self,
        *,
        frame: pd.DataFrame,
        synthetic_trials: Sequence[SyntheticTrial],
        trace_columns: Sequence[str],
    ) -> pd.DataFrame:
        records: List[Dict[str, object]] = []
        metadata_columns = [col for col in frame.columns if col not in trace_columns]

        for trial in synthetic_trials:
            row = {col: trial.metadata.get(col) for col in metadata_columns}
            row.update({col: float("nan") for col in [
                "AUC-Before",
                "AUC-During",
                "AUC-After",
                "TimeToPeak-During",
                "Peak-Value",
                "mean_before_fly",
                "std_before_trial",
                "threshold_trial",
            ]})
            row.update({col: value for col, value in zip(trace_columns, trial.trace)})
            row["trial_label"] = trial.trial_label
            row["is_synthetic"] = 1
            row["synthetic_fly_id"] = trial.synthetic_fly_id
            row["synthetic_trial_id"] = trial.synthetic_trial_id
            row["parent_fly_id"] = trial.parent_fly_id
            row["parent_trial_ids"] = ",".join(trial.parent_trial_ids)
            row["aug_op"] = trial.aug_op
            row["seed"] = trial.seed
            row[LABEL_COLUMN] = trial.label
            row[LABEL_INTENSITY_COLUMN] = float(trial.intensity if trial.label else 0.0)
            row["conflict_flag"] = False
            row["pred_prob"] = float("nan")
            row["decision"] = "keep"
            row["final_label"] = trial.label
            records.append(row)

        synthetic_df = pd.DataFrame(records)
        synthetic_df = attach_metrics(
            synthetic_df,
            trace_columns=trace_columns,
            fly_column="synthetic_fly_id",
            missing_ttpd_policy=self.config.missing_ttpd_policy,
        )
        return synthetic_df


def save_synthetic_artifacts(
    result: SyntheticResult,
    *,
    config: SyntheticConfig,
) -> Tuple[Path, Path]:
    directory = config.save_synthetics_dir
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "synthetics.csv"
    manifest_path = directory / "synthetics_manifest.csv"
    result.dataframe.to_csv(csv_path, index=False)
    result.manifest.to_csv(manifest_path, index=False)
    payload = {
        "config": {
            "synthetic_fly_ratio": config.synthetic_fly_ratio,
            "synthetic_ops": list(config.synthetic_ops),
            "mixup_alpha": config.mixup_alpha,
            "preview_synthetics": config.preview_synthetics,
            "auto_filter_threshold": config.auto_filter_threshold,
            "missing_ttpd_policy": config.missing_ttpd_policy,
            "seed": config.seed,
        }
    }
    if result.preview_image is not None:
        payload["preview_image"] = str(result.preview_image)
    with (directory / "synthetics_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    return csv_path, manifest_path


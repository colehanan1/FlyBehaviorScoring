"""Synthetic data generation and preview gating utilities."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load as joblib_load

from . import augment

LOGGER = logging.getLogger(__name__)

DEFAULT_SYNTHETIC_OPS: tuple[str, ...] = (
    "jitter",
    "scale",
    "time_shift",
    "crop_resize",
    "mixup_same_class",
)


@dataclass(slots=True)
class SyntheticConfig:
    """Configuration controlling synthetic generation and preview."""

    synthetic_ratio: float = 0.5
    synthetic_ops: Sequence[str] = DEFAULT_SYNTHETIC_OPS
    mixup_alpha: float = 0.2
    preview_synthetics: int = 12
    preview_score_checkpoint: Path | None = None
    auto_filter_threshold: float = 0.0
    save_synthetics_dir: Path = Path("./synthetics")
    seed: int = 42


def _resolve_ops(names: Sequence[str]) -> list[str]:
    resolved = []
    available = set(DEFAULT_SYNTHETIC_OPS)
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if name not in available:
            raise ValueError(f"Unsupported synthetic op: {name}")
        resolved.append(name)
    if not resolved:
        raise ValueError("At least one synthetic operation must be provided.")
    return resolved


def _op_callable(name: str):
    return getattr(augment, name)


def make_synthetics(
    X_train: np.ndarray,
    y_train: Sequence[int],
    ids_train: Sequence[str],
    class_names: Sequence[int],
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, ...]], list[str], np.ndarray]:
    """Generate synthetic samples using augmentation primitives."""

    if config.synthetic_ratio <= 0.0:
        LOGGER.info("Synthetic ratio %.3f <= 0; skipping generation.", config.synthetic_ratio)
        return (
            np.empty((0,) + X_train.shape[1:], dtype=np.float32),
            np.empty((0,), dtype=int),
            [],
            [],
            np.empty((0,), dtype=np.int64),
        )

    ops = _resolve_ops(config.synthetic_ops)
    rng = np.random.default_rng(config.seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_arr = np.asarray(y_train, dtype=int)
    id_list = list(ids_train)

    if X_train.shape[0] != y_arr.shape[0] or len(id_list) != y_arr.shape[0]:
        raise ValueError("Training arrays and identifiers must have matching lengths.")

    n_real = X_train.shape[0]
    target_total = int(round(n_real * config.synthetic_ratio))
    n_classes = len(class_names)
    if target_total == 0:
        LOGGER.info("Synthetic ratio target rounded to zero; skipping generation.")
        return (
            np.empty((0,) + X_train.shape[1:], dtype=np.float32),
            np.empty((0,), dtype=int),
            [],
            [],
            np.empty((0,), dtype=np.int64),
        )

    per_class_counts: dict[int, int] = {int(label): 0 for label in class_names}
    base = target_total // max(1, n_classes)
    remainder = target_total % max(1, n_classes)
    for label in per_class_counts:
        per_class_counts[label] = base
    if remainder:
        sorted_classes = sorted(class_names, key=lambda c: -np.sum(y_arr == c))
        for label in sorted_classes[:remainder]:
            per_class_counts[int(label)] += 1

    class_to_indices: dict[int, np.ndarray] = {}
    for label in per_class_counts:
        mask = np.where(y_arr == label)[0]
        if mask.size == 0:
            LOGGER.warning("No training samples for class %s; skipping synthetic generation.", label)
            per_class_counts[label] = 0
        class_to_indices[label] = mask

    samples: list[np.ndarray] = []
    labels: list[int] = []
    parents: list[tuple[str, ...]] = []
    ops_used: list[str] = []
    seeds: list[int] = []

    for label, target in per_class_counts.items():
        if target <= 0:
            continue
        indices = class_to_indices[label]
        if indices.size == 0:
            continue
        for _ in range(target):
            chosen_op = None
            for _attempt in range(len(ops) * 2):
                candidate = rng.choice(ops)
                if candidate == "mixup_same_class" and indices.size < 2:
                    continue
                chosen_op = candidate
                break
            if chosen_op is None:
                LOGGER.warning(
                    "Unable to select suitable op for class %s due to insufficient samples; falling back to jitter.",
                    label,
                )
                chosen_op = "jitter"
            op_seed = int(rng.integers(0, 2**32 - 1))
            op_rng = np.random.default_rng(op_seed)
            parent_idx_a = int(rng.choice(indices))
            parent_ids: list[str] = [id_list[parent_idx_a]]
            x_a = X_train[parent_idx_a]
            if chosen_op == "mixup_same_class":
                parent_idx_b = int(rng.choice(indices))
                x_b = X_train[parent_idx_b]
                parent_ids.append(id_list[parent_idx_b])
                augmented = augment.mixup_same_class(x_a, x_b, alpha=config.mixup_alpha, rng=op_rng)
            else:
                op_fn = _op_callable(chosen_op)
                augmented = op_fn(x_a, rng=op_rng)
            samples.append(augmented.astype(np.float32))
            labels.append(int(label))
            parents.append(tuple(parent_ids))
            ops_used.append(chosen_op)
            seeds.append(op_seed)

    if not samples:
        LOGGER.info("No synthetic samples generated after applying constraints.")
        return (
            np.empty((0,) + X_train.shape[1:], dtype=np.float32),
            np.empty((0,), dtype=int),
            [],
            [],
            np.empty((0,), dtype=np.int64),
        )

    X_syn = np.stack(samples, axis=0).astype(np.float32)
    y_syn = np.asarray(labels, dtype=int)
    seeds_arr = np.asarray(seeds, dtype=np.int64)
    ratio_actual = len(samples) / float(n_real)
    LOGGER.info(
        "Generated %d synthetic samples (ratio=%.3f target=%.3f) using ops %s",
        len(samples),
        ratio_actual,
        config.synthetic_ratio,
        ",".join(ops),
    )
    for label in sorted(per_class_counts):
        LOGGER.info(
            "Class %s -> %d synthetics (real=%d)",
            label,
            sum(y_syn == label),
            int(np.sum(y_arr == label)),
        )
    return X_syn, y_syn, parents, ops_used, seeds_arr


def _load_model(path: Path):
    try:
        return joblib_load(path)
    except Exception:  # pragma: no cover - fallback path
        with path.open("rb") as handle:
            return pickle.load(handle)


def score_synthetics(model_path: Path | None, X_syn: pd.DataFrame) -> np.ndarray | None:
    """Score synthetic samples with a trained model, returning reaction probabilities."""

    if model_path is None:
        LOGGER.info("No checkpoint provided for scoring synthetics.")
        return None
    resolved = Path(model_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    LOGGER.info("Scoring synthetics with checkpoint %s", resolved)
    model = _load_model(resolved)
    if not hasattr(model, "predict_proba"):
        LOGGER.warning("Loaded model does not expose predict_proba; skipping scoring.")
        return None
    proba = model.predict_proba(X_syn)
    if isinstance(proba, np.ndarray):
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(float)
        if proba.ndim == 1:
            return proba.astype(float)
    LOGGER.warning("Unexpected probability output shape %s; skipping scoring.", None if proba is None else proba.shape)
    return None


def _auto_filter_decisions(
    probs: np.ndarray | None,
    labels: Sequence[int],
    threshold: float,
) -> dict[int, str]:
    actions: dict[int, str] = {}
    if probs is None or threshold <= 0:
        return actions
    margin_pos = 0.5 - threshold
    margin_neg = 0.5 + threshold
    for idx, (label, prob) in enumerate(zip(labels, probs)):
        if label == 1 and prob < margin_pos:
            actions[idx] = f"AUTO DROP (p={prob:.3f} < {margin_pos:.3f})"
        elif label == 0 and prob > margin_neg:
            actions[idx] = f"AUTO DROP (p={prob:.3f} > {margin_neg:.3f})"
    return actions


def _is_interactive_backend() -> bool:
    backend = matplotlib.get_backend().lower()
    return backend not in {"agg", "pdf", "template"}


def _format_title(
    syn_id: str,
    label: int,
    op: str,
    parents: Sequence[str],
    seed: int,
    prob: float | None,
    decision: str,
    auto_note: str | None,
) -> str:
    prob_text = f" | p(reaction)={prob:.2f}" if prob is not None else ""
    auto_text = f" ({auto_note})" if auto_note else ""
    parent_text = ",".join(parents)
    return (
        f"{syn_id} | label={label} | op={op} | parents=[{parent_text}] | seed={seed}{prob_text}"
        f" | decision={decision.upper()}{auto_text}"
    )


def _save_preview_grid(
    save_dir: Path,
    seed: int,
    X_syn: np.ndarray,
    y_syn: Sequence[int],
    syn_ids: Sequence[str],
    ops: Sequence[str],
    parents: Sequence[tuple[str, ...]],
    seeds: Sequence[int],
    probs: np.ndarray | None,
    decisions: Sequence[str],
    class_names: Sequence[int],
    max_per_class: int,
) -> Path:
    per_class: dict[int, List[int]] = {int(cls): [] for cls in class_names}
    for idx, label in enumerate(y_syn):
        per_class.setdefault(int(label), []).append(idx)
    max_cols = 0
    for cls in class_names:
        indices = per_class.get(int(cls), [])
        if max_per_class > 0:
            indices = indices[:max_per_class]
        max_cols = max(max_cols, len(indices))
    if max_cols == 0:
        return save_dir / f"preview_grid_seed{seed}.png"
    fig, axes = plt.subplots(len(class_names), max_cols, figsize=(4 * max_cols, 3 * len(class_names)))
    if not isinstance(axes, np.ndarray):  # pragma: no cover - matplotlib API quirk
        axes = np.asarray([[axes]])
    axes = np.atleast_2d(axes)
    for row, cls in enumerate(class_names):
        indices = per_class.get(int(cls), [])
        if max_per_class > 0:
            indices = indices[:max_per_class]
        for col in range(max_cols):
            ax = axes[row, col] if axes.ndim == 2 else axes[row]
            if col >= len(indices):
                ax.axis("off")
                continue
            idx = indices[col]
            series = X_syn[idx]
            time = np.arange(series.shape[0])
            ax.plot(time, series)
            prob = probs[idx] if probs is not None else None
            title = _format_title(
                syn_ids[idx],
                int(y_syn[idx]),
                ops[idx],
                parents[idx],
                int(seeds[idx]),
                prob,
                decisions[idx],
                None,
            )
            ax.set_title(title, fontsize=9)
    for ax in axes.flat:
        if ax.has_data():
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
    fig.tight_layout()
    path = save_dir / f"preview_grid_seed{seed}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved synthetic preview grid to %s", path.resolve())
    return path


def preview_and_gate(
    *,
    syn_ids: Sequence[str],
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    parents: Sequence[tuple[str, ...]],
    ops: Sequence[str],
    seeds: Sequence[int],
    class_names: Sequence[int],
    config: SyntheticConfig,
    probs: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Preview synthetics, collect user decisions, and persist provenance."""

    if len(X_syn) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int), []

    save_dir = Path(config.save_synthetics_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    decisions = ["keep"] * len(X_syn)
    final_labels = np.asarray(y_syn, dtype=int).copy()
    auto_flags = _auto_filter_decisions(probs, final_labels, config.auto_filter_threshold)
    for idx, note in auto_flags.items():
        decisions[idx] = "drop"
        LOGGER.info("Auto decision for %s -> %s", syn_ids[idx], note)

    preview_indices: list[int] = []
    per_class = {int(cls): [] for cls in class_names}
    for idx, label in enumerate(final_labels):
        per_class.setdefault(int(label), []).append(idx)
    for cls in class_names:
        candidates = per_class.get(int(cls), [])
        if not candidates:
            continue
        count = min(config.preview_synthetics, len(candidates)) if config.preview_synthetics > 0 else min(4, len(candidates))
        preview_indices.extend(candidates[:count])

    interactive = config.preview_synthetics > 0 and _is_interactive_backend()
    manifest_decisions: list[str] = ["pending"] * len(X_syn)

    if interactive and preview_indices:
        LOGGER.info(
            "Launching interactive preview for %d synthetic samples (backend=%s)",
            len(preview_indices),
            matplotlib.get_backend(),
        )
        print("Synthetic preview controls: [k]eep, [d]rop, [0]/[1] relabel, [n]ext, [q]uit", flush=True)
        for idx in preview_indices:
            decision = decisions[idx]
            auto_note = auto_flags.get(idx)
            fig, ax = plt.subplots(figsize=(8, 4))
            series = X_syn[idx]
            time = np.arange(series.shape[0])
            ax.plot(time, series)
            prob = probs[idx] if probs is not None else None
            title = _format_title(
                syn_ids[idx],
                int(final_labels[idx]),
                ops[idx],
                parents[idx],
                seeds[idx],
                prob,
                decision,
                auto_note,
            )
            ax.set_title(title)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            done = False

            class AbortPreview(RuntimeError):
                """Signal to abort preview."""

            def _on_key(event):
                nonlocal decision, done
                if event.key == "k":
                    decision = "keep"
                    final_labels[idx] = int(y_syn[idx])
                    done = True
                    plt.close(fig)
                elif event.key == "d":
                    decision = "drop"
                    final_labels[idx] = int(y_syn[idx])
                    done = True
                    plt.close(fig)
                elif event.key == "0":
                    decision = "relabel0"
                    final_labels[idx] = 0
                    done = True
                    plt.close(fig)
                elif event.key == "1":
                    decision = "relabel1"
                    final_labels[idx] = 1
                    done = True
                    plt.close(fig)
                elif event.key == "n":
                    done = True
                    plt.close(fig)
                elif event.key == "q":  # pragma: no cover - manual interaction
                    plt.close(fig)
                    raise AbortPreview("Synthetic preview aborted by user.")

            cid = fig.canvas.mpl_connect("key_press_event", _on_key)
            try:
                while not done:
                    plt.show()
            except AbortPreview as exc:  # pragma: no cover - manual interaction
                raise exc
            finally:
                fig.canvas.mpl_disconnect(cid)
            decisions[idx] = decision
    elif config.preview_synthetics > 0 and not _is_interactive_backend():
        LOGGER.warning(
            "Matplotlib backend %s is non-interactive; skipping preview UI but decisions can be adjusted via manifest.",
            matplotlib.get_backend(),
        )

    for idx in range(len(X_syn)):
        decision = decisions[idx]
        if decision == "keep":
            manifest_decisions[idx] = "keep"
        elif decision == "drop":
            manifest_decisions[idx] = "drop"
        elif decision == "relabel0":
            manifest_decisions[idx] = "relabel0"
        elif decision == "relabel1":
            manifest_decisions[idx] = "relabel1"
        else:
            manifest_decisions[idx] = decision

    mask_keep = np.array([dec != "drop" for dec in decisions], dtype=bool)
    manifest_records = []
    for idx, syn_id in enumerate(syn_ids):
        prob = probs[idx] if probs is not None else None
        entry = {
            "synthetic_id": syn_id,
            "parent_ids": ";".join(parents[idx]),
            "op": ops[idx],
            "seed": int(seeds[idx]),
            "orig_label": int(y_syn[idx]),
            "predicted_prob": None if prob is None or np.isnan(prob) else float(prob),
            "decision": manifest_decisions[idx],
            "final_label": int(final_labels[idx]),
        }
        manifest_records.append(entry)

    manifest_df = pd.DataFrame(manifest_records)
    csv_path = save_dir / "synthetics_manifest.csv"
    json_path = save_dir / "synthetics_manifest.json"
    manifest_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(manifest_records, indent=2))
    LOGGER.info("Saved synthetic manifest to %s", csv_path.resolve())
    LOGGER.info("Saved synthetic manifest JSON to %s", json_path.resolve())

    _save_preview_grid(
        save_dir,
        config.seed,
        X_syn,
        final_labels,
        syn_ids,
        ops,
        parents,
        seeds,
        probs,
        decisions,
        class_names,
        config.preview_synthetics if config.preview_synthetics > 0 else 4,
    )

    return mask_keep, final_labels, manifest_decisions


__all__ = [
    "SyntheticConfig",
    "make_synthetics",
    "score_synthetics",
    "preview_and_gate",
]


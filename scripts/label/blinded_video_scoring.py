#!/usr/bin/env python3
"""Blinded fly behavior scoring GUI — video + trace side-by-side.

Displays randomised, blinded testing trials with the video (black-box
overlay hiding the odor label) on the left and the envelope trace on the
right.  The user scores each trial 0-5 and optionally adds a comment.
Results are saved to CSV with resume capability.

Usage:
    python scripts/label/blinded_video_scoring.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Paths & constants — edit these if your layout changes
# ---------------------------------------------------------------------------
INPUT_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/"
    "all_envelope_rows_wide_combined_base.csv"
)
FLAGGED_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/flagged-flys-truth.csv"
)
VIDEOS_ROOT = Path("/securedstorage/DATAsec/cole/Data-secured/")
OUTPUT_CSV = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/"
    "blinded_video_scores.csv"
)
SEED_FILE = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/"
    "blinded_video_scoring_seed.json"
)
SKIPPED_FILE = Path(
    "/home/ramanlab/Documents/cole/Data/CSVs-ALL-Opto-Flys/"
    "blinded_video_scoring_skipped.json"
)

# Black-box overlay (x, y, width, height) in raw video pixel coords
BLACK_BOX_X, BLACK_BOX_Y, BLACK_BOX_W, BLACK_BOX_H = 0, 0, 300, 60

# Playback / plot
MAX_SECONDS = 90
DEFAULT_FPS = 40.0
RANDOM_SEED = 42

# Trace-plot styling
ODOR_ON_S = 32.0
ODOR_OFF_S = 62.0
THRESHOLD_STD_MULT = 3.0
FIXED_Y_MAX = 100.0
ODOR_SHADE_COLOR = "#9e9e9e"
ODOR_SHADE_ALPHA = 0.20
THRESHOLD_COLOR = "tab:red"
THRESHOLD_ALPHA = 0.9
THRESHOLD_LW = 1.0
TRACE_COLOR = "black"
TRACE_LW = 1.2
ODOR_LINE_LW = 1.0
MAX_FRAMES = 3600  # dir_val columns to use (~90 s at 40 fps)

# Display sizes
VIDEO_W, VIDEO_H = 1080, 1080

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print("Loading envelope data …")
    df = pd.read_csv(INPUT_CSV)
    df = df[df["trial_type"].str.strip().str.lower() == "testing"].copy()
    print(f"  Testing rows: {len(df)}")

    # Keep only rows with "distances" in trial_label
    mask_distances = df["trial_label"].str.contains("distances", case=False, na=False)
    before = len(df)
    df = df[mask_distances].copy()
    print(f"  Keeping only 'distances' rows: {before} → {len(df)} rows ({before - len(df)} removed)")

    # Exclude testing_11 trials
    import re
    mask = pd.Series([
        not re.match(r'testing_11(?:\D|$)', str(tl).strip())
        for tl in df["trial_label"]
    ], index=df.index)
    before = len(df)
    df = df[mask].copy()
    print(f"  Excluding testing_11: {before} → {len(df)} rows ({before - len(df)} removed)")

    return df


def load_exclusion_set() -> set[tuple[str, int]]:
    if not FLAGGED_CSV.exists():
        print("  No flagged CSV found, skipping exclusions.")
        return set()
    flagged = pd.read_csv(FLAGGED_CSV)
    score_col = [c for c in flagged.columns if "State" in c][0]
    flagged[score_col] = pd.to_numeric(flagged[score_col], errors="coerce")
    bad = flagged[flagged[score_col] <= 0]
    exclude: set[tuple[str, int]] = set()
    for _, row in bad.iterrows():
        exclude.add((str(row["fly"]).strip(), int(row["fly_number"])))
    exclude.add(("october_14_batch_1", 1))
    return exclude


def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    exclude = load_exclusion_set()
    if not exclude:
        return df
    before = len(df)
    mask = pd.Series(
        [
            (str(r["fly"]).strip(), int(r["fly_number"])) in exclude
            for _, r in df.iterrows()
        ],
        index=df.index,
    )
    df = df[~mask].copy()
    print(f"  Exclusion: {before} → {len(df)} rows ({before - len(df)} removed)")
    return df


# ---------------------------------------------------------------------------
# Video path resolution
# ---------------------------------------------------------------------------

def _strip_leading_zeros_in_dates(name: str) -> str:
    """september_09_fly_1 -> september_9_fly_1 (strip leading zeros after month)."""
    import re
    return re.sub(r'(?<=_)0+(\d)', r'\1', name)


def _try_video_patterns(dataset_path: Path, fly: str, base_fly: str, trial_label: str) -> Path | None:
    """Try Pattern A and B with a given base_fly name for the video filename."""
    stem = f"{base_fly}_{trial_label}"
    filename = f"{stem}_distance_annotated.mp4"

    # Try with original fly name AND base_fly name (in case fly dir was stripped of leading zeros too)
    for fly_dir_name in [fly, base_fly]:
        # Pattern A: {fly_dir}/{base_fly}_{trial}/{base_fly}_{trial}_distance_annotated.mp4
        path_a = dataset_path / fly_dir_name / stem / filename
        if path_a.exists():
            return path_a

        # Pattern B: {fly_dir}/videos_with_rms/testing/{filename}
        path_b = dataset_path / fly_dir_name / "videos_with_rms" / "testing" / filename
        if path_b.exists():
            return path_b

    return None


def resolve_video_path(dataset: str, fly: str, trial_label: str) -> Path | None:
    """Find the _distance_annotated.mp4 for a given trial.

    Handles: _rig_N suffixes, flagged/ subfolder, leading-zero date mismatches,
    and trial_labels with suffixes like _fly1_angle_distance_rms_envelope.
    """
    import re

    # Extract the core trial ID from trial_label
    # E.g., "testing_1_fly1_angle_distance_rms_envelope" -> "testing_1"
    match = re.match(r'(testing_\d+)', trial_label)
    core_trial_label = match.group(1) if match else trial_label

    # Dataset roots to search: normal + flagged/
    dataset_roots = [VIDEOS_ROOT / dataset]
    if "flagged" in dataset.lower():
        dataset_roots.append(VIDEOS_ROOT / "flagged" / dataset)

    # Fly name variants: original, stripped _rig_N, stripped leading zeros
    base_flies = [fly]
    stripped = re.sub(r'_rig_\d+$', '', fly)
    if stripped != fly:
        base_flies.append(stripped)
    # Also try stripping leading zeros in date portions
    for bf in list(base_flies):
        no_zeros = _strip_leading_zeros_in_dates(bf)
        if no_zeros != bf:
            base_flies.append(no_zeros)

    for ds_root in dataset_roots:
        for base_fly in base_flies:
            # Try with core trial label first
            result = _try_video_patterns(ds_root, fly, base_fly, core_trial_label)
            if result:
                return result
            # Also try with stripped zeros in trial label
            tl_no_zeros = _strip_leading_zeros_in_dates(core_trial_label)
            if tl_no_zeros != core_trial_label:
                result = _try_video_patterns(ds_root, fly, base_fly, tl_no_zeros)
                if result:
                    return result

    return None


# ---------------------------------------------------------------------------
# Randomisation helpers
# ---------------------------------------------------------------------------

def randomize_order(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, list[int]]:
    rng = np.random.RandomState(seed)
    indices = df.index.tolist()
    rng.shuffle(indices)
    shuffled = df.loc[indices].reset_index(drop=True)
    return shuffled, indices


def save_seed_info(seed: int, order: list[int]) -> None:
    info = {"random_seed": seed, "order": order}
    with SEED_FILE.open("w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)


def load_seed_info() -> dict | None:
    if not SEED_FILE.exists():
        return None
    with SEED_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Envelope extraction & threshold
# ---------------------------------------------------------------------------

def get_dir_val_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("dir_val_")]
    cols.sort(key=lambda c: int(c.split("_")[-1]))
    return cols


def extract_envelope(row: pd.Series, dir_val_cols: list[str]) -> np.ndarray:
    vals = row[dir_val_cols].to_numpy(dtype=float)
    vals = vals[:MAX_FRAMES]
    finite_mask = np.isfinite(vals)
    if not finite_mask.any():
        return np.empty(0, dtype=float)
    last_finite = int(np.max(np.where(finite_mask)[0]))
    return vals[: last_finite + 1]


def compute_theta(env: np.ndarray, fps: float) -> float:
    n_before = int(round(ODOR_ON_S * fps))
    before = env[:n_before]
    before = before[np.isfinite(before)]
    if before.size < 3:
        return float("nan")
    med = float(np.nanmedian(before))
    mad = float(np.nanmedian(np.abs(before - med)))
    return med + THRESHOLD_STD_MULT * 1.4826 * mad


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_trace(fig: plt.Figure, env: np.ndarray, fps: float) -> plt.Line2D | None:
    """Plot the envelope trace. Returns the playback cursor line (or None)."""
    fig.clear()
    ax = fig.add_subplot(111)
    t = np.arange(len(env), dtype=float) / fps

    ax.plot(t, env, linewidth=TRACE_LW, color=TRACE_COLOR)
    ax.axvline(ODOR_ON_S, linestyle="--", linewidth=ODOR_LINE_LW, color="black")
    ax.axvline(ODOR_OFF_S, linestyle="--", linewidth=ODOR_LINE_LW, color="black")
    ax.axvspan(ODOR_ON_S, ODOR_OFF_S, alpha=ODOR_SHADE_ALPHA, color=ODOR_SHADE_COLOR)

    theta = compute_theta(env, fps)
    if math.isfinite(theta):
        ax.axhline(theta, linestyle="-", linewidth=THRESHOLD_LW,
                    color=THRESHOLD_COLOR, alpha=THRESHOLD_ALPHA)

    # Playback cursor — vertical line that tracks the current video frame
    cursor_line = ax.axvline(0.0, linestyle="-", linewidth=1.5, color="tab:blue", alpha=0.8)

    ax.set_ylim(0, FIXED_Y_MAX)
    x_max = t[-1] if len(t) > 0 else 120.0
    ax.set_xlim(0, x_max)
    ax.margins(x=0, y=0.02)
    ax.set_ylabel("Max Distance x Angle %", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return cursor_line


# ---------------------------------------------------------------------------
# Resume / save helpers
# ---------------------------------------------------------------------------

def trial_key(row: pd.Series) -> tuple[str, str, int, str]:
    """Create a unique key for resume matching.
    Uses core trial label (testing_1) for matching, even if full label has suffixes.
    """
    full_trial_label = str(row["trial_label"]).strip()
    core_trial_label = _extract_core_trial_label(full_trial_label)
    return (
        str(row["dataset"]).strip(),
        str(row["fly"]).strip(),
        int(row["fly_number"]),
        core_trial_label,
    )


def load_existing_scores() -> set[tuple[str, str, int, str]]:
    if not OUTPUT_CSV.exists():
        return set()
    try:
        df = pd.read_csv(OUTPUT_CSV)
    except Exception:
        return set()
    scored: set[tuple[str, str, int, str]] = set()
    for _, row in df.iterrows():
        scored.add((
            str(row["dataset"]).strip(),
            str(row["fly"]).strip(),
            int(row["fly_number"]),
            str(row["trial_label"]).strip(),
        ))
    return scored


def load_skipped_trials() -> set[tuple[str, str, int, str]]:
    """Load set of skipped trial keys from persistent storage."""
    if not SKIPPED_FILE.exists():
        return set()
    try:
        with SKIPPED_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return set(tuple(k) for k in data.get("skipped", []))
    except Exception:
        return set()


def save_skipped_trials(skipped: set[tuple[str, str, int, str]]) -> None:
    """Save set of skipped trial keys to persistent storage."""
    data = {"skipped": [list(k) for k in skipped]}
    with SKIPPED_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _extract_core_trial_label(full_label: str) -> str:
    """Extract core trial ID: testing_1_fly1_distances_... -> testing_1"""
    import re
    match = re.match(r'(testing_\d+)', full_label)
    return match.group(1) if match else full_label


def save_score(row: pd.Series, score: int, comment: str) -> None:
    file_exists = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0
    full_trial_label = str(row["trial_label"]).strip()
    core_trial_label = _extract_core_trial_label(full_trial_label)

    row_data = {
        "dataset": str(row["dataset"]).strip(),
        "fly": str(row["fly"]).strip(),
        "fly_number": int(row["fly_number"]),
        "trial_type": str(row["trial_type"]).strip(),
        "trial_label": core_trial_label,
        "user_score": score,
        "comment": comment,
    }
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row_data.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class BlindedVideoScoringApp:
    def __init__(
        self,
        master: tk.Tk,
        df: pd.DataFrame,
        dir_val_cols: list[str],
        video_paths: list[Path],
        scored_keys: set[tuple[str, str, int, str]],
        show_skipped: bool = False,
    ) -> None:
        self.master = master
        self.df = df
        self.dir_val_cols = dir_val_cols
        self.video_paths = video_paths
        self.total = len(df)
        self.show_skipped = show_skipped

        # Load persistent skipped trials
        self.skipped_keys: set[tuple[str, str, int, str]] = load_skipped_trials()

        if self.show_skipped:
            # Show ONLY previously skipped trials (that haven't been scored yet)
            self.pending_indices: list[int] = [
                idx for idx in range(self.total)
                if trial_key(self.df.iloc[idx]) in self.skipped_keys
                and trial_key(self.df.iloc[idx]) not in scored_keys
            ]
        else:
            # Normal mode: skip scored and skipped trials
            self.pending_indices: list[int] = [
                idx for idx in range(self.total)
                if trial_key(self.df.iloc[idx]) not in scored_keys
                and trial_key(self.df.iloc[idx]) not in self.skipped_keys
            ]
        self.current_pending_pos = 0
        self.already_scored = self.total - len(self.pending_indices)

        # Playback state
        self.cap: cv2.VideoCapture | None = None
        self.fps = DEFAULT_FPS
        self.playing = False
        self.frame_counter = 0
        self.max_playback_frames = 0
        self.slider_active = False
        self.slider_resume_playback = False
        self.slider_updating = False

        # matplotlib rcParams
        plt.rcParams.update({
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
        })

        # --- Window ---
        title = "Blinded Fly Scoring — SKIPPED TRIALS" if self.show_skipped else "Blinded Fly Scoring — Video + Trace"
        self.master.title(title)
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)
        self.master.configure(bg="#f8f8fb")

        style = ttk.Style(self.master)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # --- Top frame: video (left) + trace (right) — fixed height ---
        top = tk.Frame(self.master, bg="#f8f8fb", height=VIDEO_H)
        top.pack(fill=tk.X, padx=8, pady=(8, 0))
        top.pack_propagate(False)

        self.video_canvas = tk.Canvas(
            top, width=VIDEO_W, height=VIDEO_H, bg="black", highlightthickness=0
        )
        self.video_canvas.pack(side=tk.LEFT, padx=(0, 4))

        # Trace plot: match video height, fill remaining width
        fig_h_in = VIDEO_H / 100.0  # match video pixel height at 100 dpi
        self.fig = plt.Figure(figsize=(12, fig_h_in), dpi=100)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=top)
        self.mpl_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Playback cursor line on the trace plot (set per-trial)
        self.cursor_line: plt.Line2D | None = None

        # --- Slider ---
        slider_frame = tk.Frame(self.master, bg="#f8f8fb")
        slider_frame.pack(fill=tk.X, padx=12, pady=(6, 0))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_scale = ttk.Scale(
            slider_frame, from_=0.0, to=1.0, orient="horizontal",
            variable=self.progress_var,
        )
        self.progress_scale.pack(fill="x")
        self.progress_scale.configure(command=self._on_slider_move)
        self.progress_scale.bind("<ButtonPress-1>", self._on_slider_press)
        self.progress_scale.bind("<ButtonRelease-1>", self._on_slider_release)

        self.time_label = tk.Label(
            self.master, text="0.00 s / 0.00 s", bg="#f8f8fb",
            font=("Helvetica", 16),
        )
        self.time_label.pack(anchor="e", padx=16)

        # --- Progress label ---
        self.progress_text = tk.Label(
            self.master, text="", bg="#f8f8fb", font=("Helvetica", 18, "bold"),
        )
        self.progress_text.pack(anchor="w", padx=16, pady=(6, 0))

        # --- Score controls ---
        score_frame = tk.Frame(self.master, bg="#f8f8fb")
        score_frame.pack(fill=tk.X, padx=16, pady=(8, 4))

        tk.Label(score_frame, text="Score (0–5):", font=("Helvetica", 22, "bold"),
                 bg="#f8f8fb").pack(side=tk.LEFT, padx=(0, 16))

        self.score_var = tk.IntVar(value=-1)
        self._score_buttons: list[tk.Button] = []
        for val in range(6):
            btn = tk.Button(
                score_frame, text=str(val), font=("Helvetica", 28, "bold"),
                width=3, height=1, relief="raised", bd=3,
                command=lambda v=val: self._select_score(v),
            )
            btn.pack(side=tk.LEFT, padx=6)
            self._score_buttons.append(btn)

        # Or type a score directly
        tk.Label(score_frame, text="  or type:", font=("Helvetica", 20),
                 bg="#f8f8fb").pack(side=tk.LEFT, padx=(20, 6))
        self.score_entry_var = tk.StringVar()
        self.score_entry = tk.Entry(
            score_frame, textvariable=self.score_entry_var, width=3,
            font=("Helvetica", 28, "bold"), justify="center",
        )
        self.score_entry.pack(side=tk.LEFT, padx=(0, 8))
        self.score_entry_var.trace_add("write", self._on_score_entry_changed)

        # --- Comment ---
        comment_frame = tk.Frame(self.master, bg="#f8f8fb")
        comment_frame.pack(fill=tk.X, padx=16, pady=(4, 6))

        tk.Label(comment_frame, text="Comment:", font=("Helvetica", 20),
                 bg="#f8f8fb").pack(side=tk.LEFT, padx=(0, 8))
        self.comment_var = tk.StringVar()
        self.comment_entry = tk.Entry(
            comment_frame, textvariable=self.comment_var, width=40,
            font=("Helvetica", 20),
        )
        self.comment_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 16))

        # --- Buttons ---
        btn_frame = tk.Frame(self.master, bg="#f8f8fb")
        btn_frame.pack(pady=(4, 14))

        self.submit_btn = tk.Button(
            btn_frame, text="Submit", font=("Helvetica", 22, "bold"),
            width=10, height=1, bg="#4CAF50", fg="white", relief="raised", bd=3,
            command=self._on_submit,
        )
        self.submit_btn.grid(row=0, column=0, padx=12)
        self.replay_btn = tk.Button(
            btn_frame, text="Replay", font=("Helvetica", 22, "bold"),
            width=10, height=1, bg="#2196F3", fg="white", relief="raised", bd=3,
            command=self._on_replay,
        )
        self.replay_btn.grid(row=0, column=1, padx=12)
        self.skip_btn = tk.Button(
            btn_frame, text="Skip", font=("Helvetica", 22, "bold"),
            width=10, height=1, bg="#FF9800", fg="white", relief="raised", bd=3,
            command=self._on_skip,
        )
        self.skip_btn.grid(row=0, column=2, padx=12)
        self.exit_btn = tk.Button(
            btn_frame, text="Save & Exit", font=("Helvetica", 22, "bold"),
            width=12, height=1, bg="#f44336", fg="white", relief="raised", bd=3,
            command=self._on_exit,
        )
        self.exit_btn.grid(row=0, column=3, padx=12)

        # Bind Enter to submit
        self.master.bind("<Return>", lambda _: self._on_submit())

        # Focus the score entry so user can just type a number right away
        self.score_entry.focus_set()

        # Resume message
        if self.show_skipped:
            messagebox.showinfo(
                "Skipped Trials Mode",
                f"Showing {len(self.pending_indices)} previously skipped trials.\n"
                f"Score them now — they'll be removed from the skipped list once scored.",
            )
        elif self.already_scored > 0:
            messagebox.showinfo(
                "Resuming",
                f"Resuming: {self.already_scored} of {self.total} already scored.\n"
                f"{len(self.pending_indices)} remaining.",
            )

        # Show first trial
        if self.pending_indices:
            self._show_current_trial()
        else:
            self._show_completion()

    # ---- Score helpers ----

    def _select_score(self, val: int) -> None:
        self.score_var.set(val)
        self.score_entry_var.set(str(val))
        self._highlight_score_button(val)

    def _highlight_score_button(self, val: int) -> None:
        for i, btn in enumerate(self._score_buttons):
            if i == val:
                btn.configure(bg="#2196F3", fg="white", relief="sunken")
            else:
                btn.configure(bg="#d9d9d9", fg="black", relief="raised")

    def _on_score_entry_changed(self, *_args: object) -> None:
        raw = self.score_entry_var.get().strip()
        if raw in ("0", "1", "2", "3", "4", "5"):
            val = int(raw)
            self.score_var.set(val)
            self._highlight_score_button(val)

    # ---- Video helpers ----

    def _draw_frame(self, frame: np.ndarray) -> None:
        # Black box overlay
        cv2.rectangle(
            frame,
            (BLACK_BOX_X, BLACK_BOX_Y),
            (BLACK_BOX_X + BLACK_BOX_W, BLACK_BOX_Y + BLACK_BOX_H),
            (0, 0, 0),
            -1,
        )
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_AREA
        )
        im = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=im)
        self.video_canvas.imgtk = imgtk  # prevent GC
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def _update_time_readout(self, frame_index: int) -> None:
        current_s = frame_index / self.fps if self.fps > 0 else 0.0
        total_s = self.max_playback_frames / self.fps if self.fps > 0 else 0.0
        self.time_label.config(text=f"{current_s:0.2f} s / {total_s:0.2f} s")

    def _update_cursor(self, frame_index: int) -> None:
        if self.cursor_line is not None:
            current_s = frame_index / self.fps if self.fps > 0 else 0.0
            self.cursor_line.set_xdata([current_s, current_s])
            self.mpl_canvas.draw_idle()

    def _set_progress(self, frame_index: int) -> None:
        if not self.slider_active:
            self.slider_updating = True
            self.progress_var.set(frame_index)
            self.slider_updating = False
        self._update_time_readout(frame_index)
        self._update_cursor(frame_index)

    def _advance(self) -> None:
        if not self.playing or self.cap is None:
            return
        if self.max_playback_frames and self.frame_counter >= self.max_playback_frames:
            self.playing = False
            return
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            return
        self.frame_counter += 1
        self._draw_frame(frame)
        self._set_progress(max(0, self.frame_counter - 1))
        if self.max_playback_frames and self.frame_counter >= self.max_playback_frames:
            self.playing = False
            return
        delay = int(1000 / max(1.0, self.fps))
        self.master.after(delay, self._advance)

    def _seek_to_frame(self, target: int, resume: bool) -> None:
        if self.cap is None:
            return
        target = max(0, min(int(round(target)), max(0, self.max_playback_frames - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self.cap.read()
        if ok:
            self.frame_counter = target + 1
            self._draw_frame(frame)
            self._set_progress(target)
        else:
            self.frame_counter = target
        if resume and (not self.max_playback_frames or self.frame_counter < self.max_playback_frames):
            self.playing = True
            delay = int(1000 / max(1.0, self.fps))
            self.master.after(delay, self._advance)
        else:
            self.playing = False

    # ---- Slider callbacks ----

    def _on_slider_move(self, value: str) -> None:
        if self.slider_updating:
            return
        try:
            self._update_time_readout(int(float(value)))
        except (TypeError, ValueError):
            pass

    def _on_slider_press(self, _event: tk.Event) -> None:
        if self.cap is None:
            return
        self.slider_active = True
        self.slider_resume_playback = self.playing
        self.playing = False

    def _on_slider_release(self, _event: tk.Event) -> None:
        if self.cap is None:
            self.slider_active = False
            return
        self.slider_active = False
        self._seek_to_frame(int(self.progress_var.get()), self.slider_resume_playback)

    # ---- Trial display ----

    def _show_current_trial(self) -> None:
        if self.current_pending_pos >= len(self.pending_indices):
            self._show_completion()
            return

        idx = self.pending_indices[self.current_pending_pos]
        row = self.df.iloc[idx]

        # Trace
        env = extract_envelope(row, self.dir_val_cols)
        fps = float(row.get("fps", DEFAULT_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = DEFAULT_FPS
        self.cursor_line = plot_trace(self.fig, env, fps)
        self.mpl_canvas.draw()

        # Video
        if self.cap is not None:
            self.cap.release()
        vp = self.video_paths[idx]
        self.cap = cv2.VideoCapture(str(vp))
        if not self.cap.isOpened():
            self.cap = None

        fps_vid = 0.0
        if self.cap is not None:
            fps_vid = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps_vid if fps_vid and fps_vid > 0 else fps
        self.max_playback_frames = int(round(self.fps * MAX_SECONDS))
        if self.cap is not None:
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                self.max_playback_frames = min(self.max_playback_frames, total_frames)

        slider_max = max(1.0, float(self.max_playback_frames - 1))
        self.progress_scale.configure(to=slider_max)
        self._set_progress(0)
        self.frame_counter = 0

        # Reset controls
        self.score_var.set(-1)
        self.score_entry_var.set("")
        self._highlight_score_button(-1)  # unhighlight all
        self.comment_var.set("")
        self.submit_btn.config(state=tk.NORMAL)
        self.score_entry.focus_set()

        fly_number = int(row.get("fly_number", 0))
        scored_so_far = self.already_scored + self.current_pending_pos
        self.progress_text.config(
            text=f"Trial {scored_so_far + 1} of {self.total}  —  Fly #{fly_number}"
        )

        # Start playback
        if self.cap is not None:
            self.playing = True
            self.master.after(0, self._advance)

    def _on_submit(self) -> None:
        score = self.score_var.get()
        if score < 0:
            messagebox.showwarning("Select a score", "Please select a score (0–5).")
            return
        self.playing = False

        idx = self.pending_indices[self.current_pending_pos]
        row = self.df.iloc[idx]
        save_score(row, score, self.comment_var.get().strip())

        # Remove from skipped set once scored
        key = trial_key(row)
        if key in self.skipped_keys:
            self.skipped_keys.discard(key)
            save_skipped_trials(self.skipped_keys)

        self.current_pending_pos += 1
        if self.current_pending_pos >= len(self.pending_indices):
            self._show_completion()
        else:
            self._show_current_trial()

    def _on_replay(self) -> None:
        if self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_counter = 0
        self._set_progress(0)
        if not self.playing:
            self.playing = True
            self.master.after(0, self._advance)

    def _on_skip(self) -> None:
        """Skip this trial, persist it, and move to the next one."""
        self.playing = False
        idx = self.pending_indices[self.current_pending_pos]
        row = self.df.iloc[idx]
        key = trial_key(row)
        self.skipped_keys.add(key)
        save_skipped_trials(self.skipped_keys)

        self.current_pending_pos += 1
        if self.current_pending_pos >= len(self.pending_indices):
            self._show_completion()
        else:
            self._show_current_trial()

    def _show_completion(self) -> None:
        self.playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "All trials scored!\nThank you.",
                ha="center", va="center", fontsize=20, transform=ax.transAxes)
        ax.set_axis_off()
        self.mpl_canvas.draw()

        self.video_canvas.delete("all")
        self.progress_text.config(text=f"Complete: {self.total} of {self.total}")
        self.submit_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Complete", f"All {self.total} trials have been scored.")

    def _on_exit(self) -> None:
        self.playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.master.destroy()

    def _on_close(self) -> None:
        self._on_exit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Blinded video + trace scoring GUI")
    parser.add_argument(
        "--show-skipped", action="store_true",
        help="Show only previously skipped trials so you can score them",
    )
    args = parser.parse_args()

    # 1. Load & filter
    df = load_data()
    df = apply_exclusions(df)
    dir_val_cols = get_dir_val_cols(df)
    print(f"  dir_val columns: {len(dir_val_cols)}")
    print(f"  Trial count after exclusions: {len(df)}")

    # 2. Resolve video paths and keep only rows with a video
    print("Resolving video paths …")
    video_paths: list[Path | None] = []
    for _, row in df.iterrows():
        vp = resolve_video_path(
            str(row["dataset"]).strip(),
            str(row["fly"]).strip(),
            str(row["trial_label"]).strip(),
        )
        video_paths.append(vp)

    has_video = [vp is not None for vp in video_paths]
    n_found = sum(has_video)
    print(f"  Videos found: {n_found} / {len(df)}")
    if n_found == 0:
        print("No videos found. Check VIDEOS_ROOT path.")
        sys.exit(1)

    # Filter to only rows with a video
    keep_mask = pd.Series(has_video, index=df.index)
    df = df[keep_mask].reset_index(drop=True)
    video_paths_clean: list[Path] = [vp for vp in video_paths if vp is not None]
    print(f"  Trials with video: {len(df)}")

    # 3. Seed & randomise
    seed_info = load_seed_info()
    seed = seed_info["random_seed"] if seed_info else RANDOM_SEED
    print(f"  Seed: {seed}")

    df_shuffled, order = randomize_order(df, seed)
    # Reorder video paths to match shuffled df
    video_paths_shuffled = [video_paths_clean[i] for i in order]
    save_seed_info(seed, order)

    # 4. Resume
    scored_keys = load_existing_scores()
    print(f"  Already scored: {len(scored_keys)}")

    # 5. Launch
    root = tk.Tk()
    BlindedVideoScoringApp(root, df_shuffled, dir_val_cols, video_paths_shuffled, scored_keys, show_skipped=args.show_skipped)
    root.mainloop()


if __name__ == "__main__":
    main()

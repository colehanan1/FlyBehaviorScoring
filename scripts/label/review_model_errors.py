#!/usr/bin/env python3
"""Review trials where the model disagrees with your blinded score.

Shows each disagreement trial exactly like the blinded scorer — video with the
odor label blacked out on the left, envelope trace on the right — plus a header
comparing YOUR score with the MODEL's out-of-fold prediction. You then decide:

    Keep my score      the label was right, the model is wrong
    Re-score to N      the label was wrong; record the corrected score
    Unsure / Skip      come back to it later
    Bad tracking       the trace/video is unusable

Decisions are written to outputs/ordinal_scorer/label_review_decisions.csv.
YOUR ORIGINAL blinded_video_scores.csv FILES ARE NEVER TOUCHED — applying
corrections is a separate, explicit step once you've finished reviewing.

The queue comes from outputs/ordinal_scorer/oof_predictions.csv (built by
scripts/eval/make_oof_predictions.py). Every prediction there is out-of-fold:
the model that predicted a trial never trained on it, so a confident
disagreement is honest evidence of either a mislabel or a video-only response.

Usage:
    python scripts/label/review_model_errors.py                  # boundary + >|1| errors
    python scripts/label/review_model_errors.py --flags boundary+2
    python scripts/label/review_model_errors.py --source new
    python scripts/label/review_model_errors.py --include-within1
    python scripts/label/review_model_errors.py --dry-run        # queue + video check, no GUI
    python scripts/label/review_model_errors.py --summary        # progress so far
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import blinded_video_scoring as bvs  # noqa: E402  (safe: __main__-guarded)

OOF_CSV = PROJECT_ROOT / "outputs" / "ordinal_scorer" / "oof_predictions.csv"
DECISIONS_CSV = PROJECT_ROOT / "outputs" / "ordinal_scorer" / "label_review_decisions.csv"
CACHE_DIR = PROJECT_ROOT / "outputs" / "thesis_figures" / "cache"

#: oof `source` column -> dataset preset in blinded_video_scoring.DATASETS
SOURCE_PRESET = {"old": "all", "new": "new"}
SCORE_LEVELS = [-1, 0, 1, 2, 3, 4, 5]
DEFAULT_FLAGS = ["boundary+2", "boundary", "gt1"]

DECISION_FIELDS = ["dataset", "fly", "fly_number", "trial_label", "source",
                   "human_score", "model_pred", "p_pred", "flag",
                   "decision", "new_score", "comment", "reviewed_at"]


def trial_key(row) -> tuple:
    return (str(row["dataset"]), str(row["fly"]), int(row["fly_number"]),
            str(row["trial_label"]), str(row["source"]))


# ---------------------------------------------------------------------------
# Queue construction
# ---------------------------------------------------------------------------
def load_queue(args) -> pd.DataFrame:
    if not OOF_CSV.exists():
        sys.exit(f"{OOF_CSV} not found — run scripts/eval/make_oof_predictions.py first.")
    df = pd.read_csv(OOF_CSV)

    flags = list(args.flags)
    if args.include_within1:
        flags.append("within1")
    df = df[df["flag"].isin(flags)]
    if args.source != "both":
        df = df[df["source"] == args.source]
    if args.min_p is not None:
        df = df[df["p_pred"] >= args.min_p]

    # Severity first, then the model's confidence — most suspicious labels first.
    sev = pd.Categorical(df["flag"], ["boundary+2", "boundary", "gt1", "within1"],
                         ordered=True)
    df = (df.assign(_sev=sev).sort_values(["_sev", "p_pred"], ascending=[True, False])
            .drop(columns="_sev").reset_index(drop=True))
    if args.limit:
        df = df.head(args.limit)
    return df


def load_decided_keys() -> dict[tuple, dict]:
    """Previously reviewed trials (later rows win, matching the scorer's semantics)."""
    decided: dict[tuple, dict] = {}
    if DECISIONS_CSV.exists():
        for r in csv.DictReader(DECISIONS_CSV.open()):
            try:
                decided[trial_key(r)] = r
            except (KeyError, ValueError):
                continue
    return decided


def append_decision(rec: dict) -> None:
    DECISIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not DECISIONS_CSV.exists()
    with DECISIONS_CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=DECISION_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(rec)


# ---------------------------------------------------------------------------
# Trace + video lookup (traces come from the model's own cache, so what you see
# is exactly what the model saw)
# ---------------------------------------------------------------------------
def load_trace_index():
    traces = np.load(CACHE_DIR / "traces.npy")
    meta = pd.read_parquet(CACHE_DIR / "meta.parquet")
    index = {trial_key(row): i for i, row in meta.iterrows()}
    return traces, index


def resolve_video(row) -> Path | None:
    preset = SOURCE_PRESET[str(row["source"])]
    bvs.configure_paths(dataset=preset)
    return bvs.resolve_video_path(str(row["dataset"]), str(row["fly"]),
                                  str(row["trial_label"]))


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class ReviewApp:
    """Lean review UI: video + trace + your-score-vs-model header + verdict buttons."""

    def __init__(self, master, queue: pd.DataFrame, traces, trace_index):
        import tkinter as tk

        self.tk = tk
        self.master = master
        self.queue = queue
        self.traces = traces
        self.trace_index = trace_index
        self.pos = 0
        self.cap = None
        self.playing = False
        self.speed = 1.0
        self.n_done_this_session = 0

        master.title("Model-disagreement label review")
        master.configure(bg="#f8f8fb")
        master.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- header: the comparison line -----------------------------------
        self.header = tk.Label(master, text="", font=("Helvetica", 20, "bold"),
                               bg="#f8f8fb", fg="#0b0b0b")
        self.header.pack(fill=tk.X, padx=16, pady=(12, 0))
        self.subheader = tk.Label(master, text="", font=("Helvetica", 13),
                                  bg="#f8f8fb", fg="#52514e")
        self.subheader.pack(fill=tk.X, padx=16, pady=(0, 6))

        # --- video (left) + trace (right) ----------------------------------
        body = tk.Frame(master, bg="#f8f8fb")
        body.pack(fill=tk.BOTH, expand=True, padx=16)
        self.video_label = tk.Label(body, bg="black",
                                    width=bvs.VIDEO_W // 2, height=bvs.VIDEO_H // 2)
        self.video_label.pack(side=tk.LEFT, padx=(0, 12))

        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.fig = plt.Figure(figsize=(9, bvs.VIDEO_H / 200.0), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.cursor_line = None

        # --- controls -------------------------------------------------------
        ctrl = tk.Frame(master, bg="#f8f8fb")
        ctrl.pack(fill=tk.X, padx=16, pady=6)
        self.back_btn = tk.Button(ctrl, text="◀ Back", command=self._on_back)
        self.back_btn.pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(ctrl, text="Replay", command=self._replay).pack(side=tk.LEFT)
        self.speed_btn = tk.Button(ctrl, text="Speed 1x", command=self._cycle_speed)
        self.speed_btn.pack(side=tk.LEFT, padx=6)
        self.progress = tk.Label(ctrl, text="", font=("Helvetica", 12),
                                 bg="#f8f8fb", fg="#52514e")
        self.progress.pack(side=tk.RIGHT)

        # --- verdict row -----------------------------------------------------
        verdict = tk.Frame(master, bg="#f8f8fb")
        verdict.pack(fill=tk.X, padx=16, pady=(4, 4))
        self.keep_btn = tk.Button(verdict, text="✓ Keep my score",
                                  font=("Helvetica", 14, "bold"),
                                  bg="#2a78d6", fg="white",
                                  command=lambda: self._decide("keep"))
        self.keep_btn.pack(side=tk.LEFT, padx=(0, 14))

        tk.Label(verdict, text="Re-score to:", font=("Helvetica", 13),
                 bg="#f8f8fb").pack(side=tk.LEFT)
        for lvl in SCORE_LEVELS:
            tk.Button(verdict, text=str(lvl), width=3, font=("Helvetica", 13),
                      command=lambda l=lvl: self._decide("rescore", l)
                      ).pack(side=tk.LEFT, padx=2)

        tk.Button(verdict, text="Unsure / Skip",
                  command=lambda: self._decide("unsure")).pack(side=tk.LEFT, padx=14)
        tk.Button(verdict, text="Bad tracking",
                  command=lambda: self._decide("bad_tracking")).pack(side=tk.LEFT)

        comment_row = tk.Frame(master, bg="#f8f8fb")
        comment_row.pack(fill=tk.X, padx=16, pady=(0, 10))
        tk.Label(comment_row, text="Comment:", font=("Helvetica", 12),
                 bg="#f8f8fb").pack(side=tk.LEFT)
        self.comment = tk.Entry(comment_row, font=("Helvetica", 12), width=70)
        self.comment.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)

        self._show_current()

    # --- trial presentation --------------------------------------------------
    def _row(self):
        return self.queue.iloc[self.pos]

    def _show_current(self):
        import cv2

        if self.pos >= len(self.queue):
            self._finish()
            return
        row = self._row()

        top3 = sorted(((float(row[f"p_{l}"]), l) for l in SCORE_LEVELS), reverse=True)[:3]
        probs = "   ".join(f"p({l})={p:.2f}" for p, l in top3 if p >= 0.01)
        self.header.config(
            text=f"YOU scored: {row['human_score']}        "
                 f"MODEL says: {row['model_pred']}  ({row['p_pred']:.2f})        "
                 f"[{row['flag']}]")
        self.subheader.config(
            text=f"{row['dataset']}  |  {row['fly']}  fly {row['fly_number']}  |  "
                 f"{row['trial_label']}  |  source: {row['source']}     {probs}")
        self.progress.config(text=f"trial {self.pos + 1} / {len(self.queue)}"
                                  f"   (decided this session: {self.n_done_this_session})")
        self.back_btn.config(state=self.tk.NORMAL if self.pos > 0 else self.tk.DISABLED)
        self.comment.delete(0, self.tk.END)

        # trace — the exact array the model was trained on
        key = trial_key(row)
        env = None
        if key in self.trace_index:
            env = self.traces[self.trace_index[key]][:bvs.MAX_FRAMES].astype(float)
        if env is not None:
            self.cursor_line = bvs.plot_trace(self.fig, env, bvs.DEFAULT_FPS)
            ax = self.fig.axes[0]
            ax.set_title(f"human {row['human_score']}  vs  model {row['model_pred']}",
                         fontsize=11)
        else:
            self.fig.clear()
            self.fig.add_subplot(111).text(.5, .5, "trace not in cache",
                                           ha="center", va="center")
            self.cursor_line = None
        self.canvas.draw()

        # video
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        vid = resolve_video(row)
        if vid is None:
            self.video_label.config(image="", text="\n  VIDEO NOT FOUND  \n"
                                    f"\n{row['dataset']} / {row['fly']}",
                                    font=("Helvetica", 16), fg="white",
                                    compound=self.tk.CENTER)
            self.playing = False
        else:
            self.cap = cv2.VideoCapture(str(vid))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or bvs.DEFAULT_FPS
            self.frame_i = 0
            self.playing = True
            self._advance()

    def _advance(self):
        import cv2
        from PIL import Image, ImageTk

        if not self.playing or self.cap is None:
            return
        skip = max(int(self.speed), 1)
        for _ in range(skip - 1):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok or self.frame_i / self.fps >= bvs.MAX_SECONDS:
            self.playing = False
            return
        self.frame_i += skip

        # blinding box: keep the odor label hidden so the re-score stays blind
        cv2.rectangle(frame, (bvs.BLACK_BOX_X, bvs.BLACK_BOX_Y),
                      (bvs.BLACK_BOX_X + bvs.BLACK_BOX_W,
                       bvs.BLACK_BOX_Y + bvs.BLACK_BOX_H), (0, 0, 0), -1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (bvs.VIDEO_W // 2, bvs.VIDEO_H // 2))
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.video_label.config(image=img, text="")
        self.video_label.image = img

        t = self.frame_i / self.fps
        if self.cursor_line is not None and self.frame_i % 8 == 0:
            self.cursor_line.set_xdata([t, t])
            self.canvas.draw_idle()

        delay = max(int(1000 / self.fps / min(self.speed, 1.0) * 1), 1) \
            if self.speed < 1 else max(int(1000 / self.fps), 1)
        self.master.after(delay, self._advance)

    def _on_back(self):
        """Revisit the previous trial. Re-deciding it appends a newer row, which
        wins over the earlier one everywhere decisions are read (last row wins)."""
        if self.pos > 0:
            self.playing = False
            self.pos -= 1
            self._show_current()

    def _replay(self):
        import cv2
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_i = 0
            if not self.playing:
                self.playing = True
                self._advance()

    def _cycle_speed(self):
        order = [1.0, 2.0, 3.0, 4.0, 0.5]
        self.speed = order[(order.index(self.speed) + 1) % len(order)]
        self.speed_btn.config(text=f"Speed {self.speed:g}x")

    # --- decisions -------------------------------------------------------------
    def _decide(self, decision: str, new_score: int | None = None):
        row = self._row()
        append_decision({
            "dataset": row["dataset"], "fly": row["fly"],
            "fly_number": int(row["fly_number"]), "trial_label": row["trial_label"],
            "source": row["source"], "human_score": int(row["human_score"]),
            "model_pred": int(row["model_pred"]), "p_pred": float(row["p_pred"]),
            "flag": row["flag"], "decision": decision,
            "new_score": "" if new_score is None else int(new_score),
            "comment": self.comment.get().strip(),
            "reviewed_at": dt.datetime.now().isoformat(timespec="seconds"),
        })
        self.n_done_this_session += 1
        self.pos += 1
        self._show_current()

    def _finish(self):
        from tkinter import messagebox
        self.playing = False
        messagebox.showinfo("Done", f"Queue finished.\nDecisions saved to\n{DECISIONS_CSV}")
        self.master.destroy()

    def _on_close(self):
        self.playing = False
        if self.cap is not None:
            self.cap.release()
        self.master.destroy()


# ---------------------------------------------------------------------------
def print_summary() -> None:
    if not DECISIONS_CSV.exists():
        print("No decisions recorded yet.")
        return
    d = pd.read_csv(DECISIONS_CSV)
    d = d.drop_duplicates(subset=["dataset", "fly", "fly_number", "trial_label", "source"],
                          keep="last")
    print(f"{len(d)} trials reviewed  ({DECISIONS_CSV})")
    print("\ndecision counts:")
    print(d["decision"].value_counts().to_string())
    rescored = d[d["decision"] == "rescore"]
    if len(rescored):
        print("\nre-scores (old -> new):")
        print(rescored.groupby(["human_score", "new_score"]).size().to_string())
        agree = (pd.to_numeric(rescored["new_score"]) ==
                 rescored["model_pred"]).mean()
        print(f"\nof the re-scored trials, {agree:.0%} moved to the model's prediction")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flags", nargs="+", default=DEFAULT_FLAGS,
                    choices=["boundary+2", "boundary", "gt1", "within1"],
                    help="which disagreement severities to review")
    ap.add_argument("--include-within1", action="store_true",
                    help="also queue off-by-one disagreements")
    ap.add_argument("--source", choices=["old", "new", "both"], default="both")
    ap.add_argument("--min-p", type=float, default=None,
                    help="only disagreements where the model's confidence >= this")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redo", action="store_true",
                    help="re-review trials that already have a decision")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the queue and check video availability; no GUI")
    ap.add_argument("--summary", action="store_true",
                    help="print decision progress and exit")
    args = ap.parse_args()

    if args.summary:
        print_summary()
        return

    queue = load_queue(args)
    decided = load_decided_keys()
    if not args.redo:
        mask = [trial_key(r) not in decided for _, r in queue.iterrows()]
        n_skip = len(queue) - sum(mask)
        queue = queue[mask].reset_index(drop=True)
    else:
        n_skip = 0

    print(f"queue: {len(queue)} trials "
          f"({n_skip} already decided, use --redo to revisit)")
    if queue.empty:
        print("Nothing to review.")
        return

    if args.dry_run:
        found = 0
        for _, row in queue.iterrows():
            vid = resolve_video(row)
            status = "ok " if vid else "MISSING"
            if vid:
                found += 1
            print(f"  [{status}] {row['flag']:<11} you={row['human_score']:>2} "
                  f"model={row['model_pred']:>2} p={row['p_pred']:.2f}  "
                  f"{row['dataset']} / {row['fly']} / {row['trial_label']}")
        print(f"\nvideos found: {found}/{len(queue)}")
        return

    traces, trace_index = load_trace_index()
    import tkinter as tk
    root = tk.Tk()
    ReviewApp(root, queue, traces, trace_index)
    root.mainloop()
    print_summary()


if __name__ == "__main__":
    main()

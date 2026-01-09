import os, sys, argparse, json, re
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
import cv2, numpy as np, pandas as pd
from PIL import Image, ImageTk

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _paths import data_dir, ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from fly_behavior import (
    BASELINE_SECONDS,
    ODOR_LATENCY_CHOICES,
    SEGMENT_DEFINITIONS,
    SEGMENT_MAP,
    analysis_duration_seconds,
    choose_signal_column,
    compute_envelope,
    compute_metrics,
    compute_segment_windows,
    compute_threshold,
    extract_time_seconds,
    parse_fly_trial,
    DEFAULT_SMOOTHING_FPS,
)


# =================== CONFIG (edit as needed) ===================
METRIC_WEIGHTS = {
    # Increase/decrease to change influence on data_score (0–5 scale per metric)
    'time_fraction': 1.0,   # fraction of frames above threshold
    'auc': 1.0,             # integral above threshold (scaled by global max)
    'time_to_threshold': 1.0  # rapid responses score higher (lower time is better)
}
# ===============================================================

# Set target display size for video frames
TARGET_W, TARGET_H = 640, 640  # You can adjust these values

def main():
    ap = argparse.ArgumentParser(description="Label fly behavior videos (0–5) and compute data-driven scores.")
    ap.add_argument("-v","--videos", required=True, help="Folder with videos (recursively scanned).")
    # Legacy CSV input is now optional/unused; kept for backward compatibility.
    ap.add_argument("-d","--data", default=None, help="(Deprecated) Folder with CSV traces. If provided and matrix row not found, will fallback.")
    ap.add_argument(
        "-o",
        "--output",
        default=str(data_dir() / "scoring_results.csv"),
        help="Output master CSV.",
    )
    # NEW: matrix + maps supplied explicitly
    ap.add_argument("--matrix", required=True, help="Path to envelope matrix .npy (float16) file.")
    ap.add_argument("--codes",  required=True, help="Path to code_maps.json (column order + code maps).")
    # Optional overrides
    ap.add_argument("--dataset", default=None, help="Dataset name (e.g., 'opto_benz'). If omitted, will try to infer.")
    ap.add_argument("--trial-type", default="testing", help="Trial type name (default: 'testing').")
    ap.add_argument("--fly-name", default=None, help="Override fly name to match code_maps['fly'] keys (e.g., 'september_24_fly_1').")
    ap.add_argument("--trial-label", default=None, help="Override trial label (e.g., 'testing_3').")
    ap.add_argument(
        "--odor-latency",
        choices=sorted(ODOR_LATENCY_CHOICES.keys()),
        default="manual",
        help="Preset for odor delivery latency (seconds).",
    )
    args = ap.parse_args()
    Path(args.output).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    # Find videos
    video_exts = (".mp4",".avi",".mov",".mpg",".mpeg",".wmv",".mkv")
    videos = []
    for root, _, files in os.walk(args.videos):
        for f in files:
            fname_lower = f.lower()
            if not fname_lower.endswith(video_exts):
                continue
            if 'testing' not in fname_lower:
                continue
            stem, _ = os.path.splitext(fname_lower)
            if not stem or not stem[-1].isdigit():
                continue
            videos.append(os.path.join(root, f))
    videos.sort()
    if not videos:
        print("No videos found.")
        sys.exit(1)

    # Odor latency preset → seconds
    odor_latency_seconds = ODOR_LATENCY_CHOICES.get(args.odor_latency, 0.0)
    segment_defs = SEGMENT_DEFINITIONS
    analysis_seconds = analysis_duration_seconds(odor_latency_seconds)

    # ----- Load matrix + code maps -----
    mat = np.load(args.matrix, mmap_mode="r")
    with open(args.codes, "r") as f:
        maps_obj = json.load(f)
    COLS = maps_obj.get("column_order", [])
    CODE = maps_obj.get("code_maps", {})

    def invert(d):
        return {v: k for k, v in d.items()}

    R = {
        "dataset": invert(CODE.get("dataset", {})),
        "fly":     invert(CODE.get("fly", {})),
        "trial_type": invert(CODE.get("trial_type", {})),
        "trial_label": invert(CODE.get("trial_label", {})),
        "fps":     invert(CODE.get("fps", {})),
    }

    IDX = {name: idx for idx, name in enumerate(COLS)}
    try:
        first_dir_idx = next(i for i, name in enumerate(COLS) if name.startswith("dir_val_"))
    except StopIteration:
        raise ValueError("Matrix is missing directional envelope columns (dir_val_*).")

    # -------- Helpers to parse video → metadata keys in code_maps --------
    def _infer_trial_label_from_name(name):
        # Look for patterns like testing_3, testing_09, etc.
        m = re.search(r'(testing(?:_|-)?\d+)', name, flags=re.IGNORECASE)
        return m.group(1).lower().replace('-', '_') if m else None

    def _infer_fly_from_path(video_path):
        # Prefer parent directory if it looks like '*_fly_*'
        parent = os.path.basename(os.path.dirname(video_path))
        if "_fly_" in parent.lower():
            return parent
        # Fallback to any dir segment that matches keys in code_maps['fly']
        parts = [p for p in os.path.normpath(video_path).split(os.sep) if p]
        fly_keys = set(CODE.get("fly", {}).keys())
        for p in reversed(parts):
            if p in fly_keys:
                return p
        # Fallback to digits → choose any fly name that endswith '_fly_<digits>'
        digits = ''.join(filter(str.isdigit, os.path.splitext(os.path.basename(video_path))[0]))
        if digits:
            suffix = f"_fly_{digits}"
            candidates = [k for k in fly_keys if k.endswith(suffix)]
            if len(candidates) == 1:
                return candidates[0]
        return None

    def encode(meta_name, meta_value):
        # Map string value to code integer stored in matrix (as float16)
        cmap = CODE.get(meta_name, {})
        if meta_value in cmap:
            return float(cmap[meta_value])
        return float(0)

    def decode_fps_code(code_float16):
        # fps map in JSON uses strings; invert() produced string keys (e.g., "1":"40.0")
        try:
            key = int(round(float(code_float16)))
        except Exception:
            key = 0
        str_val = R["fps"].get(key, "0")
        try:
            return float(str_val)
        except Exception:
            return 0.0

    def find_matrix_row(video_path):
        # Determine metadata strings
        fly_name = args.fly_name or _infer_fly_from_path(video_path)
        base_name = os.path.basename(video_path)
        trial_label = (
            args.trial_label
            or _infer_trial_label_from_name(base_name)
            or _infer_trial_label_from_name(os.path.splitext(base_name)[0])
        )
        dataset_candidates = [k for k in CODE.get("dataset", {}) if k != "UNKNOWN"]
        dataset = args.dataset or (dataset_candidates[0] if dataset_candidates else "UNKNOWN")
        trial_type = args.trial_type or "testing"

        # Encode to codes
        want = {
            "dataset": encode("dataset", dataset),
            "fly": encode("fly", fly_name) if fly_name else float(0),
            "trial_type": encode("trial_type", trial_type),
            "trial_label": encode("trial_label", trial_label) if trial_label else float(0),
        }

        col_d = IDX.get("dataset")
        col_f = IDX.get("fly")
        col_tt = IDX.get("trial_type")
        col_tl = IDX.get("trial_label")

        rows = []
        for r in range(mat.shape[0]):
            ok = True
            if col_d is not None and want["dataset"] != 0 and mat[r, col_d] != want["dataset"]:
                ok = False
            if col_f is not None and want["fly"] != 0 and mat[r, col_f] != want["fly"]:
                ok = False
            if col_tt is not None and want["trial_type"] != 0 and mat[r, col_tt] != want["trial_type"]:
                ok = False
            if col_tl is not None and want["trial_label"] != 0 and mat[r, col_tl] != want["trial_label"]:
                ok = False
            if ok:
                rows.append(r)
        return rows[0] if rows else None

    # Precompute metrics + global max AUC for scaling
    items = []
    global_max_auc = {seg.key: 0.0 for seg in segment_defs}
    legacy_csv_cache = {}
    for vp in videos:
        row_idx = find_matrix_row(vp)
        if row_idx is None:
            # Optional fallback: try CSV if user passed --data
            if not args.data:
                print(f"[WARN] No matrix row for {vp}; skipping. (Provide --fly-name/--trial-label if needed.)")
                continue
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"[WARN] Cannot open video {vp}, skipping.")
            cap.release()
            continue
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_val) if fps_val and fps_val > 0 else 40.0
        frame_count_val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = int(frame_count_val) if frame_count_val and frame_count_val > 0 else None
        cap.release()

        if row_idx is not None:
            row = mat[row_idx, :]
            fps_col = IDX.get("fps")
            fps_code = row[fps_col] if fps_col is not None else 0
            fps_from_matrix = decode_fps_code(fps_code)
            fps_for_calc = fps_from_matrix if fps_from_matrix > 0 else (fps if fps > 0 else DEFAULT_SMOOTHING_FPS)
            envelope = row[first_dir_idx:].astype(np.float32)
            if envelope.size and envelope[-1] == 0.0:
                nz = np.nonzero(envelope)[0]
                if nz.size:
                    envelope = envelope[: (nz[-1] + 1)]
            time_axis = (
                np.arange(len(envelope), dtype=float) / float(fps_for_calc)
                if fps_for_calc > 0
                else np.arange(len(envelope), dtype=float)
            )
            threshold = compute_threshold(time_axis, envelope, fps_for_calc, BASELINE_SECONDS)
            source_csv = None
        else:
            base = os.path.splitext(os.path.basename(vp))[0]
            candidate = os.path.join(os.path.dirname(vp), base + ".csv")
            cp = candidate if os.path.exists(candidate) else None
            if not cp and args.data:
                if base in legacy_csv_cache:
                    cp = legacy_csv_cache[base]
                else:
                    found = None
                    for root, _, files in os.walk(args.data):
                        for f in files:
                            if f.lower().endswith('.csv') and os.path.splitext(f)[0] == base:
                                found = os.path.join(root, f)
                                break
                        if found:
                            break
                    legacy_csv_cache[base] = found
                    cp = found
            if not cp:
                print(f"[WARN] No matrix row and no CSV for {vp}; skipping.")
                continue
            df = pd.read_csv(cp)
            sig_raw = choose_signal_column(df)
            sig_series = pd.to_numeric(pd.Series(sig_raw), errors='coerce').fillna(0.0).to_numpy(dtype=float)
            fps_for_calc = fps if fps and fps > 0 else DEFAULT_SMOOTHING_FPS
            time_axis = extract_time_seconds(df, fps_for_calc)
            envelope = compute_envelope(sig_series, fps_for_calc)
            threshold = compute_threshold(time_axis, envelope, fps_for_calc, BASELINE_SECONDS)
            source_csv = cp

        segment_metrics = {}
        frames_total_analysis = int(round(fps_for_calc * analysis_seconds)) if fps_for_calc > 0 else 0
        if frames_total_analysis > 0:
            signal_limit_frames = min(len(envelope), frames_total_analysis)
        else:
            signal_limit_frames = len(envelope)

        segment_windows = compute_segment_windows(
            fps_for_calc,
            odor_latency_seconds,
            signal_limit_frames,
        )

        for seg in segment_defs:
            start_idx, end_idx = segment_windows.get(seg.key, (0, 0))
            if end_idx < start_idx:
                end_idx = start_idx
            seg_signal = envelope[start_idx:end_idx]
            metrics = compute_metrics(seg_signal, fps_for_calc, threshold) if seg_signal.size else None
            if metrics:
                global_max_auc[seg.key] = max(global_max_auc[seg.key], metrics['auc'])
            segment_metrics[seg.key] = metrics

        max_frames = int(round(fps_for_calc * analysis_seconds)) if fps_for_calc > 0 else 0
        if frame_count is not None:
            max_frames = min(max_frames, frame_count) if max_frames > 0 else frame_count
        if max_frames <= 0:
            max_frames = frames_total_analysis if frames_total_analysis > 0 else signal_limit_frames
        if max_frames <= 0:
            print(f"[WARN] No playable frames for {vp}, skipping.")
            continue

        fly_id, trial_id = parse_fly_trial(vp)
        item = {
            'video_path': vp,
            'video_file': os.path.basename(vp),
            'csv_path': source_csv,
            'matrix_row_index': row_idx if row_idx is not None else -1,
            'fly_id': fly_id,
            'trial_id': trial_id,
            'segments': segment_metrics,
            'threshold': threshold,
            'fps': fps_for_calc,
            'max_frames': max_frames,
            'display_duration': min(
                analysis_seconds,
                max_frames / fps_for_calc if fps_for_calc > 0 else analysis_seconds,
            ),
            'segment_windows': segment_windows,
            'odor_latency_seconds': odor_latency_seconds,
        }
        items.append(item)

    if not items:
        print("Nothing to score (no metric-bearing pairs).")
        sys.exit(1)

    # Load any prior annotations so we can optionally reuse them
    existing_rows = []
    existing_by_video = {}
    if os.path.exists(args.output):
        try:
            prior_df = pd.read_csv(args.output)
            existing_rows = prior_df.to_dict("records")
            for row in existing_rows:
                video_file = row.get("video_file")
                if isinstance(video_file, str) and video_file:
                    existing_by_video.setdefault(video_file, []).append(row)
        except Exception as exc:
            print(f"[WARN] Failed to load existing output CSV '{args.output}': {exc}")
            existing_rows = []
            existing_by_video = {}

    processed_videos = set()
    completed_rows = []

    def gather_output_rows():
        remaining = [
            row for row in existing_rows if row.get("video_file") not in processed_videos
        ]
        return remaining + completed_rows

    def finalize_and_close(message=None, show_dialog=True):
        nonlocal cap, slider_active, slider_resume_playback
        out_rows = gather_output_rows()
        if out_rows:
            pd.DataFrame(out_rows).to_csv(args.output, index=False)
        else:
            pd.DataFrame([]).to_csv(args.output, index=False)
        if cap:
            cap.release()
        slider_active = False
        slider_resume_playback = False
        cap = None
        if show_dialog and message:
            messagebox.showinfo("Progress Saved", message)
        root.destroy()

    # ---------- GUI ----------
    root = tk.Tk()
    root.title("Fly Behavior Scoring")
    try:
        root.iconbitmap(default='')
    except Exception:
        pass
    root.configure(bg="#f8f8fb")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("Likert.TRadiobutton", padding=6)
    style.map("Likert.TRadiobutton", background=[("active", "#e6eefc")])
    style.configure("Likert.TLabel", background="#f8f8fb")
    style.configure("Likert.TFrame", background="#f8f8fb")

    # Replace max_w, max_h computation and canvas creation:
    max_w, max_h = TARGET_W, TARGET_H
    canvas = tk.Canvas(root, width=TARGET_W, height=TARGET_H, bg="black", highlightthickness=0)
    canvas.pack()

    progress_var = tk.DoubleVar(value=0.0)
    progress_scale = ttk.Scale(
        root,
        from_=0.0,
        to=1.0,
        orient="horizontal",
        variable=progress_var,
    )
    progress_scale.pack(fill="x", padx=12, pady=(6, 0))
    progress_label = ttk.Label(root, text="0.00 s / 0.00 s", style="Likert.TLabel")
    progress_label.pack(anchor="e", padx=12, pady=(0, 6))

    info_text = (
        f"Watch the first {int(round(analysis_seconds))} seconds "
        f"(30 s baseline + {odor_latency_seconds:.1f} s latency + "
        f"{int(SEGMENT_MAP['odor'].duration_seconds)} s odor + "
        f"{int(SEGMENT_MAP['post'].duration_seconds)} s post-odor).\n"
        "Provide a rating for each rateable interval using the 0–5 scale below, then submit to reveal the data metrics."
    )
    info = ttk.Label(root, text=info_text, style="Likert.TLabel", wraplength=TARGET_W, justify="left")
    info.pack(pady=(6, 4), padx=8, anchor="w")

    rateable_segments = [seg for seg in segment_defs if seg.rateable]
    score_vars = {}

    def build_likert_scale(parent, seg):
        # Reduce padding and font size for compactness
        container = ttk.Frame(parent, padding=(6, 4), style="Likert.TFrame")
        container.pack(fill="x", pady=2)
        ttk.Label(container, text=seg.label, font=("Helvetica", 10, "bold"), style="Likert.TLabel").pack(anchor="w")

        descriptors = ttk.Frame(container, style="Likert.TFrame")
        descriptors.pack(fill="x", pady=(4, 2))
        ttk.Label(descriptors, text="No Reaction to Odor", style="Likert.TLabel", font=("Helvetica", 9)).pack(side="left")
        ttk.Label(descriptors, text="Strong Reaction to Odor", style="Likert.TLabel", font=("Helvetica", 9)).pack(side="right")

        scale_inner = ttk.Frame(container, style="Likert.TFrame")
        scale_inner.pack()

        var = tk.IntVar(value=-1)
        score_vars[seg.key] = var

        for idx, val in enumerate(range(0, 6)):
            cell = ttk.Frame(scale_inner, padding=1, style="Likert.TFrame")
            cell.grid(row=0, column=idx, padx=2)
            btn = ttk.Radiobutton(cell, variable=var, value=val, style="Likert.TRadiobutton", takefocus=0)
            btn.pack()
            ttk.Label(cell, text=str(val), style="Likert.TLabel", font=("Helvetica", 8)).pack(pady=(2, 0))

    # Rating row
    rating_frame = ttk.Frame(root, padding=(8, 4), style="Likert.TFrame")
    rating_frame.pack(pady=4, fill="x")
    for seg in rateable_segments:
        build_likert_scale(rating_frame, seg)
        
    # Buttons
    btns = ttk.Frame(root, padding=6, style="Likert.TFrame"); btns.pack(pady=8)
    submit_btn = ttk.Button(btns, text="Submit Score")
    next_btn   = ttk.Button(btns, text="Next Video", state=tk.DISABLED)
    replay_btn = ttk.Button(btns, text="Replay Video")
    exit_btn   = ttk.Button(btns, text="Save && Exit")
    submit_btn.grid(row=0,column=0,padx=4)
    next_btn.grid(row=0,column=1,padx=4)
    replay_btn.grid(row=0,column=2,padx=4)
    exit_btn.grid(row=0,column=3,padx=4)

    # Data panel (revealed post-submit)
    data_panel = ttk.Frame(root, padding=(12, 8), style="Likert.TFrame")
    data_panel.pack(pady=(4, 10), fill="both", expand=True)
    data_text = tk.Text(data_panel, height=8, wrap="word", state="disabled", bg="#ffffff", relief="flat")
    data_scroll = ttk.Scrollbar(data_panel, orient="vertical", command=data_text.yview)
    data_text.configure(yscrollcommand=data_scroll.set)
    data_text.pack(side="left", fill="both", expand=True)
    data_scroll.pack(side="right", fill="y")

    # State
    idx = 0
    cap = None
    fps = 40.0
    playing = False
    frame_counter = 0
    current_max_frames = 0
    current_duration_seconds = 0.0
    slider_active = False
    slider_resume_playback = False
    slider_updating = False

    def update_progress_readout(frame_index):
        total_seconds = current_duration_seconds
        if (not total_seconds) and current_max_frames and fps > 0:
            total_seconds = current_max_frames / fps
        if total_seconds is None:
            total_seconds = 0.0
        current_seconds = frame_index / fps if fps > 0 else 0.0
        progress_label.config(text=f"{current_seconds:0.2f} s / {total_seconds:0.2f} s")

    def set_progress(frame_index):
        nonlocal slider_updating
        if not slider_active:
            slider_updating = True
            progress_var.set(frame_index)
            slider_updating = False
        update_progress_readout(frame_index)

    def draw_frame(frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        im = Image.fromarray(frame_resized)
        imgtk = ImageTk.PhotoImage(image=im)
        canvas.imgtk = imgtk
        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    def seek_to_frame(target_frame, resume_playback):
        nonlocal cap, frame_counter, playing
        if cap is None:
            return
        if current_max_frames:
            max_index = max(0, current_max_frames - 1)
            target_frame = max(0, min(int(round(target_frame)), max_index))
        else:
            target_frame = max(0, int(round(target_frame)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = cap.read()
        if ok:
            frame_counter = target_frame + 1
            draw_frame(frame)
            set_progress(target_frame)
        else:
            frame_counter = target_frame
        if resume_playback and (not current_max_frames or frame_counter < current_max_frames):
            playing = True
            delay = int(1000 / max(1.0, fps))
            root.after(delay, advance)
        else:
            playing = False

    def on_slider_move(value):
        if slider_updating:
            return
        try:
            frame_index = float(value)
        except (TypeError, ValueError):
            frame_index = 0.0
        update_progress_readout(frame_index)

    progress_scale.configure(command=on_slider_move)

    def on_slider_press(event):
        nonlocal slider_active, slider_resume_playback, playing
        if cap is None:
            return
        slider_active = True
        slider_resume_playback = playing
        playing = False

    def on_slider_release(event):
        nonlocal slider_active
        if cap is None:
            slider_active = False
            return
        slider_active = False
        seek_to_frame(progress_var.get(), slider_resume_playback)

    progress_scale.bind("<ButtonPress-1>", on_slider_press)
    progress_scale.bind("<ButtonRelease-1>", on_slider_release)

    def play_current():
        nonlocal cap, fps, playing, frame_counter, current_max_frames
        nonlocal current_duration_seconds, slider_active, slider_resume_playback
        if cap:
            cap.release()
        item = items[idx]
        vp = item['video_path']
        video_file = item['video_file']

        if video_file in processed_videos:
            advance_to_next_video()
            return

        prior_rows = existing_by_video.get(video_file)
        if prior_rows:
            reuse = messagebox.askyesno(
                "Reuse prior annotation?",
                (
                    f"An entry for '{video_file}' already exists in {args.output}.\n"
                    "Do you want to reuse the previous annotation instead of rescoring?"
                ),
            )
            if reuse:
                processed_videos.add(video_file)
                completed_rows.append(dict(prior_rows[-1]))
                messagebox.showinfo(
                    "Annotation Reused",
                    f"Using previously saved annotations for '{video_file}'.",
                )
                advance_to_next_video()
                return

        submit_btn.config(state=tk.NORMAL)
        next_btn.config(state=tk.DISABLED)
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video: {vp}")
            return
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_val) if fps_val and fps_val > 0 else item['fps']
        if not fps or fps <= 0:
            fps = item['fps'] if item['fps'] > 0 else 40.0
        frame_counter = 0
        current_max_frames = item['max_frames']
        current_duration_seconds = item.get('display_duration', 0.0) or 0.0
        if current_max_frames and fps > 0:
            current_duration_seconds = max(current_duration_seconds, current_max_frames / fps)
        slider_max = 1.0
        if current_max_frames:
            slider_max = max(1.0, float(current_max_frames - 1))
        elif current_duration_seconds and fps > 0:
            slider_max = max(1.0, current_duration_seconds * fps)
        progress_scale.configure(to=slider_max)
        set_progress(0.0)
        for var in score_vars.values():
            var.set(-1)
        data_text.configure(state="normal")
        data_text.delete("1.0", tk.END)
        data_text.configure(state="disabled")
        info.config(text=(
            f"{os.path.basename(vp)}  [{idx+1}/{len(items)}] — "
            f"windows include 30 s baseline + 2x{odor_latency_seconds:.1f}s latency. "
            f"Rate each active interval (0–5). Showing first {item['display_duration']:.1f}s."
        ))
        playing = True
        root.after(0, advance)

    def advance():
        nonlocal playing, cap, fps, frame_counter, current_max_frames
        if not playing or cap is None:
            return
        if current_max_frames and frame_counter >= current_max_frames:
            playing = False
            return
        ok, frame = cap.read()
        if not ok:
            playing = False
            return
        frame_counter += 1
        draw_frame(frame)
        set_progress(max(0, frame_counter - 1))
        if current_max_frames and frame_counter >= current_max_frames:
            playing = False
            return
        delay = int(1000/max(1.0,fps))
        root.after(delay, advance)

    def on_replay():
        nonlocal cap, playing, frame_counter
        if not cap: return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_counter = 0
        set_progress(0.0)
        if not playing:
            playing = True
            root.after(0, advance)

    def on_submit():
        nonlocal playing
        missing = [seg.label for seg in rateable_segments if score_vars[seg.key].get() == -1]
        if missing:
            messagebox.showwarning("Select scores", f"Please score each interval: {', '.join(missing)}")
            return
        playing = False
        it = items[idx]
        fly_id, trial_id = it.get('fly_id'), it.get('trial_id')
        threshold_value = float(it.get('threshold', 0.0))
        latency_value = float(it.get('odor_latency_seconds', odor_latency_seconds))

        segment_results = {}
        lines = [
            f"Threshold (μ_baseline + 4σ_baseline): {threshold_value:.3f}",
            f"Odor latency applied: {latency_value:.2f} s ({args.odor_latency})",
        ]
        for seg in segment_defs:
            metrics = it['segments'].get(seg.key)
            user_score = int(score_vars[seg.key].get()) if seg.rateable else None
            duration = metrics['duration'] if metrics else 0.0
            time_fraction = metrics['time_fraction'] if metrics else 0.0
            auc_val = metrics['auc'] if metrics else 0.0
            time_to_threshold = metrics.get('time_to_threshold') if metrics else None
            crossed_threshold = metrics.get('crossed_threshold', False) if metrics else False
            time_to_peak = metrics.get('time_to_peak') if metrics else None
            peak_value = metrics.get('peak_value') if metrics else None
            rise_speed = metrics.get('rise_speed') if metrics else None
            rise_acceleration = metrics.get('rise_acceleration') if metrics else None

            # Scale metrics to 0–5
            if metrics:
                m_parts = {
                    'time_fraction': max(0.0, min(5.0, time_fraction * 5.0)),
                    'auc': max(0.0, min(5.0, (auc_val / global_max_auc[seg.key] * 5.0) if global_max_auc[seg.key] > 0 else 0.0)),
                }
                if 'time_to_threshold' in METRIC_WEIGHTS:
                    if duration > 0 and time_to_threshold is not None:
                        clamped_time = max(0.0, min(time_to_threshold, duration))
                        response_score = 5.0 * (1.0 - (clamped_time / duration))
                    else:
                        response_score = 0.0
                    m_parts['time_to_threshold'] = max(0.0, min(5.0, response_score))
            else:
                m_parts = {k: 0.0 for k in METRIC_WEIGHTS}

            wsum = 0.0
            score_sum = 0.0
            for k, w in METRIC_WEIGHTS.items():
                if k in m_parts:
                    score_sum += float(w) * float(m_parts[k])
                    wsum += float(w)
            data_score = int(round(score_sum / wsum)) if wsum > 0 else 0

            # Adjust combined score weighting: user (75%) and data (25%)
            combined = (
                (3 * user_score + data_score) / 4.0
                if seg.rateable and user_score is not None
                else float(data_score)
            )

            time_above = time_fraction * duration
            pct = time_fraction * 100.0
            if metrics:
                label = seg.label
                if not seg.rateable:
                    label += " (not rated)"
                entry_lines = [
                    f"{label}:",
                    f"  Time above threshold: {time_above:.2f}s ({pct:.1f}%)",
                    f"  AUC over threshold: {auc_val:.3f}",
                ]
                if time_to_threshold is not None:
                    entry_lines.append(f"  Time to threshold: {time_to_threshold:.2f}s")
                else:
                    entry_lines.append("  Time to threshold: not reached")
                if time_to_peak is not None:
                    entry_lines.append(f"  Time to peak: {time_to_peak:.2f}s")
                if peak_value is not None:
                    entry_lines.append(f"  Peak value: {peak_value:.3f}")
                if rise_speed is not None:
                    entry_lines.append(f"  Rise speed: {rise_speed:.3f}/s")
                if rise_acceleration is not None:
                    entry_lines.append(f"  Rise acceleration: {rise_acceleration:.3f}/s²")
                entry_lines.append(f"  Data-suggested score: {data_score}")
                if seg.rateable:
                    entry_lines.append(f"  Your score: {user_score}")
                    entry_lines.append(f"  Combined score: {combined:.1f}")
                lines.append("\n".join(entry_lines))
            else:
                label = seg.label + (" (not rated)" if not seg.rateable else "")
                lines.append(f"{label}: No data available for this interval.")

            segment_results[seg.key] = {
                'user_score': user_score,
                'data_score': data_score,
                'combined_score': combined,
                'time_fraction': time_fraction,
                'auc': auc_val,
                'duration': duration,
                'time_above_threshold': time_above,
                'time_to_threshold': time_to_threshold,
                'crossed_threshold': crossed_threshold,
                'time_to_peak': time_to_peak,
                'peak_value': peak_value,
                'rise_speed': rise_speed,
                'rise_acceleration': rise_acceleration,
            }

        row = {
            "fly_id": fly_id,
            "trial_id": trial_id,
            "video_file": os.path.basename(it['video_path']),
            "csv_file": os.path.basename(it['csv_path']) if it['csv_path'] else "",
            "matrix_row_index": it.get("matrix_row_index", -1),
            "display_duration_seconds": it['display_duration'],
            "odor_latency_seconds": latency_value,
            "threshold": threshold_value,
        }

        for seg_key, seg_res in segment_results.items():
            user_entry = seg_res['user_score'] if seg_res['user_score'] is not None else None
            row[f"user_score_{seg_key}"] = user_entry
            row[f"data_score_{seg_key}"] = seg_res['data_score']
            row[f"combined_score_{seg_key}"] = seg_res['combined_score']
            row[f"time_fraction_{seg_key}"] = seg_res['time_fraction']
            row[f"auc_{seg_key}"] = seg_res['auc']
            row[f"time_above_threshold_{seg_key}"] = seg_res['time_above_threshold']
            row[f"segment_duration_{seg_key}"] = seg_res['duration']
            row[f"time_to_threshold_{seg_key}"] = seg_res.get('time_to_threshold')
            row[f"crossed_threshold_{seg_key}"] = seg_res.get('crossed_threshold')
            row[f"time_to_peak_{seg_key}"] = seg_res.get('time_to_peak')
            row[f"peak_value_{seg_key}"] = seg_res.get('peak_value')
            row[f"rise_speed_{seg_key}"] = seg_res.get('rise_speed')
            row[f"rise_acceleration_{seg_key}"] = seg_res.get('rise_acceleration')

        completed_rows.append(row)
        processed_videos.add(row['video_file'])

        data_text.configure(state="normal")
        data_text.delete("1.0", tk.END)
        data_text.insert("1.0", "\n\n".join(lines))
        data_text.configure(state="disabled")
        data_text.yview_moveto(0.0)

        submit_btn.config(state=tk.DISABLED)
        next_btn.config(state=tk.NORMAL)

    def advance_to_next_video():
        nonlocal idx, cap, playing
        playing = False
        if cap:
            cap.release()
            cap = None
        idx += 1
        if idx >= len(items):
            finalize_and_close(
                message=f"All videos scored.\nSaved: {args.output}",
                show_dialog=True,
            )
            return
        submit_btn.config(state=tk.NORMAL)
        next_btn.config(state=tk.DISABLED)
        play_current()

    def on_next():
        advance_to_next_video()

    def on_exit():
        if messagebox.askyesno(
            "Save and Exit",
            "Save current progress to the output CSV and exit the scorer?",
        ):
            finalize_and_close(
                message=f"Progress saved to {args.output}.",
                show_dialog=True,
            )

    submit_btn.config(command=on_submit)
    next_btn.config(command=on_next)
    replay_btn.config(command=on_replay)
    exit_btn.config(command=on_exit)

    root.protocol("WM_DELETE_WINDOW", on_exit)

    play_current()
    root.mainloop()

if __name__ == "__main__":
    main()

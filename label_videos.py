import os
import sys
import argparse
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import re

# =================== CONFIG (edit as needed) ===================
METRIC_WEIGHTS = {
    # Increase/decrease to change influence on data_score (0–10 scale per metric)
    'time_fraction': 1.0,   # fraction of frames above threshold
    'auc': 1.0              # integral above threshold (scaled by global max)
}

SEGMENTS = {
    'odor': {
        'label': 'During odor',
        'duration_seconds': 60.0
    },
    'post': {
        'label': '30s after odor',
        'duration_seconds': 30.0
    }
}

TOTAL_DISPLAY_SECONDS = sum(seg['duration_seconds'] for seg in SEGMENTS.values())
# Threshold computation parameters (match envelope script behaviour)
ODOR_ON_SECONDS = 30.0
DEFAULT_SMOOTHING_FPS = 40.0
ENVELOPE_WINDOW_SECONDS = 0.25
# ===============================================================

def _analytic_signal(values):
    """Return analytic signal via FFT-based Hilbert transform (SciPy-free)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return arr.astype(complex)
    spectrum = np.fft.fft(arr, n)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * h)

def compute_envelope(signal, fps):
    """Compute clipped analytic envelope smoothed over ENVELOPE_WINDOW_SECONDS."""
    arr = np.asarray(signal, dtype=float)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 100.0)
    analytic = _analytic_signal(arr)
    env = np.abs(analytic)
    fps_for_window = fps if fps and fps > 0 else DEFAULT_SMOOTHING_FPS
    window_frames = max(int(round(ENVELOPE_WINDOW_SECONDS * fps_for_window)), 1)
    if window_frames > 1:
        env = (
            pd.Series(env)
            .rolling(window=window_frames, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )
    return env

def compute_threshold(time_s, envelope, fps):
    """Compute μ_before + 4σ_before using pre-odor window (< ODOR_ON_SECONDS)."""
    env = np.asarray(envelope, dtype=float)
    if env.size == 0:
        return 0.0
    times = np.asarray(time_s, dtype=float)
    if times.size != env.size:
        times = np.arange(env.size, dtype=float) / (fps if fps and fps > 0 else DEFAULT_SMOOTHING_FPS)
    pre_mask = times < ODOR_ON_SECONDS
    if np.any(pre_mask):
        pre_vals = env[pre_mask]
    else:
        n_pre = max(int(round(ODOR_ON_SECONDS * (fps if fps and fps > 0 else DEFAULT_SMOOTHING_FPS))), 1)
        pre_vals = env[:n_pre]
    if pre_vals.size == 0:
        return 0.0
    mu = float(np.nanmean(pre_vals))
    sigma = float(np.nanstd(pre_vals, ddof=0))
    return mu + 4.0 * sigma

def compute_metrics(signal, fps, threshold):
    """
    Compute core reaction metrics from a 1D signal.
    Returns dict with: time_fraction, auc, duration.
    """
    signal = np.asarray(signal).astype(float)
    n = len(signal)
    if n == 0:
        return None
    duration = n / fps if fps and fps > 0 else float(n)

    thresh = float(threshold) if threshold is not None else 0.0
    above = signal > thresh
    time_fraction = float(np.count_nonzero(above)) / float(n)

    if above.any():
        auc = float(np.sum(signal[above] - thresh)) / (fps if fps and fps > 0 else 1.0)
    else:
        auc = 0.0

    return {'time_fraction': time_fraction, 'auc': auc, 'duration': duration}

def choose_signal_column(df):
    """Pick a likely signal column from a dataframe."""
    preferred = ['RMS', 'rms', 'envelope', 'Envelope', 'distance', 'Eye_Prob_Dist']
    for p in preferred:
        if p in df.columns:
            return df[p].values
    # Otherwise, pick the first non-time column
    candidates = [c for c in df.columns if c.lower() not in ('time','timestamp','frame','frames','t','ms','seconds')]
    if candidates:
        return df[candidates[0]].values
    # Fallback: last column
    return df.iloc[:, -1].values

def extract_time_seconds(df, fps):
    """Derive a time axis in seconds from dataframe columns or FPS."""
    time_like = []
    for col in df.columns:
        lower = col.lower()
        if any(token in lower for token in ('time', 'second', 'timestamp')):
            series = pd.to_numeric(df[col], errors='coerce')
            if series.notna().sum() > 0:
                time_like.append(series.to_numpy(dtype=float))
    if time_like:
        # Prefer the first viable column
        return time_like[0]
    n = len(df)
    fps_val = fps if fps and fps > 0 else DEFAULT_SMOOTHING_FPS
    if fps_val and fps_val > 0:
        return np.arange(n, dtype=float) / float(fps_val)
    return np.arange(n, dtype=float)

def parse_fly_trial(path):
    """Try to parse fly_id and trial_id from path/filename digits."""
    fname = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))

    fly_id = None
    trial_id = None

    # Prefer parent dir if like 'fly2736'
    if parent.lower().startswith('fly'):
        digits = ''.join(filter(str.isdigit, parent))
        if digits:
            fly_id = digits
    elif parent.isdigit():
        fly_id = parent

    nums = re.findall(r'\d+', fname)
    if nums:
        if len(nums) >= 2:
            fly_id = fly_id or nums[0]
            trial_id = nums[1]
        else:
            if fly_id is None:
                fly_id = nums[0]
            else:
                trial_id = nums[0]
    return fly_id, trial_id

def main():
    ap = argparse.ArgumentParser(description="Label fly behavior videos (0–10) and compute data-driven scores.")
    ap.add_argument("-v","--videos", required=True, help="Folder with videos (recursively scanned).")
    ap.add_argument("-d","--data", default=None, help="Folder with CSV traces (if not next to videos).")
    ap.add_argument("-o","--output", default="scoring_results.csv", help="Output master CSV.")
    args = ap.parse_args()

    # Find videos
    video_exts = (".mp4",".avi",".mov",".mpg",".mpeg",".wmv",".mkv")
    videos = []
    for root,_,files in os.walk(args.videos):
        for f in files:
            fname_lower = f.lower()
            if fname_lower.endswith(video_exts) and 'testing' in fname_lower:
                videos.append(os.path.join(root,f))
    videos.sort()
    if not videos:
        print("No videos found.")
        sys.exit(1)

    # Map base name -> CSV path
    data_map = {}
    if args.data:
        for root,_,files in os.walk(args.data):
            for f in files:
                fname_lower = f.lower()
                if fname_lower.endswith(".csv") and 'testing' in fname_lower:
                    data_map[os.path.splitext(f)[0]] = os.path.join(root,f)

    def csv_for(video_path):
        base = os.path.splitext(os.path.basename(video_path))[0]
        if 'testing' not in base.lower():
            return None
        if base in data_map:
            return data_map[base]
        candidate = os.path.join(os.path.dirname(video_path), base + ".csv")
        return candidate if os.path.exists(candidate) else None

    # Precompute metrics + global max AUC for scaling
    items = []
    global_max_auc = {key: 0.0 for key in SEGMENTS}
    for vp in videos:
        cp = csv_for(vp)
        if cp is None or not os.path.exists(cp):
            print(f"[WARN] No CSV for {vp}, skipping.")
            continue
        try:
            df = pd.read_csv(cp)
        except Exception as e:
            print(f"[WARN] Failed reading {cp}: {e}")
            continue

        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"[WARN] Cannot open video {vp}, skipping.")
            cap.release()
            continue
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_val) if fps_val and fps_val > 0 else 30.0
        frame_count_val = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_count = int(frame_count_val) if frame_count_val and frame_count_val > 0 else None
        cap.release()

        sig_raw = choose_signal_column(df)
        sig_series = pd.to_numeric(pd.Series(sig_raw), errors='coerce').fillna(0.0).to_numpy(dtype=float)
        fps_for_calc = fps if fps and fps > 0 else 30.0
        time_axis = extract_time_seconds(df, fps_for_calc)
        envelope = compute_envelope(sig_series, fps_for_calc)
        threshold = compute_threshold(time_axis, envelope, fps_for_calc)

        segment_metrics = {}
        signal_limit_frames = len(envelope)
        frames_total_display = int(round(fps_for_calc * TOTAL_DISPLAY_SECONDS)) if fps_for_calc > 0 else 0
        if frames_total_display > 0:
            signal_limit_frames = min(signal_limit_frames, frames_total_display)

        start_idx = 0
        segment_items = list(SEGMENTS.items())
        for seg_idx, (seg_key, seg_cfg) in enumerate(segment_items):
            seg_frames = int(round(fps_for_calc * seg_cfg['duration_seconds'])) if fps_for_calc > 0 else 0
            if seg_idx == len(segment_items) - 1 and signal_limit_frames > start_idx:
                end_idx = signal_limit_frames
            else:
                end_idx = start_idx + seg_frames if seg_frames > 0 else start_idx
                if signal_limit_frames >= 0:
                    end_idx = min(signal_limit_frames, end_idx)
            if end_idx < start_idx:
                end_idx = start_idx
            seg_signal = envelope[start_idx:end_idx]
            start_idx = end_idx
            metrics = compute_metrics(seg_signal, fps_for_calc, threshold) if len(seg_signal) else None
            if metrics:
                global_max_auc[seg_key] = max(global_max_auc[seg_key], metrics['auc'])
            segment_metrics[seg_key] = metrics

        max_frames = int(round(fps_for_calc * TOTAL_DISPLAY_SECONDS)) if fps_for_calc > 0 else 0
        if frame_count is not None:
            max_frames = min(max_frames, frame_count) if max_frames > 0 else frame_count
        if max_frames <= 0:
            max_frames = frames_total_display if frames_total_display > 0 else signal_limit_frames
        if max_frames <= 0:
            print(f"[WARN] No playable frames for {vp}, skipping.")
            continue

        item = {
            'video_path': vp,
            'csv_path': cp,
            'fly_id': None,
            'trial_id': None,
            'segments': segment_metrics,
            'threshold': threshold,
            'fps': fps_for_calc,
            'max_frames': max_frames,
            'display_duration': min(TOTAL_DISPLAY_SECONDS, max_frames / fps_for_calc if fps_for_calc > 0 else TOTAL_DISPLAY_SECONDS)
        }
        item['fly_id'], item['trial_id'] = parse_fly_trial(vp)
        items.append(item)

    if not items:
        print("Nothing to score (no metric-bearing pairs).")
        sys.exit(1)

    # ---------- GUI ----------
    root = tk.Tk()
    root.title("Fly Behavior Scoring")

    # Compute max canvas size
    max_w, max_h = 640, 480
    for it in items:
        cap = cv2.VideoCapture(it['video_path'])
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
            cap.release()
            max_w = max(max_w, w)
            max_h = max(max_h, h)

    canvas = tk.Canvas(root, width=max_w, height=max_h, bg="black")
    canvas.pack()

    info = tk.Label(root, text="Watch the first 90 seconds. Score each interval 0 (NR) … 10 (Strongest). Submit to reveal data metrics.")
    info.pack(pady=6)

    # Rating row
    rating_frame = tk.Frame(root)
    rating_frame.pack(pady=4, fill="x")
    score_vars = {}
    labels = [("0 (NR)",0),("1",1),("2",2),("3",3),("4",4),("5 (Average)",5),("6",6),("7",7),("8",8),("9",9),("10 (Strongest)",10)]
    for seg_key, seg_cfg in SEGMENTS.items():
        seg_frame = tk.LabelFrame(rating_frame, text=seg_cfg['label'])
        seg_frame.pack(fill="x", pady=2)
        var = tk.IntVar(value=-1)
        score_vars[seg_key] = var
        for text,val in labels:
            tk.Radiobutton(seg_frame, text=text, variable=var, value=val).pack(side=tk.LEFT)

    # Buttons
    btns = tk.Frame(root); btns.pack(pady=4)
    submit_btn = tk.Button(btns, text="Submit Score")
    next_btn   = tk.Button(btns, text="Next Video", state=tk.DISABLED)
    replay_btn = tk.Button(btns, text="Replay Video")
    submit_btn.grid(row=0,column=0,padx=4)
    next_btn.grid(row=0,column=1,padx=4)
    replay_btn.grid(row=0,column=2,padx=4)

    # Data panel (revealed post-submit)
    data_lbl = tk.Label(root, text="", fg="blue", justify=tk.LEFT, anchor="w")
    data_lbl.pack(pady=6, fill="x")

    # State
    idx = 0
    results = []
    cap = None
    fps = 30.0
    playing = False
    frame_counter = 0
    current_max_frames = 0

    def play(index):
        nonlocal cap, fps, playing, frame_counter, current_max_frames
        if cap: cap.release()
        item = items[index]
        vp = item['video_path']
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video: {vp}")
            return
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_val) if fps_val and fps_val > 0 else item['fps']
        if not fps or fps <= 0:
            fps = item['fps'] if item['fps'] > 0 else 30.0
        frame_counter = 0
        current_max_frames = item['max_frames']
        for var in score_vars.values():
            var.set(-1)
        data_lbl.config(text="")
        info.config(text=(
            f"{os.path.basename(vp)}  [{index+1}/{len(items)}] — "
            f"rate each interval (0–10). Showing first {item['display_duration']:.1f}s."
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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=im)
        canvas.imgtk = imgtk
        canvas.create_image(0,0,anchor=tk.NW,image=imgtk)
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
        if not playing:
            playing = True
            root.after(0, advance)

    def on_submit():
        nonlocal playing
        missing = [seg_cfg['label'] for seg_key, seg_cfg in SEGMENTS.items() if score_vars[seg_key].get() == -1]
        if missing:
            messagebox.showwarning("Select scores", f"Please score each interval: {', '.join(missing)}")
            return
        playing = False
        it = items[idx]
        fly_id, trial_id = it.get('fly_id'), it.get('trial_id')
        threshold_value = float(it.get('threshold', 0.0))

        segment_results = {}
        lines = [f"Threshold (μ_before + 4σ_before): {threshold_value:.3f}"]
        for seg_key, seg_cfg in SEGMENTS.items():
            metrics = it['segments'].get(seg_key)
            user_score = int(score_vars[seg_key].get())
            duration = metrics['duration'] if metrics else 0.0
            time_fraction = metrics['time_fraction'] if metrics else 0.0
            auc_val = metrics['auc'] if metrics else 0.0

            # Scale metrics to 0–10
            if metrics:
                m_parts = {
                    'time_fraction': max(0.0, min(10.0, time_fraction * 10.0)),
                    'auc': max(0.0, min(10.0, (auc_val / global_max_auc[seg_key] * 10.0) if global_max_auc[seg_key] > 0 else 0.0))
                }
            else:
                m_parts = {'time_fraction': 0.0, 'auc': 0.0}

            wsum = 0.0
            score_sum = 0.0
            for k, w in METRIC_WEIGHTS.items():
                if k in m_parts:
                    score_sum += float(w) * float(m_parts[k])
                    wsum += float(w)
            data_score = int(round(score_sum / wsum)) if wsum > 0 else 0
            combined = (user_score + data_score) / 2.0

            time_above = time_fraction * duration
            pct = time_fraction * 100.0
            if metrics:
                lines.append(
                    f"{seg_cfg['label']}:\n"
                    f"  Time above threshold: {time_above:.2f}s ({pct:.1f}%)\n"
                    f"  AUC over threshold: {auc_val:.3f}\n"
                    f"  Data-suggested score: {data_score}"
                )
            else:
                lines.append(f"{seg_cfg['label']}: No data available for this interval.")

            segment_results[seg_key] = {
                'user_score': user_score,
                'data_score': data_score,
                'combined_score': combined,
                'time_fraction': time_fraction,
                'auc': auc_val,
                'duration': duration,
                'time_above_threshold': time_above
            }

        row = {
            "fly_id": fly_id,
            "trial_id": trial_id,
            "video_file": os.path.basename(it['video_path']),
            "csv_file": os.path.basename(it['csv_path']),
            "display_duration_seconds": it['display_duration']
        }
        row["threshold"] = threshold_value
        for seg_key, seg_res in segment_results.items():
            row[f"user_score_{seg_key}"] = seg_res['user_score']
            row[f"data_score_{seg_key}"] = seg_res['data_score']
            row[f"combined_score_{seg_key}"] = seg_res['combined_score']
            row[f"time_fraction_{seg_key}"] = seg_res['time_fraction']
            row[f"auc_{seg_key}"] = seg_res['auc']
            row[f"time_above_threshold_{seg_key}"] = seg_res['time_above_threshold']
            row[f"segment_duration_{seg_key}"] = seg_res['duration']

        results.append(row)

        data_lbl.config(text="\n\n".join(lines))

        submit_btn.config(state=tk.DISABLED)
        next_btn.config(state=tk.NORMAL)

    def on_next():
        nonlocal idx, cap
        idx += 1
        if idx >= len(items):
            out = args.output
            pd.DataFrame(results).to_csv(out, index=False)
            if cap:
                cap.release()
            messagebox.showinfo("Done", f"All videos scored.\nSaved: {out}")
            root.destroy()
            return
        next_btn.config(state=tk.DISABLED)
        submit_btn.config(state=tk.NORMAL)
        play(idx)

    submit_btn.config(command=on_submit)
    next_btn.config(command=on_next)
    replay_btn.config(command=on_replay)

    play(idx)
    root.mainloop()

if __name__ == "__main__":
    main()

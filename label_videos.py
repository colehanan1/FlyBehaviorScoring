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
THRESHOLD = 1.0  # Signal threshold defining 'reaction'
METRIC_WEIGHTS = {
    # Increase/decrease to change influence on data_score (0–10 scale per metric)
    'time_fraction': 1.0,   # fraction of frames above THRESHOLD
    'auc': 1.0              # integral above THRESHOLD (scaled by global max)
}
# ===============================================================

def compute_metrics(signal, fps):
    """
    Compute core reaction metrics from a 1D signal.
    Returns dict with: time_fraction, auc, duration.
    """
    signal = np.asarray(signal).astype(float)
    n = len(signal)
    if n == 0:
        return None
    duration = n / fps if fps and fps > 0 else float(n)

    above = signal > THRESHOLD
    time_fraction = float(np.count_nonzero(above)) / float(n)

    if above.any():
        auc = float(np.sum(signal[above] - THRESHOLD)) / (fps if fps and fps > 0 else 1.0)
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
            if f.lower().endswith(video_exts):
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
                if f.lower().endswith(".csv"):
                    data_map[os.path.splitext(f)[0]] = os.path.join(root,f)

    def csv_for(video_path):
        base = os.path.splitext(os.path.basename(video_path))[0]
        if base in data_map:
            return data_map[base]
        candidate = os.path.join(os.path.dirname(video_path), base + ".csv")
        return candidate if os.path.exists(candidate) else None

    # Precompute metrics + global max AUC for scaling
    items = []
    global_max_auc = 0.0
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

        # FPS from video (fallback 30)
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        sig = choose_signal_column(df)
        m = compute_metrics(sig, fps)
        if m is None:
            print(f"[WARN] No metrics for {vp}, skipping.")
            continue

        m['video_path'] = vp
        m['csv_path'] = cp
        m['fly_id'], m['trial_id'] = parse_fly_trial(vp)
        items.append(m)
        global_max_auc = max(global_max_auc, m['auc'])

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

    info = tk.Label(root, text="Watch the video. Select 0 (NR) … 10 (Strongest). Submit to reveal data metrics.")
    info.pack(pady=6)

    # Rating row
    rating_frame = tk.Frame(root)
    rating_frame.pack(pady=4)
    score_var = tk.IntVar(value=-1)
    labels = [("0 (NR)",0),("1",1),("2",2),("3",3),("4",4),("5 (Average)",5),("6",6),("7",7),("8",8),("9",9),("10 (Strongest)",10)]
    for text,val in labels:
        tk.Radiobutton(rating_frame, text=text, variable=score_var, value=val).pack(side=tk.LEFT)

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

    def play(index):
        nonlocal cap, fps, playing
        if cap: cap.release()
        vp = items[index]['video_path']
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video: {vp}")
            return
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_val) if fps_val and fps_val > 0 else 30.0
        score_var.set(-1)
        data_lbl.config(text="")
        info.config(text=f"{os.path.basename(vp)}  [{index+1}/{len(items)}] — rate 0 (NR) to 10 (Strongest)")
        playing = True
        root.after(0, advance)

    def advance():
        nonlocal playing, cap, fps
        if not playing or cap is None:
            return
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=im)
            canvas.imgtk = imgtk
            canvas.create_image(0,0,anchor=tk.NW,image=imgtk)
            root.after(int(1000/max(1.0,fps)), advance)
        else:
            playing = False

    def on_replay():
        nonlocal cap, playing
        if not cap: return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not playing:
            playing = True
            root.after(0, advance)

    def on_submit():
        nonlocal playing
        if score_var.get() == -1:
            messagebox.showwarning("Select a score", "Please select 0–10 before submitting.")
            return
        playing = False
        it = items[idx]
        time_frac = it['time_fraction']
        auc_val   = it['auc']

        # Scale metrics to 0–10
        m_parts = {}
        m_parts['time_fraction'] = max(0.0, min(10.0, time_frac * 10.0))
        m_parts['auc'] = max(0.0, min(10.0, (auc_val / global_max_auc * 10.0) if global_max_auc > 0 else 0.0))

        # Weighted average
        wsum = 0.0; score_sum = 0.0
        for k,w in METRIC_WEIGHTS.items():
            if k in m_parts:
                score_sum += float(w) * float(m_parts[k])
                wsum += float(w)
        data_score = int(round(score_sum / wsum)) if wsum > 0 else 0

        user_score = int(score_var.get())
        combined = (user_score + data_score) / 2.0

        fly_id, trial_id = it.get('fly_id'), it.get('trial_id')

        results.append({
            "fly_id": fly_id,
            "trial_id": trial_id,
            "video_file": os.path.basename(it['video_path']),
            "csv_file": os.path.basename(it['csv_path']),
            "user_score": user_score,
            "data_score": data_score,
            "combined_score": combined,
            "time_fraction": time_frac,
            "auc": auc_val,
            "duration": it['duration']
        })

        # Reveal metrics
        secs = time_frac * it['duration']
        pct = time_frac * 100.0
        data_lbl.config(text=(
            f"Time above threshold: {secs:.2f}s ({pct:.1f}%)\n"
            f"AUC over threshold: {auc_val:.3f}\n"
            f"Data-suggested score: {data_score}"
        ))

        submit_btn.config(state=tk.DISABLED)
        next_btn.config(state=tk.NORMAL)

    def on_next():
        nonlocal idx
        idx += 1
        if idx >= len(items):
            out = args.output
            pd.DataFrame(results).to_csv(out, index=False)
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

"""Regenerate every signal-analysis and model-feature figure for the thesis supplement.

Two figure families, both computed on the *same* rows the combined multiclass
scorer is trained on (notebook 05's two blinded-scoring sources):

  S1–S8   Signal characterisation — what the raw envelope traces look like in
          time and in frequency, split by human score.  These are the evidence
          that the spectral features are worth extracting at all.
  M1–M6   Model features — how the 24 features behave across the score scale,
          how redundant they are, which ones the model actually uses, and what
          the trained model gets right and wrong.

Every figure is written as vector PDF (for the thesis) and 400 dpi PNG, plus a
``captions.md`` giving each figure's caption and the reason it is in the
supplement.

Usage::

    python scripts/figures/make_thesis_figures.py            # use cached data
    python scripts/figures/make_thesis_figures.py --rebuild  # re-read source CSVs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: write files, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.lines import Line2D
from scipy import signal as sps
from scipy.stats import kruskal
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prep_data as prep  # noqa: E402
from flybehavior_response.ordinal_cost import (  # noqa: E402
    N_CLASSES, build_penalty_matrix, make_objective)
import vizstyle as vz  # noqa: E402

OUT = prep.PROJECT_ROOT / "outputs" / "thesis_figures"
FPS = prep.FPS
ODOR_ON, ODOR_OFF = prep.ODOR_ON_SEC, prep.ODOR_OFF_SEC
N_FIG_FRAMES = 3600           # 90 s — the window every recording covers
SCORE_LEVELS = prep.SCORE_LEVELS
SEED = prep.SEED

BANDS = prep.BANDS
BAND_LABELS = prep.BAND_LABELS

FIG_W2 = 7.2      # two-column / full-page width, inches
FIG_W1 = 3.5      # single-column width, inches

CAPTIONS: list[tuple[str, str, str]] = []   # (stem, title, caption body)


def caption(stem: str, title: str, body: str) -> None:
    CAPTIONS.append((stem, title, " ".join(body.split())))


# =============================================================================
# Shared computation
# =============================================================================
def compute_spectral(traces: np.ndarray) -> dict:
    """Welch PSDs (whole trial and per epoch), spectrograms and band powers."""
    print("Computing spectral summaries…")
    centred = traces - traces.mean(axis=1, keepdims=True)

    f_full, psd_full = sps.welch(centred, fs=FPS, nperseg=512, noverlap=256, axis=1)

    on, off = ODOR_ON * FPS, ODOR_OFF * FPS
    epochs = {"Before (0–30 s)": traces[:, :on],
              "During (30–60 s)": traces[:, on:off],
              "After (60–90 s)": traces[:, off:]}
    psd_epoch, f_ep = {}, None
    for name, seg in epochs.items():
        f_ep, p = sps.welch(seg - seg.mean(axis=1, keepdims=True), fs=FPS,
                            nperseg=256, noverlap=128, axis=1)
        psd_epoch[name] = p

    f_spec, t_spec, sxx = sps.spectrogram(centred, fs=FPS, nperseg=256, noverlap=128, axis=1)
    sxx = np.moveaxis(sxx, 1, -1) if sxx.ndim == 3 and sxx.shape[1] != len(f_spec) else sxx

    band_pw = {bn: psd_full[:, (f_full >= lo) & (f_full < hi)].sum(axis=1)
               for bn, (lo, hi) in BANDS.items()}

    return {"f_full": f_full, "psd_full": psd_full, "f_ep": f_ep, "psd_epoch": psd_epoch,
            "f_spec": f_spec, "t_spec": t_spec, "sxx": sxx, "band_pw": band_pw}


# =============================================================================
# S1–S8 — signal characterisation
# =============================================================================
def fig_s1_traces(traces, yv, colors, levels):
    t = np.arange(traces.shape[1]) / FPS
    fig, axes = plt.subplots(2, 4, figsize=(FIG_W2, 3.6), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, lvl in zip(axes, levels):
        m = yv == lvl
        seg = traces[m]
        med = np.median(seg, axis=0)
        q1, q3 = np.percentile(seg, [25, 75], axis=0)
        vz.mark_odor_window(ax, ODOR_ON, ODOR_OFF)
        ax.fill_between(t, q1, q3, color=colors[lvl], alpha=0.30, linewidth=0)
        ax.plot(t, med, color=colors[lvl], linewidth=1.0)
        ax.set_title(f"score {lvl}", fontsize=8)
        ax.text(0.97, 0.93, f"n = {m.sum()}", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.5, color=vz.MUTED)
        vz.tidy(ax)

    ax = axes[7]
    vz.mark_odor_window(ax, ODOR_ON, ODOR_OFF)
    for lvl in levels:
        ax.plot(t, np.median(traces[yv == lvl], axis=0), color=colors[lvl], linewidth=1.0)
    ax.set_title("all medians", fontsize=8)
    vz.tidy(ax)

    for ax in axes[4:]:
        ax.set_xlabel("Time (s)")
    for ax in axes[::4]:
        ax.set_ylabel("Envelope (a.u.)")
    axes[0].set_xlim(0, 90)

    handles = [Line2D([], [], color=colors[l], lw=1.6, label=f"{l}") for l in levels]
    fig.legend(handles=handles, title="Human score", loc="upper center", ncol=7,
               bbox_to_anchor=(0.5, 1.10), title_fontsize=7)
    fig.tight_layout()
    vz.save(fig, OUT, "S1_trace_overview_by_score")
    plt.close(fig)

    caption("S1_trace_overview_by_score", "Envelope traces by human score",
            """Median (line) and interquartile range (shaded) of the proboscis-extension
            envelope for each human score, over the 90 s window every recording covers;
            the grey band marks the 30–60 s odor presentation. The final panel overlays
            all seven medians on one axis. **Why it is here:** it establishes the basic
            premise of the feature set. From score 0 upward the amplitude of the
            odor-evoked deflection grows with score, and score −1 is the one level whose
            median *falls* during odor. But scores 0, 1 and 2 are barely separable by
            deflection height and the interquartile bands overlap heavily throughout, so a
            single amplitude threshold cannot recover the human scale — which is what
            motivates a multi-feature model.""")


def fig_s2_psd(sp, yv, colors, levels):
    fig, ax = plt.subplots(figsize=(FIG_W2, 3.0))
    f, psd = sp["f_full"], sp["psd_full"]
    keep = f <= 10
    for lvl in levels:
        m = yv == lvl
        ax.semilogy(f[keep], psd[m][:, keep].mean(axis=0), color=colors[lvl],
                    linewidth=1.2, label=f"{lvl}  (n = {m.sum()})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (a.u.²/Hz)")
    ax.set_xlim(0, 10)
    ax.set_title("Mean Welch power spectrum by human score")
    ax.legend(title="Human score", ncol=2, loc="upper right")
    vz.tidy(ax, grid_axis="both")
    fig.tight_layout()
    vz.save(fig, OUT, "S2_psd_by_score")
    plt.close(fig)

    caption("S2_psd_by_score", "Whole-trial power spectrum by score",
            """Mean Welch power spectral density (512-sample segments, 50% overlap,
            mean-removed) over the whole 90 s trial, averaged within each score group and
            plotted on a log power axis. **Why it is here:** the groups differ across the
            whole 0–10 Hz range rather than at one narrow peak — score 5 pulls away from
            the rest above roughly 2 Hz and stays separated out to 10 Hz. There is no
            single dominant frequency to key on, which is the justification for summarising
            the spectrum with five broad band-power features rather than a peak
            frequency.""")


def fig_s3_psd_ratio(sp, yv, colors, levels):
    f, psd = sp["f_full"], sp["psd_full"]
    keep = f <= 10
    ref = psd[yv == 0][:, keep].mean(axis=0)

    fig, ax = plt.subplots(figsize=(FIG_W2, 3.0))
    ax.axhline(0, color=vz.AXIS, linewidth=0.8, zorder=1)
    for lvl in levels:
        if lvl == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log2(psd[yv == lvl][:, keep].mean(axis=0) / ref)
        ax.plot(f[keep], ratio, color=colors[lvl], linewidth=1.2, label=f"{lvl}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log₂ (power ÷ score-0 power)")
    ax.set_xlim(0, 10)
    ax.set_title("Spectral power relative to non-responders (score 0)")
    ax.legend(title="Human score", ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), title_fontsize=7)
    vz.tidy(ax, grid_axis="both")
    fig.tight_layout()
    vz.save(fig, OUT, "S3_psd_ratio_to_score0")
    plt.close(fig)

    caption("S3_psd_ratio_to_score0", "Spectral power relative to non-responders",
            """Each score group's mean power spectrum divided by the score-0 (no response)
            mean spectrum, on a log₂ axis, so 0 means "same power as a non-responder" and
            +1 means "double". **Why it is here:** it isolates what changes with response
            strength rather than what is common to every trace. Almost every group has more
            power than score 0 at almost every frequency, and scores 3, 4 and 5 separate
            cleanly and in order. Scores −1, 1 and 2 instead sit together in a band around
            +1 log₂ unit and are *not* ordered — a first warning that the low end of the
            scale is where the model will struggle, borne out in M6. The figure also shows
            that the informative quantity is a power *ratio* rather than absolute power,
            which is exactly the form the five `power_ratio_*` features take.""")


def fig_s4_spectrograms(sp, yv, colors, levels):
    f, ts, sxx = sp["f_spec"], sp["t_spec"], sp["sxx"]
    fkeep = f <= 10
    mats = {lvl: np.log10(sxx[yv == lvl].mean(axis=0)[fkeep] + 1e-10) for lvl in levels}
    vmin = min(m.min() for m in mats.values())
    vmax = max(m.max() for m in mats.values())

    stack = np.concatenate([m.ravel() for m in mats.values()])
    vmin, vmax = np.percentile(stack, [2, 99.5])

    fig, axes = plt.subplots(2, 4, figsize=(FIG_W2, 3.4), sharex=True, sharey=True)
    axes = axes.ravel()
    im = None
    for ax, lvl in zip(axes, levels):
        im = ax.pcolormesh(ts, f[fkeep], mats[lvl], cmap=vz.SEQ_CMAP,
                           vmin=vmin, vmax=vmax, shading="gouraud", rasterized=True)
        for x in (ODOR_ON, ODOR_OFF):
            ax.axvline(x, color=vz.PRIMARY, linewidth=0.5, alpha=0.35)
        ax.set_title(f"score {lvl}", fontsize=8)
        ax.grid(False)
    axes[7].axis("off")
    for ax in axes[4:7]:
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[4].set_ylabel("Frequency (Hz)")
    axes[0].set_xlim(0, 90)
    axes[0].set_xticks([0, 30, 60, 90])

    fig.tight_layout()
    vz.colorbar(fig, im, axes[7], "log₁₀ power", fraction=0.14, pad=-0.55, aspect=12)
    vz.save(fig, OUT, "S4_spectrogram_by_score")
    plt.close(fig)

    caption("S4_spectrogram_by_score", "Time–frequency structure by score",
            """Group-average spectrograms (256-sample windows, 50% overlap), one panel per
            human score, on a shared log-power colour scale; the vertical rules bracket the
            odor presentation. **Why it is here:** it shows *when* the spectral change
            happens. Power rises at odor onset and decays after offset in the responding
            groups, which is what licenses splitting every trace into before / during /
            after epochs and comparing them, rather than computing one spectrum per
            trial.""")


def fig_s5_spectrogram_diff(sp, yv, levels):
    f, ts, sxx = sp["f_spec"], sp["t_spec"], sp["sxx"]
    fkeep = f <= 10
    ref = np.log10(sxx[yv == 0].mean(axis=0)[fkeep] + 1e-10)
    diff_levels = [l for l in levels if l != 0]
    diffs = {l: np.log10(sxx[yv == l].mean(axis=0)[fkeep] + 1e-10) - ref for l in diff_levels}
    vlim = min(max(np.abs(d).max() for d in diffs.values()), 1.5)

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W2, 3.4), sharex=True, sharey=True)
    axes = axes.ravel()
    im = None
    for ax, lvl in zip(axes, diff_levels):
        im = ax.pcolormesh(ts, f[fkeep], diffs[lvl], cmap=vz.DIV_CMAP,
                           vmin=-vlim, vmax=vlim, shading="gouraud", rasterized=True)
        for x in (ODOR_ON, ODOR_OFF):
            ax.axvline(x, color=vz.PRIMARY, linewidth=0.6, alpha=0.5)
        ax.set_title(f"score {lvl} − score 0", fontsize=8)
        ax.grid(False)
    for ax in axes[3:]:
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[3].set_ylabel("Frequency (Hz)")
    axes[0].set_xlim(0, 90)
    axes[0].set_xticks([0, 30, 60, 90])

    fig.tight_layout()
    vz.colorbar(fig, im, list(axes), "Δ log₁₀ power vs score 0", fraction=0.022, aspect=28)
    vz.save(fig, OUT, "S5_spectrogram_difference")
    plt.close(fig)

    caption("S5_spectrogram_difference", "Odor-evoked spectral change relative to score 0",
            """The same group-average spectrograms as S4, each with the score-0 spectrogram
            subtracted, on a diverging scale where neutral grey is "no difference from a
            non-responder". **Why it is here:** it localises the discriminative signal to
            the odor window and the seconds following it, and shows the effect scaling with
            score. Any feature that averages over the whole trial would dilute this;
            it is the direct argument for the during-vs-before ratio features
            (`auc_ratio`, `std_ratio`, `power_ratio_*`, `total_power_ratio`).""")


def fig_s6_epoch_psd(sp, yv, colors, levels):
    f = sp["f_ep"]
    keep = f <= 10
    names = list(sp["psd_epoch"].keys())

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W2, 2.6), sharey=True)
    for ax, name in zip(axes, names):
        psd = sp["psd_epoch"][name]
        for lvl in levels:
            ax.semilogy(f[keep], psd[yv == lvl][:, keep].mean(axis=0),
                        color=colors[lvl], linewidth=1.1, label=f"{lvl}")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_xlim(0, 10)
        vz.tidy(ax, grid_axis="both")
    axes[0].set_ylabel("Power spectral density")
    axes[2].legend(title="Human score", ncol=2, loc="upper right")
    fig.tight_layout()
    vz.save(fig, OUT, "S6_psd_by_epoch")
    plt.close(fig)

    caption("S6_psd_by_epoch", "Power spectrum per epoch",
            """Mean Welch spectra computed separately on the pre-odor (0–30 s), odor
            (30–60 s) and post-odor (60–90 s) epochs, overlaid by score. **Why it is
            here:** it shows that the score signal lives almost entirely in the odor
            epoch. Before onset the groups are nearly indistinguishable apart from score
            −1, which is elevated above about 3 Hz; during odor they fan out in score
            order; after offset they re-converge. Because baseline activity varies between
            animals without carrying score information, every spectral feature in the model
            is expressed as a during ÷ before ratio, and the amplitude features are
            normalised by baseline statistics (`mean_shift_z`, `peak_z` and `std_ratio` all
            divide by the pre-odor mean or standard deviation).""")


def fig_s7_during_before(sp, yv, colors, levels):
    f = sp["f_ep"]
    keep = f <= 10
    before = sp["psd_epoch"]["Before (0–30 s)"]
    during = sp["psd_epoch"]["During (30–60 s)"]

    fig, ax = plt.subplots(figsize=(FIG_W2, 3.0))
    ax.axhline(0, color=vz.AXIS, linewidth=0.8, zorder=1)
    for lo, hi in BANDS.values():
        ax.axvline(hi, color=vz.GRID, linewidth=0.5, zorder=0)
    for lvl in levels:
        m = yv == lvl
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log2(during[m][:, keep].mean(axis=0) / before[m][:, keep].mean(axis=0))
        ax.plot(f[keep], r, color=colors[lvl], linewidth=1.2, label=f"{lvl}")
    for i, ((lo, hi), lab) in enumerate(zip(BANDS.values(), BAND_LABELS.values())):
        ax.text((lo + min(hi, 10)) / 2, 1.02 + 0.055 * (i % 2), lab,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.5, color=vz.MUTED)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("log₂ (during ÷ before)")
    ax.set_xlim(0, 10)
    ax.set_title("Odor-evoked spectral gain, by score", pad=22)
    ax.legend(title="Human score", ncol=2, loc="lower right")
    vz.tidy(ax, grid_axis="y")
    fig.tight_layout()
    vz.save(fig, OUT, "S7_during_before_gain")
    plt.close(fig)

    caption("S7_during_before_gain", "Odor-evoked spectral gain and the band definitions",
            """Ratio of odor-epoch to pre-odor power, per score, on a log₂ axis; the
            vertical hairlines and top labels mark the five frequency bands the model uses.
            **Why it is here:** this is the most direct picture of the model's spectral
            features. Each `power_ratio_*` feature is the area of one of these curves within
            one band, so the figure shows both what those features measure and that they
            order by score across the whole range. It also shows why five bands rather than
            one summary number: the middle-score curves cross zero at different
            frequencies, so *where* a fly gains power — not only how much — carries
            information.""")


def fig_s8_band_power(sp, yv, colors, levels):
    fig, axes = plt.subplots(1, 5, figsize=(FIG_W2, 2.9), sharex=True)
    for ax, (bn, lab) in zip(axes, BAND_LABELS.items()):
        bp = sp["band_pw"][bn]
        data = [bp[yv == l] for l in levels]
        bpl = ax.boxplot(data, positions=range(len(levels)), widths=0.62,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(linewidth=1.0),
                         whiskerprops=dict(color=vz.MUTED, linewidth=0.6),
                         capprops=dict(color=vz.MUTED, linewidth=0.6))
        for patch, med, lvl in zip(bpl["boxes"], bpl["medians"], levels):
            patch.set_facecolor(colors[lvl])
            patch.set_edgecolor(vz.SURFACE)
            patch.set_linewidth(0.8)
            med.set_color(vz.ink_on(colors[lvl]))
        stat, p = kruskal(*data)
        ax.set_yscale("log")
        ax.set_title(lab, fontsize=8, pad=13)
        ax.text(0.0, 1.01, f"p = {p:.0e}", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=6.5, color=vz.MUTED)
        ax.yaxis.set_major_locator(mticker.LogLocator(numticks=5))
        ax.set_xlim(-0.65, len(levels) - 0.35)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(l) for l in levels], fontsize=6.5)
        ax.set_xlabel("Score")
        vz.tidy(ax)
    axes[0].set_ylabel("Band power (log)")
    fig.tight_layout()
    vz.save(fig, OUT, "S8_band_power_by_score")
    plt.close(fig)

    caption("S8_band_power_by_score", "Band power distributions and their significance",
            """Whole-trial power summed within each of the five model bands, distributed by
            human score (box = interquartile range, whisker = 1.5 × IQR, outliers omitted,
            log power axis). Each panel reports a Kruskal–Wallis test of whether band power
            differs across scores. **Why it is here:** it converts the spectral picture into
            the per-trial numbers the model actually receives, and shows that every band
            carries a significant, ordered effect — so none of the five is redundant on
            statistical grounds. The heavy overlap between adjacent scores is the reason the
            spectral features alone are not sufficient (see M5).""")


# =============================================================================
# Model training (replicates notebook 05)
# =============================================================================
ORIGINAL_CLASSES = np.array(SCORE_LEVELS)
ORIGINAL_TO_XGBOOST = {o: i for i, o in enumerate(ORIGINAL_CLASSES)}
XGBOOST_TO_ORIGINAL = {i: o for i, o in enumerate(ORIGINAL_CLASSES)}

# The cost matrix and objective live in the package so the trained model
# (scripts/train/train_ordinal_scorer.py) and these figures cannot disagree.
PENALTY = build_penalty_matrix()
ordinal_objective = make_objective(PENALTY)


def train_models(X_eng, X_sig, y, meta):
    print("Training the three feature-set models (replicates notebook 05)…")
    X_all = pd.concat([X_eng, X_sig], axis=1)
    idx = np.arange(len(y))
    eval_mask = meta["trial_num"].values != prep.EXCLUDE_TRIAL_NUM_FROM_EVAL
    eval_idx = idx[eval_mask]

    pool_idx, test_idx = train_test_split(eval_idx, test_size=0.10, random_state=SEED,
                                          stratify=y.iloc[eval_idx])
    train_core, val_idx = train_test_split(pool_idx, test_size=0.1111111111,
                                           random_state=SEED, stratify=y.iloc[pool_idx])
    train_idx = np.concatenate([train_core, idx[~eval_mask]])

    y_tr = y.iloc[train_idx].map(ORIGINAL_TO_XGBOOST).values
    y_va = y.iloc[val_idx].map(ORIGINAL_TO_XGBOOST).values
    print(f"  train {len(train_idx):,}   val {len(val_idx):,}   test {len(test_idx):,}")

    def fit(X):
        dtrain = xgb.DMatrix(X.iloc[train_idx], label=y_tr)
        dval = xgb.DMatrix(X.iloc[val_idx], label=y_va)
        params = {"num_class": N_CLASSES, "tree_method": "hist", "max_depth": 3,
                  "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 1.0,
                  "seed": SEED, "disable_default_eval_metric": 1}
        return xgb.train(params, dtrain, num_boost_round=300, obj=ordinal_objective,
                         evals=[(dval, "val")], verbose_eval=False)

    sets = {"Engineered only": X_eng, "Signal only": X_sig, "Combined": X_all}
    models = {k: fit(v) for k, v in sets.items()}

    def predict(model, X, rows):
        margin = model.predict(xgb.DMatrix(X.iloc[rows]), output_margin=True)
        return np.array([XGBOOST_TO_ORIGINAL[i]
                         for i in np.argmax(margin.reshape(-1, N_CLASSES), axis=1)])

    y_test = y.iloc[test_idx]
    rows = []
    preds = {}
    for name, X in sets.items():
        p = predict(models[name], X, test_idx)
        preds[name] = p
        rows.append({"feature_set": name,
                     "Accuracy": accuracy_score(y_test, p),
                     "Macro F1": f1_score(y_test, p, average="macro"),
                     "Within ±1": float(np.mean(np.abs(p - y_test.values) <= 1))})
    results = pd.DataFrame(rows)
    print(results.round(3).to_string(index=False))

    gain = models["Combined"].get_score(importance_type="gain")
    importance = pd.DataFrame({
        "feature": list(X_all.columns),
        "gain": [gain.get(f, 0.0) for f in X_all.columns],
        "family": ["Engineered"] * X_eng.shape[1] + ["Signal"] * X_sig.shape[1],
    }).sort_values("gain", ascending=False).reset_index(drop=True)

    return {"results": results, "importance": importance, "X_all": X_all,
            "y_test": y_test, "pred": preds["Combined"], "test_idx": test_idx}


# =============================================================================
# M1–M7 — model features
# =============================================================================
def fig_m1_profiles(X_all, y, importance, colors, levels):
    order = importance["feature"].tolist()
    fam = dict(zip(importance["feature"], importance["family"]))
    Z = X_all[order].rank(pct=True)          # rank-transform: robust to the heavy tails
    prof = np.vstack([Z[y == lvl].mean(axis=0).values for lvl in levels]).T - 0.5

    fig, ax = plt.subplots(figsize=(FIG_W2, 5.0))
    im = ax.imshow(prof, cmap=vz.DIV_CMAP, vmin=-0.35, vmax=0.35, aspect="auto")
    ax.set_xticks(range(len(levels)), [str(l) for l in levels])
    ax.set_yticks(range(len(order)), order, fontsize=7)
    for tick, feat in zip(ax.get_yticklabels(), order):
        tick.set_color(vz.CAT[0] if fam[feat] == "Engineered" else vz.CAT[1])
    ax.set_xlabel("Human score")
    ax.set_title("Feature value vs score (rank-transformed group mean)")
    ax.grid(False)
    vz.cell_grid(ax, len(order), len(levels))
    for sp_ in ax.spines.values():
        sp_.set_visible(False)

    vz.colorbar(fig, im, ax, "mean percentile − 0.5", fraction=0.030)
    handles = [Line2D([], [], color=vz.CAT[0], lw=3, label="Engineered (11)"),
               Line2D([], [], color=vz.CAT[1], lw=3, label="Signal (13)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.055), ncol=2)
    fig.tight_layout()
    vz.save(fig, OUT, "M1_feature_score_profiles")
    plt.close(fig)

    caption("M1_feature_score_profiles", "All 24 model features across the score scale",
            """Every feature the model receives, rows ordered by the model's own gain
            importance and colour-coded by family in the axis labels. Each feature is
            rank-transformed to percentiles (robust to the heavy tails of the ratio
            features) and averaged within each score group; the diverging scale is centred
            on the overall median, so grey means "typical" and the two poles mean
            "systematically high / low for this score". **Why it is here:** it is the
            single-page summary of the whole feature set. Features whose rows run smoothly
            from one pole to the other are the ones carrying ordinal information; flat rows
            are features that contribute only in interaction with others, which is what the
            gain ranking in M4 then quantifies.""")


def fig_m2_top_features(X_all, y, importance, colors, levels):
    top = importance.head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W2, 4.0))
    for ax, feat in zip(axes.ravel(), top):
        data = [X_all.loc[y == l, feat].values for l in levels]
        bpl = ax.boxplot(data, positions=range(len(levels)), widths=0.62,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(linewidth=1.0),
                         whiskerprops=dict(color=vz.MUTED, linewidth=0.6),
                         capprops=dict(color=vz.MUTED, linewidth=0.6))
        for patch, med, lvl in zip(bpl["boxes"], bpl["medians"], levels):
            patch.set_facecolor(colors[lvl])
            patch.set_edgecolor(vz.SURFACE)
            patch.set_linewidth(0.8)
            med.set_color(vz.ink_on(colors[lvl]))
        stat, p = kruskal(*data)
        ax.set_title(feat, fontsize=8, pad=13)
        ax.text(0.0, 1.01, f"Kruskal–Wallis p = {p:.0e}", transform=ax.transAxes,
                va="bottom", ha="left", fontsize=6.5, color=vz.MUTED)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(l) for l in levels], fontsize=6.5)
        vz.tidy(ax)
    for ax in axes[1]:
        ax.set_xlabel("Human score")
    fig.tight_layout()
    vz.save(fig, OUT, "M2_top_feature_distributions")
    plt.close(fig)

    caption("M2_top_feature_distributions", "Distribution of the six most-used features",
            """The six features with the highest gain in the trained combined model, shown
            as per-score distributions (box = IQR, whiskers = 1.5 × IQR, outliers omitted),
            with a Kruskal–Wallis p-value per feature. **Why it is here:** M1 shows
            direction; this shows *spread*. The medians rise with score, but the boxes for
            adjacent scores overlap substantially, and at the low end (−1, 0, 1) several
            features are almost indistinguishable. That is the quantitative statement of
            why the model is asked to predict an ordinal score under a cost-sensitive
            objective (M6) rather than to hit exact classes.""")


def fig_m3_correlation(X_all, importance):
    n_eng = int((importance["family"] == "Engineered").sum())
    order = ([f for f in X_all.columns if importance.set_index("feature").loc[f, "family"] == "Engineered"]
             + [f for f in X_all.columns if importance.set_index("feature").loc[f, "family"] == "Signal"])
    corr = X_all[order].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(corr.values, cmap=vz.DIV_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(order)), order, fontsize=6.5)
    for tick, feat in zip(ax.get_xticklabels() + ax.get_yticklabels(), order + order):
        tick.set_color(vz.CAT[0] if feat in order[:n_eng] else vz.CAT[1])
    for pos in (n_eng - 0.5,):
        ax.axhline(pos, color=vz.SURFACE, linewidth=1.6)
        ax.axvline(pos, color=vz.SURFACE, linewidth=1.6)
    ax.set_title("Spearman correlation between model features")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    vz.colorbar(fig, im, ax, "Spearman ρ")
    fig.tight_layout()
    vz.save(fig, OUT, "M3_feature_correlation")
    plt.close(fig)

    caption("M3_feature_correlation", "Redundancy within the feature set",
            """Spearman rank correlation between all 24 features, with engineered features
            (blue labels) in the upper-left block and signal features (orange labels) in the
            lower-right; the light rules separate the two families. **Why it is here:** it
            is the redundancy audit. The strong within-family blocks (particularly among the
            local-extremum features and among the five band-power ratios) show that neither
            family is 24 independent measurements, while the weak between-family
            correlations show the two families are measuring genuinely different things.
            That off-block weakness is the mechanistic reason the combined model beats
            either family alone in M5.""")


def fig_m4_importance(importance):
    fig, ax = plt.subplots(figsize=(FIG_W2, 4.4))
    d = importance.iloc[::-1].reset_index(drop=True)
    cols = [vz.CAT[0] if f == "Engineered" else vz.CAT[1] for f in d["family"]]
    ax.barh(range(len(d)), d["gain"], color=cols, height=0.66)
    ax.set_yticks(range(len(d)), d["feature"], fontsize=7)
    ax.set_xlabel("Gain (mean loss reduction per split)")
    ax.set_title("Feature importance in the trained combined model")
    top = d.iloc[-1]
    ax.text(top["gain"], len(d) - 1, f"  {top['gain']:.1f}", va="center", ha="left",
            fontsize=7, color=vz.SECONDARY)
    handles = [Line2D([], [], color=vz.CAT[0], lw=5, label="Engineered"),
               Line2D([], [], color=vz.CAT[1], lw=5, label="Signal")]
    ax.legend(handles=handles, loc="lower right")
    vz.tidy(ax, grid_axis="x")
    fig.tight_layout()
    vz.save(fig, OUT, "M4_feature_importance")
    plt.close(fig)

    caption("M4_feature_importance", "Which features the model actually uses",
            """Gain — the average reduction in the boundary-aware ordinal loss contributed
            by each feature's splits — for the trained combined model, coloured by feature
            family. **Why it is here:** it separates "features that discriminate" from
            "features the model relies on". Importance is heavily concentrated in a single
            baseline-normalised amplitude feature, with the spectral ratio features forming
            the bulk of the remaining useful signal. It also documents which features could
            be dropped in a reduced model, and pairs with M3: several near-zero features are
            near-zero because a correlated partner absorbs their splits, not because they
            are uninformative.""")


def fig_m5_ablation(results):
    metrics = ["Accuracy", "Macro F1", "Within ±1"]
    sets = results["feature_set"].tolist()

    fig, ax = plt.subplots(figsize=(FIG_W1 * 1.55, 2.4))
    yrow = np.arange(len(metrics))[::-1]

    # One sub-row per feature set inside each metric band, so close values never collide.
    offsets = {s: (1 - i) * 0.21 for i, s in enumerate(sets)}

    for r, metric in zip(yrow, metrics):
        vals = results[metric].values
        ax.plot([vals.min(), vals.max()], [r, r], color=vz.GRID, linewidth=1.2, zorder=1)

    for s, col in zip(sets, vz.CAT):
        vals = results.loc[results["feature_set"] == s, metrics].values.ravel()
        ypos = yrow + offsets[s]
        ax.plot(vals, ypos, "o", markersize=6.0, color=col, label=s, linestyle="none",
                markeredgecolor=vz.SURFACE, markeredgewidth=1.2, zorder=3)
        if s == "Combined":                       # direct-label only the headline series
            for v, yp in zip(vals, ypos):
                ax.text(v + 0.012, yp, f"{v:.2f}", ha="left", va="center",
                        fontsize=7, color=vz.SECONDARY)

    ax.set_yticks(yrow, metrics)
    ax.set_ylim(-0.55, len(metrics) - 0.45)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Held-out test score")
    ax.set_title("Feature-set ablation")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    vz.tidy(ax, grid_axis="x")
    fig.tight_layout()
    vz.save(fig, OUT, "M5_feature_set_ablation")
    plt.close(fig)

    caption("M5_feature_set_ablation", "What each feature family contributes",
            """The same model architecture and split trained three times — on the 11
            engineered features only, the 13 signal features only, and both together —
            evaluated on the held-out test set; each row is one metric, each dot one
            feature set, and the combined model's value is labelled. "Within ±1" is the
            fraction of trials predicted within one score of the human label.
            **Why it is here:** it is the
            justification for the whole spectral pipeline in one chart. Neither family is
            sufficient alone, and the combined set beats both on every metric, which is what
            the weak between-family correlation in M3 predicts.""")


def fig_m6_confusion(y_test, pred):
    cm = confusion_matrix(y_test, pred, labels=SCORE_LEVELS)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W2, 3.2))

    ax = axes[0]
    im = ax.imshow(norm, cmap=vz.SEQ_CMAP, vmin=0, vmax=1)
    for i in range(len(SCORE_LEVELS)):
        for j in range(len(SCORE_LEVELS)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5,
                        color=vz.SURFACE if norm[i, j] > 0.55 else vz.PRIMARY)
    ax.set_xticks(range(len(SCORE_LEVELS)), SCORE_LEVELS, fontsize=7)
    ax.set_yticks(range(len(SCORE_LEVELS)), SCORE_LEVELS, fontsize=7)
    ax.set_xlabel("Model prediction")
    ax.set_ylabel("Human score")
    ax.set_title("Held-out confusion matrix")
    ax.grid(False)
    vz.cell_grid(ax, len(SCORE_LEVELS), len(SCORE_LEVELS))
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    vz.colorbar(fig, im, ax, "fraction of true class", fraction=0.045, pad=0.03)

    ax = axes[1]
    im = ax.imshow(PENALTY, cmap=vz.SEQ_CMAP, vmin=0, vmax=PENALTY.max())
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{PENALTY[i, j]:g}", ha="center", va="center", fontsize=6.5,
                    color=vz.SURFACE if PENALTY[i, j] > PENALTY.max() * 0.55 else vz.PRIMARY)
    ax.set_xticks(range(N_CLASSES), SCORE_LEVELS, fontsize=7)
    ax.set_yticks(range(N_CLASSES), SCORE_LEVELS, fontsize=7)
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("True score")
    ax.set_title("Training misclassification cost")
    ax.grid(False)
    vz.cell_grid(ax, N_CLASSES, N_CLASSES)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    vz.colorbar(fig, im, ax, "penalty", fraction=0.045, pad=0.03)

    fig.tight_layout()
    vz.save(fig, OUT, "M6_confusion_and_cost")
    plt.close(fig)

    acc = accuracy_score(y_test, pred)
    w1 = float(np.mean(np.abs(pred - y_test.values) <= 1))
    caption("M6_confusion_and_cost", "Model errors and the cost structure that shapes them",
            f"""**Left:** held-out confusion matrix for the combined model --- cell text is the
            trial count, shading is the fraction of each true class, so a perfect model
            would be a solid diagonal. Exact accuracy is {acc:.2f} and {w1:.2f} of trials fall
            within one score of the human label. **Right:** the misclassification cost
            supplied to the custom training objective, which charges the model an expected
            cost rather than a flat log-loss. **Why it is here:** the two panels have to be
            read together, because the shape of the confusion matrix is a designed outcome
            rather than an accident. The cost matrix is deliberately **asymmetric**: rows
            are the true score and columns the prediction, so a column is the cost of
            *saying* that score. Under-calling a responder is charged heavily and scaled by
            distance --- a true 2 called 0 costs 8.0 against 3.0 for calling it 1, so when
            the model must err on a responder it is pushed to the nearer, smaller error ---
            while over-calling a non-responder is charged only moderately (a true 0 called
            2 costs 2.0). That asymmetry is load-bearing. Because score 0 is roughly half
            the labelled data, any cost that is large in the true-0 row makes the
            corresponding column unaffordable everywhere and that class becomes
            unpredictable: an earlier symmetric matrix left score 1 unreachable, and
            raising 0/2 symmetrically to compensate did the same to score 2. Charging 3.0
            for missing a true 1 downward against 0.5 for guessing 1 on a true 0 is what
            makes score 1 reachable at all.""")


# =============================================================================
N_ROWS = 0
N_DROPPED = 0


# =============================================================================
# LaTeX supplement
# =============================================================================
# Which reference each figure's method rests on. Keys are from
# docs/supplementary_references.bib.
CITES: dict[str, list[str]] = {
    "S1_trace_overview_by_score": [],
    "S2_psd_by_score": ["welch1967", "harris1978"],
    "S3_psd_ratio_to_score0": ["welch1967"],
    "S4_spectrogram_by_score": ["oppenheim2010", "cooley1965"],
    "S5_spectrogram_difference": ["oppenheim2010"],
    "S6_psd_by_epoch": ["welch1967"],
    "S7_during_before_gain": ["welch1967", "harris1978"],
    "S8_band_power_by_score": ["kruskal1952"],
    "M1_feature_score_profiles": [],
    "M2_top_feature_distributions": ["kruskal1952"],
    "M3_feature_correlation": ["spearman1904"],
    "M4_feature_importance": ["chen2016", "breiman1984", "friedman2001"],
    "M5_feature_set_ablation": ["chen2016", "pedregosa2011"],
    "M6_confusion_and_cost": ["frank2001", "elkan2001", "pedregosa2011"],
}

# Rendered width as a fraction of \textwidth, per figure aspect ratio.
FIG_WIDTH: dict[str, float] = {
    "M3_feature_correlation": 0.86,
    "M5_feature_set_ablation": 0.80,
}

_UNICODE = {
    "—": "---", "–": "--", "−": "$-$", "₂": "$_2$", "₁": "$_1$", "₀": "$_0$",
    "×": "$\\times$", "÷": "$\\div$", "±": "$\\pm$", "²": "$^2$",
    "≤": "$\\le$", "≥": "$\\ge$", "ρ": "$\\rho$", "Δ": "$\\Delta$",
    "…": "\\ldots{}", "’": "'", "“": "``", "”": "''",
}


def _latex_escape(text: str) -> str:
    for ch, repl in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
                     ("#", "\\#"), ("_", "\\_"), ("$", "\\$"),
                     ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}")]:
        text = text.replace(ch, repl)
    for ch, repl in _UNICODE.items():
        text = text.replace(ch, repl)
    return text


def md_to_latex(body: str) -> str:
    """Convert the caption's small markdown subset to LaTeX, code spans first."""
    import re

    out = []
    for i, part in enumerate(re.split(r"`([^`]*)`", body)):
        if i % 2:                                    # inside a `code` span
            out.append("\\texttt{" + _latex_escape(part) + "}")
            continue
        esc = _latex_escape(part)
        esc = re.sub(r'"([^"]*)"', r"``\1''", esc)
        esc = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", esc)
        esc = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\textit{\1}", esc)
        out.append(esc)
    return "".join(out)


TEX_PREAMBLE = r"""% ============================================================================
%  supplementary_figures.tex  --  AUTO-GENERATED, do not hand-edit.
%
%  Regenerate with:  python scripts/figures/make_thesis_figures.py
%  Bibliography:     docs/supplementary_references.bib
%
%  Compiles standalone:
%      pdflatex supplementary_figures && bibtex supplementary_figures \
%          && pdflatex supplementary_figures && pdflatex supplementary_figures
%
%  To splice into an existing thesis instead, copy everything between
%  "BEGIN FIGURES" and "END FIGURES" into your appendix and make sure your own
%  preamble loads graphicx and points \graphicspath at the figure PDFs.
%
%  Figure PDFs live in outputs/thesis_figures/pdf/ (git-ignored). For Overleaf,
%  copy that folder to docs/figures/ -- the \graphicspath below checks both.
% ============================================================================
\documentclass[11pt]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{booktabs}
\usepackage{url}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[hidelinks]{hyperref}

\graphicspath{{../outputs/thesis_figures/pdf/}{figures/}{./}}

% Figure numbers are set literally per figure (S1..S8, M1..M6) so that the
% cross-references inside the captions match the printed labels.
\renewcommand{\figurename}{Figure}
\setlength{\parskip}{0.4\baselineskip}
\setlength{\parindent}{0pt}

\title{Supplementary Figures\\[2pt]
  \large Signal characterisation and feature construction for the automated
  fly-behaviour scorer}
\date{}

\begin{document}
\maketitle
"""

TEX_CLOSING = r"""
\bibliographystyle{unsrtnat}
\bibliography{supplementary_references}

\end{document}
"""


def write_supplement_tex() -> None:
    """Write the LaTeX supplement: every figure with its caption and citations."""
    dst = prep.PROJECT_ROOT / "docs" / "supplementary_figures.tex"
    dst.parent.mkdir(parents=True, exist_ok=True)

    L = [TEX_PREAMBLE, r"\section*{Overview}", ""]

    L += [
        "All supplementary figures were computed on the same trials the combined "
        f"multiclass scorer is trained on: both blinded-scoring sources merged to their "
        f"human labels, {N_ROWS:,} scored trials. The signal figures (S1--S8) use the 90~s "
        f"window that every recording covers, which excludes {N_DROPPED} truncated "
        "recordings; the model-feature figures (M1--M6) use each trace at full length, "
        "exactly as the model receives it. All model figures come from a single fresh "
        "training run of the pipeline (fixed split seed, boundary-aware ordinal "
        "objective).",
        "",
        r"\paragraph{Software.} Analysis was carried out in Python~3 \citep{python3} "
        r"using NumPy \citep{harris2020}, SciPy \citep{virtanen2020} and pandas "
        r"\citep{mckinney2010}. Models were fitted with XGBoost \citep{chen2016} and "
        r"evaluated with scikit-learn \citep{pedregosa2011}. Figures were drawn with "
        r"Matplotlib \citep{hunter2007}.",
        "",
        r"\paragraph{Figure design.} Human score is an ordered quantity, so it is encoded "
        r"throughout with a single-hue blue ramp running light (low) to dark (high) rather "
        r"than a rainbow or a set of categorical hues, following current guidance on "
        r"colour in scientific figures \citep{crameri2020,rougier2014}. Nominal groups "
        r"(feature family, feature set) use fixed categorical slots; signed quantities "
        r"(log ratios, correlations, differences) use a blue-to-red diverging ramp with a "
        r"neutral grey midpoint at zero. Every palette was checked against "
        r"colour-vision-deficiency separation and surface-contrast thresholds before use "
        r"\citep{okabe2008}.",
        "",
        r"\paragraph{Related methods used elsewhere in the pipeline.} Pairwise band-power "
        r"comparisons in the exploratory analysis used the Mann--Whitney $U$ test "
        r"\citep{mann1947}; where many bands are tested at once, false-discovery-rate "
        r"control \citep{benjamini1995} is appropriate. Exploratory notebook figures used "
        r"seaborn \citep{waskom2021}, and the blinded video-scoring interface used OpenCV "
        r"\citep{bradski2000} for video decoding.",
        "",
        r"\clearpage",
        "",
        "% ---------------------------------------------------------------- BEGIN FIGURES",
        "",
    ]

    for stem, title, body in CAPTIONS:
        code = stem.split("_")[0]
        width = FIG_WIDTH.get(stem, 1.0)
        cites = CITES.get(stem, [])
        tex_body = md_to_latex(body)
        if cites:
            cite_cmd = f"~\\citep{{{','.join(cites)}}}"
            tex_body = (tex_body[:-1] + cite_cmd + "." if tex_body.endswith(".")
                        else tex_body + cite_cmd)
        L += [
            f"% ---- Figure {code}: {title}",
            f"\\renewcommand{{\\thefigure}}{{{code}}}",
            r"\begin{figure}[htbp]",
            r"  \centering",
            f"  \\includegraphics[width={width}\\textwidth]{{{stem}}}",
            f"  \\caption[{md_to_latex(title)}]{{\\textbf{{{md_to_latex(title)}.}} "
            f"{tex_body}}}",
            f"  \\label{{fig:{code.lower()}}}",
            r"\end{figure}",
            r"\clearpage",
            "",
        ]

    L += [r"\renewcommand{\thefigure}{\arabic{figure}}  % restore normal numbering",
          "% ------------------------------------------------------------------ END FIGURES",
          TEX_CLOSING]

    dst.write_text("\n".join(L))
    n_cited = len({k for v in CITES.values() for k in v})
    print(f"  wrote {dst.relative_to(prep.PROJECT_ROOT)} "
          f"({len(CAPTIONS)} figures, {n_cited} per-figure citations)")


def write_captions() -> None:
    lines = [
        "# Supplementary figures — captions",
        "",
        "Generated by `scripts/figures/make_thesis_figures.py`. Every figure is written as",
        "vector PDF (`pdf/`, for the thesis) and 400 dpi PNG (`png/`, for quick viewing).",
        "",
        "**Data.** All figures are computed on the same rows the combined multiclass scorer",
        f"is trained on: both blinded-scoring sources merged to their human labels, {N_ROWS:,}",
        "scored trials. The signal figures (S-series) use the 90 s window that every",
        f"recording covers, which excludes {N_DROPPED} truncated recordings; the model features",
        "(M-series) are computed on each trace's full length, exactly as the model receives",
        "them. Model figures come from a fresh training run of the notebook-05 pipeline",
        "(same split seed, same boundary-aware ordinal objective), so the numbers here",
        "supersede any earlier run on a smaller label set.",
        "",
        "**Colour.** Human score is an ordered quantity, so it is encoded with a single-hue",
        "blue ramp running light (low) to dark (high) — never a rainbow and never eight",
        "categorical hues. Nominal groups (feature family, feature set) use fixed",
        "categorical slots; signed quantities (log ratios, correlations, differences) use a",
        "blue-to-red diverging ramp with a neutral grey midpoint at zero. Every palette was",
        "checked against colour-vision-deficiency and contrast thresholds before use.",
        "",
        "## Index",
        "",
        "| Figure | Title |",
        "|---|---|",
    ]
    for stem, title, _ in CAPTIONS:
        lines.append(f"| {stem.split('_')[0]} | {title} |")
    lines += ["", "---", ""]
    for stem, title, body in CAPTIONS:
        code = stem.split("_")[0]
        lines += [f"### Figure {code} — {title}", "", f"`{stem}.pdf`", "", body, "", "---", ""]
    (OUT / "captions.md").write_text("\n".join(lines))
    print(f"  wrote captions.md ({len(CAPTIONS)} captions)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-read the source CSVs")
    args = ap.parse_args()

    vz.apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    bundle = prep.load(force=args.rebuild)
    traces_full, y, meta = bundle["traces"], bundle["y"], bundle["meta"]
    X_eng, X_sig = bundle["X_engineered"], bundle["X_signal"]

    # S-series: restrict to the 90 s window every recording covers.
    tr90 = traces_full[:, :N_FIG_FRAMES].astype(np.float64)
    changes = np.abs(np.diff(tr90, axis=1)) > 1e-9
    last = np.array([np.where(r)[0].max() + 1 if r.any() else 0 for r in changes])
    covers90 = last >= N_FIG_FRAMES - 10
    global N_ROWS, N_DROPPED
    N_ROWS, N_DROPPED = len(covers90), int((~covers90).sum())
    print(f"S-series uses {covers90.sum():,} of {len(covers90):,} traces "
          f"({N_DROPPED} truncated recordings excluded)")
    T, yv = tr90[covers90], y.values[covers90]

    levels = [l for l in SCORE_LEVELS if (yv == l).sum() >= 5]
    colors = vz.score_colors(levels)

    sp = compute_spectral(T)
    print("Signal figures…")
    fig_s1_traces(T, yv, colors, levels)
    fig_s2_psd(sp, yv, colors, levels)
    fig_s3_psd_ratio(sp, yv, colors, levels)
    fig_s4_spectrograms(sp, yv, colors, levels)
    fig_s5_spectrogram_diff(sp, yv, levels)
    fig_s6_epoch_psd(sp, yv, colors, levels)
    fig_s7_during_before(sp, yv, colors, levels)
    fig_s8_band_power(sp, yv, colors, levels)
    del sp, T

    fit = train_models(X_eng, X_sig, y, meta)
    mlevels = [l for l in SCORE_LEVELS if (y.values == l).sum() >= 5]
    mcolors = vz.score_colors(mlevels)

    print("Model-feature figures…")
    fig_m1_profiles(fit["X_all"], y.values, fit["importance"], mcolors, mlevels)
    fig_m2_top_features(fit["X_all"], y.values, fit["importance"], mcolors, mlevels)
    fig_m3_correlation(fit["X_all"], fit["importance"])
    fig_m4_importance(fit["importance"])
    fig_m5_ablation(fit["results"])
    fig_m6_confusion(fit["y_test"], fit["pred"])

    fit["results"].to_csv(OUT / "feature_set_ablation.csv", index=False)
    fit["importance"].to_csv(OUT / "feature_importance.csv", index=False)
    write_captions()
    write_supplement_tex()
    print(f"\nDone — {OUT}")


if __name__ == "__main__":
    main()

"""Out-of-fold model predictions for every labelled trial — the label-QC input.

Every trial gets a prediction from a model that did NOT train on it (5-fold
stratified CV over all labelled rows), so a disagreement is honest evidence and
never train-set memorisation. This is label quality control, not a performance
claim — testing_11 rows are included in the folds here even though they are
train-only in the published evaluation.

Writes outputs/ordinal_scorer/oof_predictions.csv with one row per trial:
    dataset, fly, fly_number, trial_label, source, human_score, model_pred,
    p_pred (confidence in its own answer), p_human (how much probability the
    model gives YOUR label), abs_diff, boundary_cross, flag

flag values, in decreasing severity:
    boundary+2  crossed the responder boundary AND off by >1
    boundary    responder/non-responder disagreement (true<=1 vs pred>=2 or vice versa)
    gt1         off by more than +/-1 without crossing the boundary
    within1     off by exactly 1
    exact       agreement

Usage:
    python scripts/eval/make_oof_predictions.py
    python scripts/eval/make_oof_predictions.py --rebuild   # re-read source CSVs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))

import prep_data as prep  # noqa: E402
from flybehavior_response.ordinal_cost import (  # noqa: E402
    N_CLASSES, SCORE_LEVELS, build_penalty_matrix, make_objective)

OUT = prep.PROJECT_ROOT / "outputs" / "ordinal_scorer"
SEED = prep.SEED
LEVELS = np.array(SCORE_LEVELS)
TO_XGB = {c: i for i, c in enumerate(SCORE_LEVELS)}

PARAMS = {
    "num_class": N_CLASSES, "tree_method": "hist", "max_depth": 3,
    "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 1.0,
    "seed": SEED, "disable_default_eval_metric": 1, "nthread": 8,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-read the source CSVs")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    penalty = build_penalty_matrix()
    objective = make_objective(penalty)

    bundle = prep.load(force=args.rebuild)
    y, meta = bundle["y"], bundle["meta"]
    X = pd.concat([bundle["X_engineered"], bundle["X_signal"]], axis=1)
    n = len(y)
    print(f"{n:,} labelled trials; {args.folds}-fold out-of-fold predictions")

    probs = np.full((n, N_CLASSES), np.nan)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        dtrain = xgb.DMatrix(X.iloc[tr], label=y.iloc[tr].map(TO_XGB).values)
        model = xgb.train(PARAMS, dtrain, 300, obj=objective, verbose_eval=False)
        margin = model.predict(xgb.DMatrix(X.iloc[va]),
                               output_margin=True).reshape(-1, N_CLASSES)
        e = np.exp(margin - margin.max(axis=1, keepdims=True))
        probs[va] = e / e.sum(axis=1, keepdims=True)
        print(f"  fold {k}/{args.folds} done ({len(va)} trials)")

    assert not np.isnan(probs).any()
    pred = LEVELS[probs.argmax(axis=1)]
    human = y.values
    diff = np.abs(pred - human)
    bnd = (human <= 1) != (pred <= 1)

    flag = np.where(bnd & (diff > 1), "boundary+2",
            np.where(bnd, "boundary",
             np.where(diff > 1, "gt1",
              np.where(diff == 1, "within1", "exact"))))

    df = meta[["dataset", "fly", "fly_number", "trial_label", "source"]].copy()
    df["human_score"] = human
    df["model_pred"] = pred
    df["p_pred"] = probs.max(axis=1).round(3)
    df["p_human"] = probs[np.arange(n), [TO_XGB[c] for c in human]].round(3)
    df["abs_diff"] = diff
    df["boundary_cross"] = bnd
    df["flag"] = flag
    for i, lvl in enumerate(SCORE_LEVELS):
        df[f"p_{lvl}"] = probs[:, i].round(3)

    # Most-suspicious first: big disagreements the model is confident about.
    sev = pd.Categorical(df["flag"],
                         ["boundary+2", "boundary", "gt1", "within1", "exact"],
                         ordered=True)
    df = df.assign(_sev=sev).sort_values(["_sev", "p_pred"],
                                         ascending=[True, False]).drop(columns="_sev")

    path = OUT / "oof_predictions.csv"
    df.to_csv(path, index=False)

    print(f"\nWrote {path}  ({len(df):,} rows)")
    print("\nflag counts:")
    print(df["flag"].value_counts().to_string())
    print(f"\nexact accuracy (OOF, all {n:,} trials): {(pred == human).mean():.3f}")
    print(f"within +/-1: {(diff <= 1).mean():.3f}   "
          f"boundary accuracy: {(~bnd).mean():.3f}")
    worst = df[df["flag"].isin(["boundary+2", "gt1", "boundary"])]
    print(f"\nreview queue (boundary or >+/-1): {len(worst)} trials "
          f"({len(worst)/n*100:.1f}% of labels)")


if __name__ == "__main__":
    main()

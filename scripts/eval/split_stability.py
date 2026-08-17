"""How much do the scorer's metrics depend on the luck of the train/test split?

Re-draws the 10% held-out test split with N different random seeds, retrains the
combined model on each complement, and reports the spread of every headline
metric. The model itself (features, cost matrix, hyperparameters, xgboost seed)
is identical across runs — only the partition changes.

With ``--grouped`` the split is drawn per FLY rather than per trial: every trial
of a fly lands on the same side, so the test metric measures generalisation to
unseen animals instead of unseen trials from familiar animals. testing_11 trials
(train-only by design) follow their fly — they join training only when their fly
is a training fly and are dropped otherwise, so no test fly ever leaks into
training through them.

    python scripts/eval/split_stability.py             # 10 trial-level splits
    python scripts/eval/split_stability.py --grouped   # 10 fly-level splits
    python scripts/eval/split_stability.py --n-seeds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))

import prep_data as prep  # noqa: E402
from flybehavior_response.ordinal_cost import (  # noqa: E402
    N_CLASSES, SCORE_LEVELS, build_penalty_matrix, make_objective)

OUT = prep.PROJECT_ROOT / "outputs" / "ordinal_scorer"
LEVELS = np.array(SCORE_LEVELS)
TO_XGB = {c: i for i, c in enumerate(SCORE_LEVELS)}

PARAMS = {"num_class": N_CLASSES, "tree_method": "hist", "max_depth": 3,
          "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 1.0,
          "seed": 42, "disable_default_eval_metric": 1, "nthread": 8}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--grouped", action="store_true",
                    help="split by fly (unseen-animal metric) instead of by trial")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    objective = make_objective(build_penalty_matrix())
    bundle = prep.load(force=args.rebuild)
    y, meta = bundle["y"], bundle["meta"]
    X = pd.concat([bundle["X_engineered"], bundle["X_signal"]], axis=1)
    idx = np.arange(len(y))
    eval_ok = meta["trial_num"].values != prep.EXCLUDE_TRIAL_NUM_FROM_EVAL

    # A fly's identity: source + dataset + batch/day + fly number within the batch.
    fly_id = (meta["source"].astype(str) + "|" + meta["dataset"].astype(str) + "|"
              + meta["fly"].astype(str) + "|" + meta["fly_number"].astype(str)).values
    mode = "fly-grouped" if args.grouped else "trial-stratified"
    print(f"{len(y):,} trials from {len(np.unique(fly_id)):,} flies; "
          f"{args.n_seeds} independent 90/10 splits ({mode})\n")

    rows = []
    for seed in range(args.n_seeds):
        if args.grouped:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
            tr_pos, te_pos = next(gss.split(idx[eval_ok], groups=fly_id[eval_ok]))
            train, test = idx[eval_ok][tr_pos], idx[eval_ok][te_pos]
            # testing_11 rows follow their fly: train-fly rows join training,
            # test-fly rows are dropped so no test fly leaks into training.
            test_flies = set(fly_id[test])
            extra = idx[~eval_ok][~np.isin(fly_id[~eval_ok], list(test_flies))]
            train = np.concatenate([train, extra])
        else:
            train, test = train_test_split(idx[eval_ok], test_size=0.10,
                                           random_state=seed,
                                           stratify=y.iloc[idx[eval_ok]])
            train = np.concatenate([train, idx[~eval_ok]])
        dtrain = xgb.DMatrix(X.iloc[train], label=y.iloc[train].map(TO_XGB).values)
        model = xgb.train(PARAMS, dtrain, 300, obj=objective, verbose_eval=False)
        margin = model.predict(xgb.DMatrix(X.iloc[test]),
                               output_margin=True).reshape(-1, N_CLASSES)
        pred, yt = LEVELS[margin.argmax(1)], y.iloc[test].values
        rows.append({
            "split_seed": seed,
            "exact_acc": (pred == yt).mean(),
            "within_1": (np.abs(pred - yt) <= 1).mean(),
            "boundary_acc": ((yt <= 1) == (pred <= 1)).mean(),
            "macro_f1": f1_score(yt, pred, average="macro", zero_division=0),
            "class1_recall": recall_score(yt, pred, labels=[1], average="macro",
                                          zero_division=0),
            "n_pred_1": int((pred == 1).sum()),
        })
        r = rows[-1]
        print(f"  seed {seed:>2}:  exact={r['exact_acc']:.3f}  within1={r['within_1']:.3f}  "
              f"boundary={r['boundary_acc']:.3f}  macroF1={r['macro_f1']:.3f}  "
              f"rec1={r['class1_recall']:.2f}")

    df = pd.DataFrame(rows)
    stats = df.drop(columns=["split_seed", "n_pred_1"]).agg(["mean", "std", "min", "max"])
    print("\nAcross splits:")
    print(stats.round(3).to_string())
    name = "split_stability_grouped.csv" if args.grouped else "split_stability.csv"
    df.to_csv(OUT / name, index=False)
    print(f"\nSaved per-split rows to {OUT / name}")


if __name__ == "__main__":
    main()

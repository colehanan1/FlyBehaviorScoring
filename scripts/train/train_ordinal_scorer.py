"""Train the combined-feature ordinal scorer under the asymmetric cost matrix.

Replaces the ad-hoc training cell in notebook 05. The cost matrix lives in
``src/flybehavior_response/ordinal_cost.py`` so the model, the supplementary
figures and the thesis text cannot drift apart.

    python scripts/train/train_ordinal_scorer.py                # train + evaluate
    python scripts/train/train_ordinal_scorer.py --rebuild      # re-read source CSVs
    python scripts/train/train_ordinal_scorer.py --cv           # 5-fold CV as well

Writes to outputs/ordinal_scorer/:
    model_ordinal_xgb.json      trained booster
    penalty_matrix.csv          the cost matrix it was trained under
    metrics.json                held-out test metrics
    test_predictions.csv        per-trial true/pred/source for the test split
    confusion_matrix.csv        held-out confusion counts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))

import prep_data as prep  # noqa: E402
from flybehavior_response.ordinal_cost import (  # noqa: E402
    N_CLASSES, SCORE_LEVELS, build_penalty_matrix, describe, expected_cost,
    make_objective)

OUT = prep.PROJECT_ROOT / "outputs" / "ordinal_scorer"
SEED = prep.SEED
LEVELS = np.array(SCORE_LEVELS)
TO_XGB = {c: i for i, c in enumerate(SCORE_LEVELS)}

PARAMS = {
    "num_class": N_CLASSES,
    "tree_method": "hist",
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "seed": SEED,
    "disable_default_eval_metric": 1,
}
N_ROUNDS = 300


def split(y: pd.Series, meta: pd.DataFrame):
    """10% test / 10% val / rest train; testing_11 is train-only, never evaluated."""
    idx = np.arange(len(y))
    eval_ok = meta["trial_num"].values != prep.EXCLUDE_TRIAL_NUM_FROM_EVAL
    pool, test = train_test_split(idx[eval_ok], test_size=0.10, random_state=SEED,
                                  stratify=y.iloc[idx[eval_ok]])
    core, val = train_test_split(pool, test_size=0.1111111111, random_state=SEED,
                                 stratify=y.iloc[pool])
    train = np.concatenate([core, idx[~eval_ok]])
    return train, val, test, pool, idx[~eval_ok]


def fit(X: pd.DataFrame, y: pd.Series, train, val, penalty) -> xgb.Booster:
    dtrain = xgb.DMatrix(X.iloc[train], label=y.iloc[train].map(TO_XGB).values)
    dval = xgb.DMatrix(X.iloc[val], label=y.iloc[val].map(TO_XGB).values)
    return xgb.train(PARAMS, dtrain, N_ROUNDS, obj=make_objective(penalty),
                     evals=[(dval, "val")], verbose_eval=False)


def predict(model: xgb.Booster, X: pd.DataFrame, rows):
    margin = model.predict(xgb.DMatrix(X.iloc[rows]), output_margin=True)
    margin = margin.reshape(-1, N_CLASSES)
    exp = np.exp(margin - margin.max(axis=1, keepdims=True))
    return LEVELS[margin.argmax(axis=1)], exp / exp.sum(axis=1, keepdims=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "within_1": float(np.mean(np.abs(y_pred - y_true) <= 1)),
        "boundary_error": float(np.mean((y_true <= 1) != (y_pred <= 1))),
        "class1_recall": float(recall_score(y_true, y_pred, labels=[1],
                                            average="macro", zero_division=0)),
        "class1_precision": float(precision_score(y_true, y_pred, labels=[1],
                                                  average="macro", zero_division=0)),
        "n_predicted_1": int((y_pred == 1).sum()),
        "n_true_1": int((y_true == 1).sum()),
    }


def run_cv(X, y, pool, holdout, penalty, folds: int = 5) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for k, (tr, va) in enumerate(skf.split(pool, y.iloc[pool]), 1):
        train_idx = np.concatenate([pool[tr], holdout])
        model = fit(X, y, train_idx, pool[va], penalty)
        pred, _ = predict(model, X, pool[va])
        rows.append({"fold": k, **metrics(y.iloc[pool[va]].values, pred)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-read the source CSVs")
    ap.add_argument("--cv", action="store_true", help="also run 5-fold CV on the pool")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    penalty = build_penalty_matrix()
    print("Misclassification cost (rows = truth, columns = prediction):")
    print(describe(penalty), "\n")

    bundle = prep.load(force=args.rebuild)
    y, meta = bundle["y"], bundle["meta"]
    X = pd.concat([bundle["X_engineered"], bundle["X_signal"]], axis=1)
    print(f"{len(y):,} scored trials")
    print("label counts:", dict(y.value_counts().reindex(SCORE_LEVELS, fill_value=0)), "\n")

    train, val, test, pool, holdout = split(y, meta)
    print(f"train={len(train):,}  val={len(val):,}  test={len(test):,} "
          f"(testing_{prep.EXCLUDE_TRIAL_NUM_FROM_EVAL} train-only: {len(holdout)})")

    if args.cv:
        cv = run_cv(X, y, pool, holdout, penalty)
        print("\n5-fold CV on the train pool:")
        print(cv.round(3).to_string(index=False))
        print("\n  mean:", {k: round(v, 3) for k, v in
                            cv.drop(columns="fold").mean().items()})
        cv.to_csv(OUT / "cv_metrics.csv", index=False)

    model = fit(X, y, train, val, penalty)
    y_test = y.iloc[test].values
    pred, probs = predict(model, X, test)
    m = metrics(y_test, pred)

    print("\nHeld-out test metrics:")
    for k, v in m.items():
        print(f"  {k:<18} {v:.3f}" if isinstance(v, float) else f"  {k:<18} {v}")
    print("\n", classification_report(y_test, pred, labels=SCORE_LEVELS, zero_division=0))

    cm = confusion_matrix(y_test, pred, labels=SCORE_LEVELS)
    cm_df = pd.DataFrame(cm, index=SCORE_LEVELS, columns=SCORE_LEVELS)
    cm_df.index.name = "true \\ pred"
    print("Confusion matrix:")
    print(cm_df.to_string())

    # Sanity check the mechanism: is class 1 ever the cheapest choice?
    ec = expected_cost(probs, penalty)
    n_argmin1 = int((ec.argmin(axis=1) == SCORE_LEVELS.index(1)).sum())
    print(f"\nclass 1 minimises expected cost on {n_argmin1}/{len(test)} test trials; "
          f"max p(1) = {probs[:, SCORE_LEVELS.index(1)].max():.3f}")

    model.save_model(str(OUT / "model_ordinal_xgb.json"))
    pd.DataFrame(penalty, index=SCORE_LEVELS,
                 columns=SCORE_LEVELS).to_csv(OUT / "penalty_matrix.csv")
    cm_df.to_csv(OUT / "confusion_matrix.csv")
    pd.DataFrame({"true": y_test, "pred": pred,
                  "source": meta.iloc[test]["source"].values,
                  "trial_label": meta.iloc[test]["trial_label"].values,
                  }).to_csv(OUT / "test_predictions.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(
        {"n_trials": int(len(y)), "n_test": int(len(test)), **m}, indent=2))

    gain = model.get_score(importance_type="gain")
    pd.DataFrame({"feature": list(X.columns),
                  "gain": [gain.get(f, 0.0) for f in X.columns]}
                 ).sort_values("gain", ascending=False).to_csv(
        OUT / "feature_importance.csv", index=False)

    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()

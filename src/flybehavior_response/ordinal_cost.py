"""Misclassification-cost matrix and expected-cost objective for the ordinal scorer.

Single source of truth for the cost structure — imported by the training script and
by the supplementary-figure script, so the model and the figures can never disagree
about what the model was trained to optimise.

Why the matrix is asymmetric
----------------------------
Cost is charged on the (true, predicted) pair. Rows are the truth, columns the
prediction, so **column k is the cost of *saying* k**. That column view is what
decides whether a class is ever predictable: the model picks the class with the
lowest expected cost, so if column k is dominated by another column under the
data's class priors, class k is never chosen no matter what the features say.

Class 0 is ~49% of the labelled data. Any cost that is large in the "true 0" *row*
therefore makes the corresponding column expensive everywhere, and that class dies.
Two failures were diagnosed this way:

* The original symmetric matrix predicted class **1** exactly zero times on all
  1,395 trials (max softmax probability for class 1 was 0.046 against a 0.081
  prior). Its ``(3, 1) = 10.0`` entry charged a distance-2 error more than the
  distance-6 error of confusing -1 with 5, and ``(2, 1) = 5.0`` made "say 1"
  costly wherever a responder was plausible.
* Simply raising 0<->2 *symmetrically* to satisfy "0<->2 harder than 1<->2" then
  killed class **2** the same way — cost 6.0 in the "true 0" row made "say 2"
  unaffordable against the class-0 prior.

The fix is to charge asymmetrically for what each mistake actually costs
scientifically:

* **Under-calling a responder** (true >= 2, predicted <= 1) is a false negative for
  the behaviour and is expensive, scaled by how far down the prediction lands:
  ``true 2 -> 0`` costs 8.0 while ``true 2 -> 1`` costs 3.0, so when the model must
  err on a responder it is pushed to the nearer, smaller error.
* **Over-calling a non-responder** (true <= 1, predicted >= 2) is a false positive.
  It is still penalised, but only moderately (``true 0 -> 2`` costs 2.0), because a
  large value here would make class 2 unreachable under the class-0 prior.
* **Missing a true 1 downward** costs 3.0 while **guessing 1 on a true 0** costs
  only 0.5, which is what makes class 1 reachable at all.

Verified by 5-fold CV on the training pool (1,255 trials): all seven classes are
predicted, class-1 recall 0.37 and class-2 recall 0.33, macro F1 0.543 against
0.489 for the original matrix, with the responder/non-responder boundary error
unchanged (0.091 vs 0.094).
"""

from __future__ import annotations

import numpy as np

SCORE_LEVELS: list[int] = [-1, 0, 1, 2, 3, 4, 5]
N_CLASSES: int = len(SCORE_LEVELS)
_IDX = {c: i for i, c in enumerate(SCORE_LEVELS)}

#: Full (true, predicted) cost specification. Outer key is the human score, inner
#: key the model's prediction. Diagonal is zero. Read a row as "what it costs me
#: to say each thing when the truth is this".
COST_SPEC: dict[int, dict[int, float]] = {
    -1: {-1: 0.0, 0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0, 5: 6.0},
     0: {-1: 1.0, 0: 0.0, 1: 0.5, 2: 2.0, 3: 3.0, 4: 4.0, 5: 6.0},
     1: {-1: 4.0, 0: 3.0, 1: 0.0, 2: 3.0, 3: 3.0, 4: 4.0, 5: 5.0},
     2: {-1: 7.0, 0: 8.0, 1: 3.0, 2: 0.0, 3: 1.0, 4: 2.0, 5: 3.0},
     3: {-1: 7.0, 0: 8.0, 1: 4.0, 2: 1.0, 3: 0.0, 4: 0.5, 5: 2.0},
     4: {-1: 8.0, 0: 7.0, 1: 5.0, 2: 2.0, 3: 0.5, 4: 0.0, 5: 0.5},
     5: {-1: 9.0, 0: 8.0, 1: 6.0, 2: 3.0, 3: 2.0, 4: 0.5, 5: 0.0},
}


def build_penalty_matrix(spec: dict[int, dict[int, float]] | None = None) -> np.ndarray:
    """Return the (true, predicted) cost matrix in ``SCORE_LEVELS`` order."""
    spec = COST_SPEC if spec is None else spec
    P = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    for true, row in spec.items():
        for pred, cost in row.items():
            P[_IDX[true], _IDX[pred]] = cost
    np.fill_diagonal(P, 0.0)
    return P


def make_objective(penalty: np.ndarray):
    """Expected-cost XGBoost objective for `penalty`.

    Minimises ``sum_j p_j * penalty[true, j]`` rather than log-loss, so the model is
    optimised against the cost of its mistakes instead of their count.
    """
    def objective(preds, dtrain):
        labels = dtrain.get_label().astype(int)
        raw = preds.reshape(len(labels), N_CLASSES)
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        cost = penalty[labels]
        grad = probs * (cost - (probs * cost).sum(axis=1, keepdims=True))
        hess = probs * (1.0 - probs) + 1e-6
        return grad.ravel(), hess.ravel()

    return objective


def expected_cost(probs: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    """Expected cost of each candidate prediction, given class probabilities."""
    return probs @ penalty


def describe(penalty: np.ndarray) -> str:
    """Readable dump of the matrix, rows = truth, columns = prediction."""
    import pandas as pd

    df = pd.DataFrame(penalty, index=SCORE_LEVELS, columns=SCORE_LEVELS)
    df.index.name = "true \\ pred"
    return df.to_string()

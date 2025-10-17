"""Sample-weight aware linear discriminant analysis."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin


class SampleWeightedLDA(BaseEstimator, ClassifierMixin):
    """Binary LDA with closed-form weighting support."""

    def __init__(self, *, reg: float = 1e-6) -> None:
        self.reg = float(reg)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("SampleWeightedLDA expects 2D input arrays.")
        if y.ndim != 1:
            raise ValueError("Target array must be one-dimensional.")
        n_samples, n_features = X.shape
        if n_samples != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != (n_samples,):
                raise ValueError("sample_weight must align with X and y.")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight entries must be non-negative.")
        total_weight = float(sample_weight.sum())
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("sample_weight must sum to a positive finite value.")

        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0, 1])):
            raise ValueError("SampleWeightedLDA currently supports binary targets encoded as {0, 1}.")

        mask0 = y == 0
        mask1 = y == 1
        w0 = sample_weight[mask0]
        w1 = sample_weight[mask1]
        w0_sum = float(w0.sum())
        w1_sum = float(w1.sum())
        if w0_sum == 0 or w1_sum == 0:
            raise ValueError("Each class must retain positive total weight.")

        mu0 = np.average(X[mask0], axis=0, weights=w0)
        mu1 = np.average(X[mask1], axis=0, weights=w1)

        def _weighted_cov(centered: np.ndarray, weights: np.ndarray) -> np.ndarray:
            scaled = centered * weights[:, None]
            return scaled.T @ centered

        centered0 = X[mask0] - mu0
        centered1 = X[mask1] - mu1
        cov = _weighted_cov(centered0, w0) + _weighted_cov(centered1, w1)
        denom = w0_sum + w1_sum
        if denom > 0:
            cov /= denom
        cov += self.reg * np.eye(n_features)

        inv_cov = np.linalg.pinv(cov, hermitian=True)
        mean_diff = mu1 - mu0
        self.coef_ = inv_cov @ mean_diff
        term1 = mu1 @ inv_cov @ mu1
        term0 = mu0 @ inv_cov @ mu0
        log_prior_ratio = math.log(w1_sum / w0_sum)
        self.intercept_ = -0.5 * (term1 - term0) + log_prior_ratio
        self.class_priors_ = np.array([w0_sum / denom, w1_sum / denom], dtype=np.float64)
        self.means_: Tuple[np.ndarray, np.ndarray] = (mu0, mu1)
        self.covariance_ = cov
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True
        return self

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("SampleWeightedLDA must be fitted before calling predict.")
        return X @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        scores = self._decision_function(X)
        return (scores >= 0.0).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        scores = self._decision_function(X)
        probs1 = expit(scores)
        probs1 = np.clip(probs1, 1e-8, 1 - 1e-8)
        return np.vstack([1 - probs1, probs1]).T

    def get_params(self, deep: bool = True) -> dict:
        return {"reg": self.reg}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


"""Custom MLP classifier with Adam optimizer and sample-weight support."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(x)
    grad[x > 0] = 1.0
    return grad


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SampleWeightedMLPClassifier(BaseEstimator, ClassifierMixin):
    """A lightweight MLP classifier that honours ``sample_weight``.

    The network uses ReLU hidden layers and a sigmoid output and is trained via
    the Adam optimiser. It is intentionally simple so it can serve as a drop-in
    replacement for scikit-learn's ``MLPClassifier`` in pipelines that require
    per-sample weighting.
    """

    def __init__(
        self,
        *,
        hidden_layer_sizes: Sequence[int] = (128,),
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iter: int = 200,
        batch_size: int = 64,
        tol: float = 1e-4,
        l2: float = 0.0,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.hidden_layer_sizes = tuple(int(size) for size in hidden_layer_sizes)
        self.learning_rate = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.batch_size = int(batch_size)
        self.tol = float(tol)
        self.l2 = float(l2)
        self.random_state = random_state
        self.verbose = bool(verbose)

    # ------------------------------------------------------------------
    # scikit-learn estimator API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("SampleWeightedMLPClassifier expects 2D input arrays.")
        if y.ndim != 1:
            raise ValueError("Target array must be one-dimensional for binary classification.")
        n_samples, n_features = X.shape
        if n_samples != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != (n_samples,):
                raise ValueError(
                    "sample_weight must be a one-dimensional array aligned with X and y."
                )
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight entries must be non-negative.")
        weight_sum = float(sample_weight.sum())
        if not math.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("sample_weight must sum to a positive finite value.")

        classes = np.unique(y)
        if not np.array_equal(classes, np.array([0.0, 1.0])):
            if not np.array_equal(classes, np.array([0.0])) and not np.array_equal(
                classes, np.array([1.0])
            ):
                raise ValueError(
                    "SampleWeightedMLPClassifier currently supports binary targets {0,1}."
                )
        self.classes_ = np.array([0, 1], dtype=int)

        rng = np.random.default_rng(self.random_state)
        layer_sizes: Tuple[int, ...] = (n_features, *self.hidden_layer_sizes, 1)
        self._weights: List[np.ndarray] = []
        self._biases: List[np.ndarray] = []
        self._m_w: List[np.ndarray] = []
        self._v_w: List[np.ndarray] = []
        self._m_b: List[np.ndarray] = []
        self._v_b: List[np.ndarray] = []

        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = math.sqrt(2.0 / fan_in)
            weights = rng.normal(0.0, limit, size=(fan_in, fan_out))
            bias = np.zeros(fan_out, dtype=np.float64)
            self._weights.append(weights)
            self._biases.append(bias)
            self._m_w.append(np.zeros_like(weights))
            self._v_w.append(np.zeros_like(weights))
            self._m_b.append(np.zeros_like(bias))
            self._v_b.append(np.zeros_like(bias))

        self._t = 0
        previous_loss = np.inf

        for epoch in range(self.max_iter):
            order = rng.permutation(n_samples)
            epoch_loss = 0.0
            total_batch_weight = 0.0
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = order[start:end]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                w_batch = sample_weight[batch_idx]
                batch_weight_sum = float(w_batch.sum())
                if batch_weight_sum <= 0:
                    continue

                activations: List[np.ndarray] = [X_batch]
                pre_activations: List[np.ndarray] = []
                for layer_index in range(len(self._weights) - 1):
                    z = activations[-1] @ self._weights[layer_index] + self._biases[layer_index]
                    pre_activations.append(z)
                    activations.append(_relu(z))

                logits = activations[-1] @ self._weights[-1] + self._biases[-1]
                pre_activations.append(logits)
                probs = _sigmoid(logits)
                activations.append(probs)

                clipped = np.clip(probs, 1e-8, 1 - 1e-8)
                losses = -(
                    y_batch * np.log(clipped)
                    + (1 - y_batch) * np.log(1 - clipped)
                )
                weighted_loss = float(np.sum(w_batch * losses)) / weight_sum
                epoch_loss += weighted_loss
                total_batch_weight += batch_weight_sum / weight_sum

                norm_weights = (w_batch / batch_weight_sum)[:, None]
                delta = (probs - y_batch[:, None]) * norm_weights

                grads_w: List[np.ndarray] = []
                grads_b: List[np.ndarray] = []

                for layer_pos in reversed(range(len(self._weights))):
                    activation_prev = activations[layer_pos]
                    grad_w = activation_prev.T @ delta
                    grad_b = delta.sum(axis=0)
                    if self.l2 > 0:
                        grad_w += self.l2 * self._weights[layer_pos]
                    grads_w.insert(0, grad_w)
                    grads_b.insert(0, grad_b)

                    if layer_pos > 0:
                        delta = delta @ self._weights[layer_pos].T
                        delta = delta * _relu_grad(pre_activations[layer_pos - 1])

                self._apply_adam_updates(grads_w, grads_b)

            if total_batch_weight > 0:
                epoch_loss /= total_batch_weight
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_iter} loss={epoch_loss:.6f}")
            if abs(previous_loss - epoch_loss) < self.tol:
                break
            previous_loss = epoch_loss

        self.n_iter_ = epoch + 1
        self.is_fitted_ = True
        return self

    def _apply_adam_updates(self, grads_w: Iterable[np.ndarray], grads_b: Iterable[np.ndarray]) -> None:
        self._t += 1
        lr = self.learning_rate
        for idx, (grad_w, grad_b) in enumerate(zip(grads_w, grads_b)):
            self._m_w[idx] = self.beta1 * self._m_w[idx] + (1 - self.beta1) * grad_w
            self._v_w[idx] = self.beta2 * self._v_w[idx] + (1 - self.beta2) * (grad_w ** 2)
            self._m_b[idx] = self.beta1 * self._m_b[idx] + (1 - self.beta1) * grad_b
            self._v_b[idx] = self.beta2 * self._v_b[idx] + (1 - self.beta2) * (grad_b ** 2)

            m_hat_w = self._m_w[idx] / (1 - self.beta1 ** self._t)
            v_hat_w = self._v_w[idx] / (1 - self.beta2 ** self._t)
            m_hat_b = self._m_b[idx] / (1 - self.beta1 ** self._t)
            v_hat_b = self._v_b[idx] / (1 - self.beta2 ** self._t)

            self._weights[idx] -= lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            self._biases[idx] -= lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def _forward(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("SampleWeightedMLPClassifier must be fitted before calling predict.")
        activations = X
        for weights, bias in zip(self._weights[:-1], self._biases[:-1]):
            activations = _relu(activations @ weights + bias)
        logits = activations @ self._weights[-1] + self._biases[-1]
        return _sigmoid(logits)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        probs = self._forward(X).reshape(-1, 1)
        probs = np.clip(probs, 1e-8, 1 - 1e-8)
        return np.hstack([1.0 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "max_iter": self.max_iter,
            "batch_size": self.batch_size,
            "tol": self.tol,
            "l2": self.l2,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


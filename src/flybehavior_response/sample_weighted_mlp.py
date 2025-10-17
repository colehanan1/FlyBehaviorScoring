"""Sample-weight aware extension of scikit-learn's MLPClassifier."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.base import is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import DERIVATIVES
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer, SGDOptimizer
from sklearn.utils import _safe_indexing, gen_batches, shuffle
from sklearn.utils.extmath import safe_sparse_dot


class SampleWeightedMLPClassifier(MLPClassifier):
    """Drop-in replacement that honours ``sample_weight`` during ``fit``.

    The implementation mirrors ``MLPClassifier`` but scales both the loss and
    gradients by the provided sample weights, ensuring that optimisation
    proceeds identically to the unweighted variant when all weights are equal.
    """

    def fit(self, X, y, sample_weight: Iterable[float] | None = None):  # type: ignore[override]
        if sample_weight is None:
            return super().fit(X, y)

        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.ndim != 1:
            raise ValueError("sample_weight must be one-dimensional")
        if sample_weight.shape[0] != len(X):
            raise ValueError("sample_weight must match the number of rows in X")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must be non-negative")
        total_weight = float(sample_weight.sum())
        if total_weight == 0:
            raise ValueError("sample_weight must sum to a positive value")

        # Persist weights for the duration of the parent fit call so that the
        # internal helpers (_fit_stochastic / _loss_grad_lbfgs) can retrieve
        # them without altering their public signatures.
        self._fit_sample_weight = sample_weight.astype(float, copy=False)
        try:
            return super().fit(X, y)
        finally:
            self._fit_sample_weight = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _set_current_sample_weight(self, sample_weight: np.ndarray | None) -> None:
        if sample_weight is None:
            self._current_sample_weight = None
            self._current_weight_sum = None
            return
        weight_array = np.asarray(sample_weight, dtype=float)
        if weight_array.ndim != 1:
            raise ValueError("Internal error: sample_weight should be 1-D")
        weight_sum = float(weight_array.sum())
        if weight_sum <= 0:
            raise ValueError("Internal error: non-positive sample weight sum")
        self._current_sample_weight = weight_array
        self._current_weight_sum = weight_sum

    # ------------------------------------------------------------------
    # Overrides of the stochastic optimiser to inject weighting
    # ------------------------------------------------------------------
    def _fit_stochastic(  # type: ignore[override]
        self,
        X,
        y,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
        layer_units,
        incremental,
    ):
        if getattr(self, "_fit_sample_weight", None) is None:
            return super()._fit_stochastic(
                X,
                y,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                layer_units,
                incremental,
            )

        params = self.coefs_ + self.intercepts_
        if not incremental or not hasattr(self, "_optimizer"):
            if self.solver == "sgd":
                self._optimizer = SGDOptimizer(
                    params,
                    self.learning_rate_init,
                    self.learning_rate,
                    self.momentum,
                    self.nesterovs_momentum,
                    self.power_t,
                )
            elif self.solver == "adam":
                self._optimizer = AdamOptimizer(
                    params,
                    self.learning_rate_init,
                    self.beta_1,
                    self.beta_2,
                    self.epsilon,
                )

        if self.early_stopping and incremental:
            raise ValueError("partial_fit does not support early_stopping=True")
        early_stopping = self.early_stopping

        sample_weight = self._fit_sample_weight.astype(float, copy=False)
        if early_stopping:
            should_stratify = is_classifier(self) and self.n_outputs_ == 1
            stratify = y if should_stratify else None
            X, X_val, y, y_val, sample_weight, sample_weight_val = train_test_split(
                X,
                y,
                sample_weight,
                random_state=self._random_state,
                test_size=self.validation_fraction,
                stratify=stratify,
            )
            if is_classifier(self):
                y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_val = None
            y_val = None
            sample_weight_val = None

        n_samples = X.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)
        total_weight = float(sample_weight.sum())

        if self.batch_size == "auto":
            batch_size = min(200, n_samples)
        else:
            if self.batch_size > n_samples:
                import warnings

                warnings.warn(
                    "Got `batch_size` less than 1 or larger than sample size. It is going to be clipped"
                )
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            self.n_iter_ = 0
            for it in range(self.max_iter):
                if self.shuffle:
                    sample_idx = shuffle(sample_idx, random_state=self._random_state)

                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    if self.shuffle:
                        batch_indices = sample_idx[batch_slice]
                        X_batch = _safe_indexing(X, batch_indices)
                        y_batch = y[batch_indices]
                        weight_batch = sample_weight[batch_indices]
                    else:
                        X_batch = X[batch_slice]
                        y_batch = y[batch_slice]
                        weight_batch = sample_weight[batch_slice]

                    self._set_current_sample_weight(weight_batch)
                    activations[0] = X_batch
                    batch_loss, coef_grads, intercept_grads = super()._backprop(
                        X_batch,
                        y_batch,
                        activations,
                        deltas,
                        coef_grads,
                        intercept_grads,
                    )
                    weight_sum = float(weight_batch.sum())
                    accumulated_loss += batch_loss * weight_sum

                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(params, grads)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / total_weight

                self.t_ += total_weight
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" % (self.n_iter_, self.loss_))

                self._update_no_improvement_count(early_stopping, X_val, y_val)
                self._optimizer.iteration_ends(self.t_)

                if self._no_improvement_count > self.n_iter_no_change:
                    if early_stopping:
                        msg = (
                            "Validation score did not improve more than tol=%f for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )
                    else:
                        msg = (
                            "Training loss did not improve more than tol=%f for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )

                    is_stopping = self._optimizer.trigger_stopping(msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break

                if self.n_iter_ == self.max_iter:
                    import warnings

                    warnings.warn(
                        "Stochastic Optimizer: Maximum iterations (%d) reached and the optimization hasn't converged yet."
                        % self.max_iter,
                        ConvergenceWarning,
                    )
        except KeyboardInterrupt:
            import warnings

            warnings.warn("Training interrupted by user.")

        self._set_current_sample_weight(None)

        if early_stopping:
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

    # ------------------------------------------------------------------
    # Loss / gradient computation with weighting support
    # ------------------------------------------------------------------
    def _compute_loss_grad(  # type: ignore[override]
        self,
        layer,
        n_samples,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
    ):
        if getattr(self, "_current_sample_weight", None) is None:
            return super()._compute_loss_grad(
                layer, n_samples, activations, deltas, coef_grads, intercept_grads
            )

        weight_sum = self._current_weight_sum
        coef_grads[layer] = safe_sparse_dot(activations[layer].T, deltas[layer])
        coef_grads[layer] += self.alpha * self.coefs_[layer]
        coef_grads[layer] /= weight_sum

        intercept_grads[layer] = deltas[layer].sum(axis=0) / weight_sum
        return coef_grads, intercept_grads

    def _backprop(  # type: ignore[override]
        self,
        X,
        y,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
    ):
        if getattr(self, "_current_sample_weight", None) is None:
            return super()._backprop(X, y, activations, deltas, coef_grads, intercept_grads)

        sample_weight = self._current_sample_weight.astype(X.dtype, copy=False)
        weight_sum = self._current_weight_sum
        activations = self._forward_pass(activations)
        loss_func_name = self.loss
        if loss_func_name == "log_loss" and self.out_activation_ == "logistic":
            loss_func_name = "binary_log_loss"

        y_pred = activations[-1]
        y_true = y
        eps = np.finfo(y_pred.dtype).eps
        if loss_func_name == "squared_error":
            loss_sample = 0.5 * np.square(y_true - y_pred).sum(axis=1)
        elif loss_func_name == "log_loss":
            y_prob = np.clip(y_pred, eps, 1 - eps)
            if y_prob.shape[1] == 1:
                y_prob = np.append(1 - y_prob, y_prob, axis=1)
            if y_true.shape[1] == 1:
                y_true = np.append(1 - y_true, y_true, axis=1)
            loss_sample = -(y_true * np.log(y_prob)).sum(axis=1)
        elif loss_func_name == "binary_log_loss":
            y_prob = np.clip(y_pred, eps, 1 - eps)
            loss_sample = -(
                y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
            ).sum(axis=1)
        else:
            raise ValueError(f"Unsupported loss function: {loss_func_name}")

        loss = np.average(loss_sample, weights=sample_weight)
        values = 0.0
        for s in self.coefs_:
            flat = s.ravel()
            values += np.dot(flat, flat)
        loss += (0.5 * self.alpha) * values / weight_sum

        last = self.n_layers_ - 2
        deltas[last] = y_pred - y_true
        deltas[last] *= sample_weight[:, np.newaxis]

        self._compute_loss_grad(last, X.shape[0], activations, deltas, coef_grads, intercept_grads)

        derivative = DERIVATIVES[self.activation]

        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            derivative(activations[i], deltas[i - 1])
            self._compute_loss_grad(
                i - 1, X.shape[0], activations, deltas, coef_grads, intercept_grads
            )

        return loss, coef_grads, intercept_grads

    # ------------------------------------------------------------------
    # L-BFGS weighting support
    # ------------------------------------------------------------------
    def _fit_lbfgs(  # type: ignore[override]
        self,
        X,
        y,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
        layer_units,
    ):
        if getattr(self, "_fit_sample_weight", None) is None:
            return super()._fit_lbfgs(
                X, y, activations, deltas, coef_grads, intercept_grads, layer_units
            )

        self._lbfgs_sample_weight = self._fit_sample_weight.astype(float, copy=False)
        try:
            return super()._fit_lbfgs(
                X, y, activations, deltas, coef_grads, intercept_grads, layer_units
            )
        finally:
            self._lbfgs_sample_weight = None

    def _loss_grad_lbfgs(  # type: ignore[override]
        self,
        packed_coef_inter,
        X,
        y,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
    ):
        if getattr(self, "_lbfgs_sample_weight", None) is None:
            return super()._loss_grad_lbfgs(
                packed_coef_inter, X, y, activations, deltas, coef_grads, intercept_grads
            )

        self._set_current_sample_weight(self._lbfgs_sample_weight)
        try:
            return super()._loss_grad_lbfgs(
                packed_coef_inter, X, y, activations, deltas, coef_grads, intercept_grads
            )
        finally:
            self._set_current_sample_weight(None)


__all__ = ["SampleWeightedMLPClassifier"]

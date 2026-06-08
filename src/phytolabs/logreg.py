"""Stage 2: logistic regression trained from scratch with mini-batch SGD.

No scikit-learn estimator here on purpose: the sigmoid, binary cross-entropy
loss, L2 regularization, and the SGD update are all implemented in NumPy. A
small standardizer is bundled in so the optimizer sees well-scaled features.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def binary_cross_entropy(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(proba, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


class StandardScaler:
    """Zero-mean, unit-variance scaling with stored statistics."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class LogisticRegressionSGD:
    """Binary logistic regression with mini-batch stochastic gradient descent."""

    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 300,
        batch_size: int = 16,
        l2: float = 1e-3,
        standardize: bool = True,
        random_state: int = 0,
    ) -> None:
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.standardize = standardize
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = None
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []

    def _prepare(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if not self.standardize:
            return X
        if fit:
            self.scaler = StandardScaler().fit(X)
        return self.scaler.transform(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionSGD":
        Xs = self._prepare(X, fit=True)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = Xs.shape

        rng = np.random.default_rng(self.random_state)
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        self.loss_history = []

        batch_size = min(self.batch_size, n_samples)
        for _epoch in range(self.epochs):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, batch_size):
                idx = order[start : start + batch_size]
                Xi, yi = Xs[idx], y[idx]
                proba = sigmoid(Xi @ self.w + self.b)
                error = proba - yi
                grad_w = Xi.T @ error / len(idx) + self.l2 * self.w
                grad_b = float(error.mean())
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

            proba_all = sigmoid(Xs @ self.w + self.b)
            loss = binary_cross_entropy(y, proba_all) + 0.5 * self.l2 * float(np.sum(self.w ** 2))
            self.loss_history.append(loss)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        Xs = self._prepare(X, fit=False)
        return Xs @ self.w + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionSGD":
        return joblib.load(path)

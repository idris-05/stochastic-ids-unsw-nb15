"""
Module 5: Logistic Regression Fusion Layer

Implements logistic regression from scratch:
    P(attack | X) = 1 / (1 + exp(-f(X)))

Fuses original features with the three module scores:
    - markov_score
    - statistical_score
    - bayesian_score
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


class LogisticRegressionFusion:
    """
    From-scratch logistic regression trained with mini-batch gradient descent.

    Parameters
    ----------
    lr          : learning rate
    n_epochs    : number of training epochs
    batch_size  : mini-batch size
    lambda_reg  : L2 regularisation coefficient
    tol         : early-stopping tolerance on loss improvement
    class_weight: 'balanced' to reweight loss by inverse class frequency
    """

    def __init__(self,
                 lr: float = 0.005,
                 n_epochs: int = 150,
                 batch_size: int = 1024,
                 lambda_reg: float = 0.01,
                 tol: float = 1e-6,
                 class_weight: str = 'balanced'):
        self.lr           = lr
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.lambda_reg   = lambda_reg
        self.tol          = tol
        self.class_weight = class_weight

        self.weights_      = None
        self.bias_         = 0.0
        self.loss_history_ = []
        self._w0           = 1.0   # weight for class 0
        self._w1           = 1.0   # weight for class 1

    # ── Internal helpers ──────────────────────────────────────────────────

    def _init_weights(self, n_features: int) -> None:
        rng = np.random.default_rng(42)
        self.weights_ = rng.normal(0, 0.01, size=n_features)
        self.bias_    = 0.0

    def _compute_class_weights(self, y: np.ndarray) -> None:
        n  = len(y)
        n1 = y.sum()
        n0 = n - n1
        if self.class_weight == 'balanced' and n0 > 0 and n1 > 0:
            self._w0 = n / (2.0 * n0)
            self._w1 = n / (2.0 * n1)
        else:
            self._w0 = self._w1 = 1.0

    def _forward(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights_ + self.bias_
        return sigmoid(z)

    def _loss(self, y_hat: np.ndarray, y: np.ndarray,
              sample_weights: np.ndarray = None) -> float:
        """Weighted binary cross-entropy + L2 regularisation."""
        eps = 1e-12
        bce = -(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
        if sample_weights is not None:
            bce = bce * sample_weights
        bce = bce.mean()
        l2  = 0.5 * self.lambda_reg * np.sum(self.weights_ ** 2)
        return float(bce + l2)

    def _make_sample_weights(self, y: np.ndarray) -> np.ndarray:
        w = np.where(y == 1, self._w1, self._w0)
        return w / w.mean()   # normalise so mean weight = 1

    def _step(self, X_b: np.ndarray, y_b: np.ndarray,
              sw_b: np.ndarray) -> None:
        """One gradient descent step on a mini-batch."""
        m     = len(y_b)
        y_hat = self._forward(X_b)
        err   = (y_hat - y_b) * sw_b   # weighted residuals

        grad_w = (X_b.T @ err) / m + self.lambda_reg * self.weights_
        grad_b = err.mean()

        self.weights_ -= self.lr * grad_w
        self.bias_    -= self.lr * grad_b

    # ── Public API ────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionFusion':
        """
        Train the logistic regression model.

        Parameters
        ----------
        X : np.ndarray  (n_samples, n_features)
        y : np.ndarray  (n_samples,) binary labels
        """
        n, d = X.shape
        self._init_weights(d)
        y  = y.astype(np.float64)
        self._compute_class_weights(y)
        sw = self._make_sample_weights(y)

        prev_loss = np.inf
        rng = np.random.default_rng(42)

        for epoch in range(self.n_epochs):
            idx = rng.permutation(n)
            X_s, y_s, sw_s = X[idx], y[idx], sw[idx]

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                self._step(X_s[start:end], y_s[start:end], sw_s[start:end])

            y_hat = self._forward(X)
            loss  = self._loss(y_hat, y, sw)
            self.loss_history_.append(loss)

            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(attack) for each sample."""
        return self._forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_weights(self) -> dict:
        return {'weights': self.weights_.copy(), 'bias': float(self.bias_)}

    def set_weights(self, weights: np.ndarray, bias: float) -> None:
        self.weights_ = weights.copy()
        self.bias_    = float(bias)


# ── Feature assembly utility ─────────────────────────────────────────────

def build_fusion_features(X_raw: np.ndarray,
                           markov_score: np.ndarray,
                           statistical_score: np.ndarray,
                           bayesian_score: np.ndarray) -> np.ndarray:
    """
    Concatenate preprocessed features with the three module scores.
    Returns shape (n_samples, n_raw_features + 3).
    """
    scores = np.stack([markov_score, statistical_score, bayesian_score], axis=1)
    return np.hstack([X_raw, scores])

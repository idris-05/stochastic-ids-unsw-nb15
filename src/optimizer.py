"""
Module 7: Stochastic Optimization

Algorithm from paper:
    x_{k+1} = x_k + δ_k    if f(x_{k+1}) >= f(x_k)
    x_{k+1} = x_k           otherwise

where δ_k ~ N(0, σ_k) with σ_k = σ_0 * decay^k  (geometric annealing)

Objective: F1(threshold) - α * FP_rate(threshold)

Key design:
- Phase 1 (60% of iterations): search threshold only, with a symmetric
  random walk.
- Phase 2 (40%): joint weight + threshold fine-tuning with smaller sigma.
- Fit against X_val/y_val only; evaluation happens on the test set afterward.
"""

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import f1_score


@dataclass
class OptParams:
    lr_weights: np.ndarray
    lr_bias: float
    threshold: float
    stat_threshold: float
    bayes_smooth: float
    markov_smooth: float

    def to_vector(self):
        return np.concatenate(
            [
                self.lr_weights,
                [
                    self.lr_bias,
                    self.threshold,
                    self.stat_threshold,
                    self.bayes_smooth,
                    self.markov_smooth,
                ],
            ]
        )

    @classmethod
    def from_vector(cls, vec, n_w):
        return cls(
            lr_weights=vec[:n_w].copy(),
            lr_bias=float(vec[n_w]),
            threshold=float(np.clip(vec[n_w + 1], 0.05, 0.95)),
            stat_threshold=float(np.clip(vec[n_w + 2], 0.05, 0.95)),
            bayes_smooth=float(np.clip(vec[n_w + 3], 1e-3, 10.0)),
            markov_smooth=float(np.clip(vec[n_w + 4], 1e-9, 1e-2)),
        )


def objective(y_true, y_proba, threshold, fp_weight=0.3):
    """F1 - fp_weight * FP_rate. Higher is better."""
    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fp_rate = fp / (fp + tn + 1e-9)
    return float(f1 - fp_weight * fp_rate)


class StochasticOptimizer:
    """
    Stochastic perturbation optimizer.

    Parameters
    ----------
    n_iter       : total iterations
    sigma_init   : initial σ for weight perturbations
    sigma_thresh : initial σ for threshold perturbation
    decay        : geometric decay applied to both sigmas each iteration
    fp_weight    : penalty weight on false-positive rate in objective
    """

    def __init__(
        self,
        n_iter=300,
        sigma_init=0.005,
        sigma_thresh=0.04,
        decay=0.997,
        fp_weight=0.3,
        random_seed=42,
    ):
        self.n_iter = n_iter
        self.sigma_init = sigma_init
        self.sigma_thresh = sigma_thresh
        self.decay = decay
        self.fp_weight = fp_weight
        self.rng = np.random.default_rng(random_seed)
        self.best_params_ = None
        self.best_obj_ = -np.inf
        self.obj_history_ = []

    def _eval(self, params, clf, X, y):
        clf.set_weights(params.lr_weights, params.lr_bias)
        return objective(y, clf.predict_proba(X), params.threshold, self.fp_weight)

    def optimize(self, classifier, X_val, y_val, init_params):
        n_w = len(init_params.lr_weights)
        x = init_params.to_vector()
        obj = self._eval(init_params, classifier, X_val, y_val)

        self.best_params_ = OptParams.from_vector(x.copy(), n_w)
        self.best_obj_ = obj
        self.obj_history_ = [obj]

        sw = self.sigma_init
        st = self.sigma_thresh
        phase1_end = int(self.n_iter * 0.6)

        for k in range(self.n_iter):
            delta = np.zeros_like(x)

            if k < phase1_end:
                # Phase 1: threshold-only search, unbiased random walk.
                delta[n_w + 1] = self.rng.normal(0.0, st)
            else:
                # Phase 2: gentle joint refinement
                delta[:n_w] = self.rng.normal(0, sw, size=n_w)
                delta[n_w] = self.rng.normal(0, sw)  # bias
                delta[n_w + 1] = self.rng.normal(0, st * 0.5)  # threshold

            x_cand = x + delta
            params_cand = OptParams.from_vector(x_cand, n_w)
            obj_cand = self._eval(params_cand, classifier, X_val, y_val)

            # Accept if improved (strict hill-climbing)
            if obj_cand >= obj:
                x = x_cand
                obj = obj_cand
                if obj_cand > self.best_obj_:
                    self.best_obj_ = obj_cand
                    self.best_params_ = OptParams.from_vector(x_cand, n_w)

            sw *= self.decay
            st *= self.decay
            self.obj_history_.append(obj)

        classifier.set_weights(self.best_params_.lr_weights, self.best_params_.lr_bias)
        return self.best_params_

    def convergence_summary(self):
        h = np.array(self.obj_history_)
        return {
            "n_iterations": len(h),
            "initial_obj": float(h[0]),
            "final_obj": float(h[-1]),
            "best_obj": float(self.best_obj_),
            "improvement": float(self.best_obj_ - h[0]),
            "converged": bool(len(h) >= 10 and abs(h[-10:].mean() - h[-1]) < 1e-4),
        }

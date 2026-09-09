"""
Module 4: Bayesian Signature Matching

Implements probabilistic pattern matching using Bayes' theorem:
    P(A|B) = P(B|A) * P(A) / P(B)

where A = attack, B = observed feature pattern.
"""

import numpy as np
import pandas as pd
from collections import defaultdict


# ── Signature features used for pattern matching ──────────────────────────
# These are high-discrimination categorical / behavioral features.
SIGNATURE_FEATURES = [
    'proto', 'state', 'service',
    'is_sm_ips_ports', 'is_ftp_login',
    'ct_state_ttl', 'ct_flw_http_mthd',
]


class BayesianSignatureModel:
    """
    Naive-Bayes style probabilistic pattern matcher.

    For each flow we compute P(attack | feature_pattern) using Bayes:
        P(attack | B) ∝ P(B | attack) * P(attack)

    Features are discretised / categorical; Laplace smoothing prevents
    zero probabilities for unseen patterns.

    Parameters
    ----------
    smoothing : float
        Laplace smoothing value (alpha).
    """

    def __init__(self, smoothing: float = 1.0):
        self.smoothing     = smoothing
        self._prior        = {}        # {0: P(normal), 1: P(attack)}
        self._likelihoods  = {}        # {feat: {cls: {val: P(val|cls)}}}
        self._classes      = [0, 1]
        self._vocab        = {}        # {feat: set of seen values}

    # ── Internal helpers ──────────────────────────────────────────────────

    def _discretise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Discretise continuous features to 5-bin categories using qcut.
        Categorical / binary features are left as-is (cast to string).
        """
        df = df.copy()
        binary_features = ['is_sm_ips_ports', 'is_ftp_login']
        integer_features = ['ct_state_ttl', 'ct_flw_http_mthd']

        for feat in SIGNATURE_FEATURES:
            if feat not in df.columns:
                df[feat] = 'unknown'
                continue

            if feat in binary_features:
                df[feat] = df[feat].fillna(0).astype(int).astype(str)
            elif feat in integer_features:
                vals = pd.to_numeric(df[feat], errors='coerce').fillna(0)
                try:
                    df[feat] = pd.cut(vals, bins=5, labels=False,
                                      duplicates='drop').fillna(0).astype(int).astype(str)
                except Exception:
                    df[feat] = vals.astype(int).astype(str)
            else:
                df[feat] = df[feat].fillna('unknown').astype(str)

        return df

    # ── Public API ────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> 'BayesianSignatureModel':
        """
        Estimate P(attack), P(normal) and likelihoods P(feature_val | class).
        """
        df = self._discretise(df)
        n  = len(y)
        n1 = int(y.sum())
        n0 = n - n1

        # Class priors
        self._prior = {
            0: (n0 + self.smoothing) / (n + 2 * self.smoothing),
            1: (n1 + self.smoothing) / (n + 2 * self.smoothing),
        }

        # Likelihoods per feature
        for feat in SIGNATURE_FEATURES:
            if feat not in df.columns:
                continue
            self._vocab[feat] = set(df[feat].unique())
            self._likelihoods[feat] = {}

            for cls, cls_mask in [(0, y == 0), (1, y == 1)]:
                cls_vals   = df.loc[cls_mask, feat]
                n_cls      = cls_mask.sum()
                vocab_size = len(self._vocab[feat])
                val_counts = cls_vals.value_counts()

                self._likelihoods[feat][cls] = {}
                for val in self._vocab[feat]:
                    count = val_counts.get(val, 0)
                    self._likelihoods[feat][cls][val] = (
                        (count + self.smoothing) /
                        (n_cls + self.smoothing * vocab_size)
                    )
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return P(attack | features) for each flow.
        Uses log-space computation for numerical stability.
        """
        df = self._discretise(df)
        n  = len(df)

        log_post = np.zeros((n, 2))
        for cls in self._classes:
            log_post[:, cls] = np.log(self._prior[cls] + 1e-12)

        for feat in SIGNATURE_FEATURES:
            if feat not in df.columns or feat not in self._likelihoods:
                continue
            vals = df[feat].values
            for cls in self._classes:
                feat_lkl = self._likelihoods[feat][cls]
                for i, v in enumerate(vals):
                    p = feat_lkl.get(v, self.smoothing / (
                        len(self._vocab.get(feat, {'_'})) * 10))
                    log_post[i, cls] += np.log(p + 1e-12)

        # Softmax to get probabilities
        log_post -= log_post.max(axis=1, keepdims=True)
        exp_post  = np.exp(log_post)
        probs     = exp_post / exp_post.sum(axis=1, keepdims=True)
        return probs[:, 1]   # P(attack)

    def bayesian_update(self, df: pd.DataFrame, y: np.ndarray,
                        weight: float = 0.1) -> None:
        """
        Online Bayesian update: blend new evidence into existing priors
        and likelihoods using a learning-rate parameter.
        """
        df = self._discretise(df)
        n  = len(y)
        n1 = int(y.sum())
        n0 = n - n1

        # Update priors with exponential moving average
        new_prior_1 = (n1 + self.smoothing) / (n + 2 * self.smoothing)
        new_prior_0 = (n0 + self.smoothing) / (n + 2 * self.smoothing)
        self._prior[1] = (1 - weight) * self._prior[1] + weight * new_prior_1
        self._prior[0] = (1 - weight) * self._prior[0] + weight * new_prior_0

        # Update likelihoods
        for feat in SIGNATURE_FEATURES:
            if feat not in df.columns or feat not in self._likelihoods:
                continue

            # Extend vocab with new values
            new_vals = set(df[feat].unique()) - self._vocab.get(feat, set())
            self._vocab.setdefault(feat, set()).update(new_vals)

            for cls, cls_mask in [(0, y == 0), (1, y == 1)]:
                cls_vals   = df.loc[cls_mask, feat]
                n_cls      = cls_mask.sum()
                if n_cls == 0:
                    continue
                vocab_size = len(self._vocab[feat])
                val_counts = cls_vals.value_counts()

                old_lkl = self._likelihoods[feat].get(cls, {})
                new_lkl = {}
                for val in self._vocab[feat]:
                    count   = val_counts.get(val, 0)
                    new_p   = (count + self.smoothing) / (n_cls + self.smoothing * vocab_size)
                    old_p   = old_lkl.get(val, new_p)
                    new_lkl[val] = (1 - weight) * old_p + weight * new_p

                self._likelihoods[feat][cls] = new_lkl

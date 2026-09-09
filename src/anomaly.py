"""
Module 3: Statistical + Queue-Based Anomaly Detection

Anomaly score: AS(X) = P(X|H0) / P(X)  (log-likelihood ratio form)
Queue model:   M/M/1 with λ, μ, L metrics

Fix: case-insensitive column resolution for real UNSW-NB15 lowercase columns.
"""

import numpy as np
import pandas as pd
from scipy import stats

# Each entry: (canonical_key, [possible column names in order of preference])
_COL_ALIASES = [
    ('spkts',   ['Spkts', 'spkts']),
    ('dpkts',   ['Dpkts', 'dpkts']),
    ('sload',   ['Sload', 'sload']),
    ('dload',   ['Dload', 'dload']),
    ('sbytes',  ['sbytes', 'Sbytes']),
    ('dbytes',  ['dbytes', 'Dbytes']),
    ('dur',     ['dur', 'Dur']),
    ('sinpkt',  ['Sintpkt', 'sinpkt', 'Sinpkt']),
    ('dinpkt',  ['Dintpkt', 'dinpkt', 'Dinpkt']),
    ('tcprtt',  ['tcprtt', 'Tcprtt']),
    ('synack',  ['synack', 'Synack']),
    ('ackdat',  ['ackdat', 'Ackdat']),
    ('sjit',    ['Sjit', 'sjit']),
    ('djit',    ['Djit', 'djit']),
    ('sttl',    ['sttl', 'Sttl']),
    ('dttl',    ['dttl', 'Dttl']),
    ('smean',   ['smeansz', 'smean', 'Smeansz']),
    ('dmean',   ['dmeansz', 'dmean', 'Dmeansz']),
    ('rate',    ['rate', 'Rate']),
    ('ct_srv_src', ['ct_srv_src']),
    ('ct_srv_dst', ['ct_srv_dst']),
]


def _resolve_col(df: pd.DataFrame, aliases: list):
    lower_map = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def _get_val(df, aliases, default=None):
    col = _resolve_col(df, aliases)
    if col is None:
        if default is not None:
            return pd.Series(np.full(len(df), default), index=df.index)
        return None
    return pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)


class StatisticalAnomalyModel:

    def __init__(self, contamination: float = 0.15):
        self.contamination = contamination
        self._params       = {}
        self._queue_params = {}
        self._threshold    = None
        self._score_mean   = 0.0
        self._score_std    = 1.0

    def _compute_queue_metrics(self, df):
        eps  = 1e-9
        dur  = _get_val(df, ['dur'], 1.0).fillna(1.0).clip(lower=eps)
        spkt = _get_val(df, ['Spkts','spkts'], 1.0).fillna(1.0)
        dpkt = _get_val(df, ['Dpkts','dpkts'], 1.0).fillna(1.0)
        lam  = (spkt / dur).clip(lower=eps)
        mu   = (dpkt / dur).clip(lower=eps)
        rho  = (lam / mu).clip(upper=0.9999)
        L    = rho / (1.0 - rho)
        return {'lambda': lam.values, 'mu': mu.values,
                'rho': rho.values, 'L': L.values}

    def fit(self, df, y=None):
        if y is None:
            for lbl in ['label','Label','LABEL']:
                if lbl in df.columns:
                    y = df[lbl].values.astype(int); break

        weights = {}
        for canonical, aliases in _COL_ALIASES:
            vals = _get_val(df, aliases)
            if vals is None:
                continue
            vals = vals.fillna(vals.median() if not vals.isna().all() else 0)

            if y is not None and len(np.unique(y)) == 2:
                vn = vals[y == 0].values; va = vals[y == 1].values
                if len(vn) < 2 or len(va) < 2: continue
                mu_n = float(vn.mean()); sig_n = max(float(vn.std()), 1e-9)
                mu_a = float(va.mean()); sig_a = max(float(va.std()), 1e-9)
                pooled = np.sqrt((sig_n**2 + sig_a**2) / 2.0)
                w = float(abs(mu_a - mu_n) / pooled)
            else:
                v = vals.values
                mu_n = mu_a = float(np.nanmean(v))
                sig_n = sig_a = max(float(np.nanstd(v)), 1e-9)
                w = 1.0

            self._params[canonical] = (mu_n, sig_n, mu_a, sig_a, w)
            weights[canonical] = w

        if not self._params:
            raise RuntimeError(
                f"StatModel: no columns found. Available: {list(df.columns)}")

        total = sum(weights.values()) + 1e-12
        for k in self._params:
            p = self._params[k]
            self._params[k] = p[:4] + (p[4] / total,)

        qm = self._compute_queue_metrics(df)
        self._queue_params = {
            'rho_mean': float(np.nanmean(qm['rho'])),
            'rho_std':  max(float(np.nanstd(qm['rho'])), 1e-9),
        }

        raw = self._raw_scores(df)
        self._score_mean = float(np.nanmean(raw))
        self._score_std  = max(float(np.nanstd(raw)), 1e-9)
        self._threshold  = float(np.nanpercentile(raw, (1-self.contamination)*100))

        fitted = list(self._params.keys())
        print(f"    [StatModel] Fitted {len(fitted)} features: {fitted}")
        return self

    def _raw_scores(self, df):
        n = len(df); scores = np.zeros(n)
        for canonical, (mu_n, sig_n, mu_a, sig_a, w) in self._params.items():
            aliases = next((al for cn,al in _COL_ALIASES if cn==canonical), [canonical])
            vals = _get_val(df, aliases)
            if vals is None: continue
            x = vals.fillna(mu_n).values
            scores += w * (stats.norm.logpdf(x, mu_a, sig_a) -
                           stats.norm.logpdf(x, mu_n, sig_n))
        qm = self._compute_queue_metrics(df)
        scores += (qm['rho'] - self._queue_params.get('rho_mean',0.5)) / \
                   self._queue_params.get('rho_std', 0.3)
        return scores

    def score(self, df):
        raw = self._raw_scores(df)
        z   = (raw - self._score_mean) / self._score_std
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, df):
        return (self.score(df) > 0.5).astype(int)

    def bayesian_update(self, df, new_threshold=None):
        alpha = 0.1
        for canonical, p in list(self._params.items()):
            aliases = next((al for cn,al in _COL_ALIASES if cn==canonical), [canonical])
            vals = _get_val(df, aliases)
            if vals is None: continue
            v = vals.dropna()
            if len(v) > 1:
                self._params[canonical] = (
                    (1-alpha)*p[0] + alpha*float(v.mean()),
                    max((1-alpha)*p[1] + alpha*(float(v.std())+1e-9), 1e-9),
                    p[2], p[3], p[4])
        if new_threshold is not None:
            self._threshold = float(new_threshold)

"""
Module 2: Markov Chain State-Sequence Modeling
Detects abnormal protocol state transitions.
Case-insensitive column resolution for real UNSW-NB15 data.
"""

import numpy as np
import pandas as pd


def _find_col(df, candidates):
    """Case-insensitive column lookup."""
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


class MarkovChainModel:

    def __init__(self, smoothing=1e-4, anomaly_percentile=10.0):
        self.smoothing          = smoothing
        self.anomaly_percentile = anomaly_percentile
        self.trans_prob_        = {}
        self.state_log_prob_    = {}
        self.state_index_       = {}
        self._states            = []
        self.threshold_         = None
        self._score_mean        = 0.0
        self._score_std         = 1.0

    def _get_state(self, row) -> str:
        proto_col = _find_col(pd.DataFrame([row]), ['proto', 'Proto'])
        state_col = _find_col(pd.DataFrame([row]), ['state', 'State'])
        proto = str(row.get('proto', row.get('Proto', 'unk'))).strip().lower()
        state = str(row.get('state', row.get('State', 'unk'))).strip().lower()
        return f"{proto}|{state}"

    def _get_states_from_df(self, df: pd.DataFrame) -> list:
        proto_col = _find_col(df, ['proto', 'Proto']) or 'proto'
        state_col = _find_col(df, ['state', 'State']) or 'state'
        protos = df[proto_col].fillna('unk').astype(str).str.strip().str.lower() \
            if proto_col in df.columns else pd.Series(['unk']*len(df))
        states = df[state_col].fillna('unk').astype(str).str.strip().str.lower() \
            if state_col in df.columns else pd.Series(['unk']*len(df))
        return [f"{p}|{s}" for p, s in zip(protos, states)]

    def fit(self, df: pd.DataFrame) -> 'MarkovChainModel':
        time_col = _find_col(df, ['Stime', 'stime', 'Ltime'])
        if time_col:
            df = df.sort_values(time_col).reset_index(drop=True)

        seq = self._get_states_from_df(df)
        unique = sorted(set(seq))
        self._states      = unique
        self.state_index_ = {s: i for i, s in enumerate(unique)}
        n = len(unique)

        counts = np.full((n, n), self.smoothing)
        for i in range(len(seq) - 1):
            counts[self.state_index_[seq[i]], self.state_index_[seq[i+1]]] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        prob_mat = counts / (row_sums + 1e-12)

        for s in unique:
            i = self.state_index_[s]
            self.trans_prob_[s] = {
                t: float(np.log(prob_mat[i, self.state_index_[t]] + 1e-12))
                for t in unique
            }

        total = len(seq)
        cnt_series = pd.Series(seq).value_counts()
        for s in unique:
            c = cnt_series.get(s, 0)
            self.state_log_prob_[s] = float(
                np.log((c + self.smoothing) / (total + self.smoothing * n)))

        raw = self._raw_scores(seq)
        self._score_mean = float(np.mean(raw))
        self._score_std  = max(float(np.std(raw)), 1e-9)
        self.threshold_  = float(np.percentile(raw, 100 - self.anomaly_percentile))
        return self

    def _raw_scores(self, seq: list) -> np.ndarray:
        scores = []
        for i, s in enumerate(seq):
            lp_m = self.state_log_prob_.get(s, np.log(self.smoothing))
            if i > 0:
                lp_t = self.trans_prob_.get(seq[i-1], {}).get(s, np.log(self.smoothing))
            else:
                lp_t = lp_m
            scores.append(-(0.5 * lp_m + 0.5 * lp_t))
        return np.array(scores, dtype=np.float64)

    def score(self, df: pd.DataFrame) -> np.ndarray:
        time_col = _find_col(df, ['Stime', 'stime', 'Ltime'])
        if time_col:
            order = df[time_col].argsort().values
            orig  = np.argsort(order)
            seq   = self._get_states_from_df(df.iloc[order])
        else:
            orig = np.arange(len(df))
            seq  = self._get_states_from_df(df)

        raw = self._raw_scores(seq)
        z   = (raw - self._score_mean) / self._score_std
        return (1.0 / (1.0 + np.exp(-z)))[orig]

    def bayesian_update(self, df: pd.DataFrame) -> None:
        alpha = 0.05
        seq   = self._get_states_from_df(df)
        for i in range(len(seq) - 1):
            sp, sn = seq[i], seq[i+1]
            if sp not in self.trans_prob_:
                self.trans_prob_[sp] = {}
            old = self.trans_prob_[sp].get(sn, np.log(self.smoothing))
            self.trans_prob_[sp][sn] = (1 - alpha) * old + alpha * np.log(1 + self.smoothing)

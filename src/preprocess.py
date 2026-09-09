"""
Module 1: Preprocessing
Handles all data preparation steps for the UNSW-NB15 dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import hashlib
import warnings
warnings.filterwarnings('ignore')


# ── Feature categories ──────────────────────────────────────────────────────
NOMINAL_COLS   = ['proto', 'state', 'service']
IP_COLS        = ['srcip', 'dstip']
TIMESTAMP_COLS = ['Stime', 'Ltime']
ID_COLS        = ['id', 'Id', 'ID']   # row index — NEVER a feature, see _resolve_id_col
BINARY_COLS    = ['is_sm_ips_ports', 'is_ftp_login']
TARGET_COL     = 'label'           # resolved dynamically — see _resolve_label_col
ATTACK_CAT_COL = 'attack_cat'

# Columns to drop after feature engineering
DROP_COLS = IP_COLS + TIMESTAMP_COLS + [ATTACK_CAT_COL]


def _resolve_label_col(df: pd.DataFrame) -> str:
    """Return the actual label column name regardless of capitalisation."""
    for candidate in ['Label', 'label', 'LABEL', 'class', 'Class']:
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if col.strip().lower() == 'label':
            return col
    return None


def _resolve_id_col(df: pd.DataFrame) -> str:
    """
    Return the actual row-id column name regardless of capitalisation.

    UNSW-NB15's train/test CSVs are NOT shuffled — rows are grouped in
    contiguous blocks by attack category. That makes the bare row index
    ('id') spuriously predictive of the label *within a single file*
    (train's id range is 1..175341, test's is a separate 1..82332), but it
    carries zero real signal and does not transfer between files. Leaving
    it in the feature set causes the model to look excellent on
    train/validation and collapse on the real test set. It must always be
    dropped before feature scaling/selection.
    """
    for candidate in ID_COLS:
        if candidate in df.columns:
            return candidate
    for col in df.columns:
        if col.strip().lower() == 'id':
            return col
    return None


def _resolve_attack_cat_col(df: pd.DataFrame) -> str:
    for candidate in ['attack_cat', 'Attack_cat', 'Attack_Cat', 'attackcat']:
        if candidate in df.columns:
            return candidate
    return None


def _hash_ip(ip: str) -> int:
    """Convert an IP address string to a compact integer hash bucket (0-999)."""
    try:
        h = int(hashlib.md5(str(ip).encode()).hexdigest(), 16)
        return h % 1000
    except Exception:
        return 0


def _encode_ips(df: pd.DataFrame) -> pd.DataFrame:
    """Hash-encode srcip and dstip into integer features."""
    for col in IP_COLS:
        if col in df.columns:
            df[f'{col}_hash'] = df[col].astype(str).apply(_hash_ip)
    return df


def _encode_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive two temporal features:
      - flow_duration  : Ltime − Stime  (seconds)
      - inter_arrival  : same as Sintpkt when available, else 0
    """
    if 'Stime' in df.columns and 'Ltime' in df.columns:
        df['flow_duration'] = (
            pd.to_numeric(df['Ltime'], errors='coerce') -
            pd.to_numeric(df['Stime'], errors='coerce')
        ).fillna(0).clip(lower=0)
    else:
        df['flow_duration'] = 0
    return df


def _handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      - numeric columns  → median imputation
      - object  columns  → 'unknown'
    """
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == 'string':
            df[col] = df[col].fillna('unknown')
        else:
            try:
                median_val = pd.to_numeric(df[col], errors='coerce').median()
                df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
            except Exception:
                df[col] = df[col].fillna('unknown')
    return df


def _one_hot_encode(df: pd.DataFrame, fit_categories: dict = None) -> tuple:
    """
    One-hot encode nominal columns.
    Returns (transformed_df, categories_dict).
    categories_dict maps column → list of known categories (for test alignment).
    """
    categories = {}
    dummies_list = []

    for col in NOMINAL_COLS:
        if col not in df.columns:
            continue

        if fit_categories and col in fit_categories:
            cats = fit_categories[col]
        else:
            cats = sorted(df[col].unique().tolist())

        categories[col] = cats
        dummies = pd.get_dummies(
            df[col].astype(str),
            prefix=col
        ).reindex(columns=[f'{col}_{c}' for c in cats], fill_value=0)
        dummies_list.append(dummies)

    df = df.drop(columns=[c for c in NOMINAL_COLS if c in df.columns])
    if dummies_list:
        df = pd.concat([df] + dummies_list, axis=1)
    return df, categories


def _normalize(df: pd.DataFrame,
               scaler: StandardScaler = None,
               feature_cols: list = None) -> tuple:
    """Standard-scale numeric columns. Returns (df, scaler, feature_cols)."""
    if feature_cols is None:
        # Exclude any plausible label column name
        exclude = {'label', 'Label', 'LABEL', 'class', 'Class'}
        feature_cols = [
            c for c in df.columns
            if c.strip().lower() not in {e.lower() for e in exclude}
            and df[c].dtype != object
        ]

    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        # align columns between train and test
        missing = [c for c in feature_cols if c not in df.columns]
        for m in missing:
            df[m] = 0
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler, feature_cols


# ── Public API ───────────────────────────────────────────────────────────────

class Preprocessor:
    """
    Stateful preprocessor: fit on training data, transform both train and test.

    Attributes
    ----------
    scaler       : fitted StandardScaler
    feature_cols : list of numeric feature column names used in scaling
    categories   : dict mapping nominal column → list of categories
    label_encoder: LabelEncoder for attack_cat (kept separate)
    """

    def __init__(self):
        self.scaler        = None
        self.feature_cols  = None
        self.categories    = None

    # ── internal helpers ─────────────────────────────────────────────────────

    def _base_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Normalise column names: strip whitespace
        df.columns = [c.strip() for c in df.columns]
        # Resolve and rename label column to canonical TARGET_COL
        lbl = _resolve_label_col(df)
        if lbl and lbl != TARGET_COL:
            df = df.rename(columns={lbl: TARGET_COL})
        # Resolve attack_cat column
        atk = _resolve_attack_cat_col(df)
        # Resolve row-id column — must be dropped before any feature is built
        rid = _resolve_id_col(df)
        df = _handle_missing(df)
        df = _encode_ips(df)
        df = _encode_timestamps(df)
        # Build drop list dynamically
        drop_list = list(DROP_COLS)
        if atk and atk not in drop_list:
            drop_list.append(atk)
        if rid and rid not in drop_list:
            drop_list.append(rid)
        df = df.drop(columns=[c for c in drop_list if c in df.columns], errors='ignore')
        return df

    # ── public methods ───────────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and return transformed features + label."""
        df = self._base_transform(df)
        df, self.categories = _one_hot_encode(df, fit_categories=None)

        # separate label before scaling
        y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None
        df_feat = df.drop(columns=[TARGET_COL], errors='ignore')
        df_feat = df_feat.apply(pd.to_numeric, errors='coerce').fillna(0)

        df_feat, self.scaler, self.feature_cols = _normalize(df_feat)

        if y is not None:
            df_feat[TARGET_COL] = y.values
        return df_feat

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted parameters."""
        df = self._base_transform(df)
        df, _ = _one_hot_encode(df, fit_categories=self.categories)

        y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None
        df_feat = df.drop(columns=[TARGET_COL], errors='ignore')
        df_feat = df_feat.apply(pd.to_numeric, errors='coerce').fillna(0)

        df_feat, self.scaler, self.feature_cols = _normalize(
            df_feat, scaler=self.scaler, feature_cols=self.feature_cols
        )

        if y is not None:
            df_feat[TARGET_COL] = y.values
        return df_feat

    def get_X_y(self, df: pd.DataFrame) -> tuple:
        """Extract feature matrix X and label vector y from processed df."""
        y = df[TARGET_COL].values.astype(int) if TARGET_COL in df.columns else None
        X = df.drop(columns=[TARGET_COL], errors='ignore').values.astype(np.float64)
        return X, y


# ── Feature Selection ────────────────────────────────────────────────────────

from sklearn.feature_selection import SelectKBest, f_classif


class FeatureSelector:
    """
    Selects the top-k most informative features using ANOVA F-score.
    Fitted on training data; transforms both train and test.

    Parameters
    ----------
    k : int or 'all'
        Number of features to keep.
    """

    def __init__(self, k: int = 60):
        self.k        = k
        self.selector = None
        self.selected_cols_ = None

    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                      feature_names: list = None) -> np.ndarray:
        self.selector = SelectKBest(f_classif, k=min(self.k, X.shape[1]))
        X_sel = self.selector.fit_transform(X, y)
        if feature_names is not None:
            mask = self.selector.get_support()
            self.selected_cols_ = [feature_names[i] for i, m in enumerate(mask) if m]
        return X_sel

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.selector.transform(X)

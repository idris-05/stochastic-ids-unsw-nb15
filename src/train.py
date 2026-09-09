"""
train.py – End-to-end training pipeline for the Stochastic IDS.

Architecture: v2 base (full 196 features, no selection) +
              stat column fix + corrected optimizer direction.
"""

import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import Preprocessor
from markov import MarkovChainModel
from anomaly import StatisticalAnomalyModel
from bayesian import BayesianSignatureModel
from classifier import LogisticRegressionFusion, build_fusion_features
from optimizer import StochasticOptimizer, OptParams

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "UNSW_NB15_training-set.csv")
TEST_FILE = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")
LR_EPOCHS = 150
VAL_FRACTION = 0.15
OPT_N_ITER = 300


def load_data(path):
    print(f"  Loading: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    print(f"  Shape: {df.shape}")
    return df


def save_model(obj, name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")


def train(train_path=TRAIN_FILE, test_path=TEST_FILE):
    print("=" * 60)
    print("STOCHASTIC IDS – TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading data …")
    df_train = load_data(train_path)
    df_test = load_data(test_path)

    # ── 2. Preprocessing (full features, no selection) ───────────────────
    print("\n[2/7] Preprocessing …")
    preprocessor = Preprocessor()
    df_train_proc = preprocessor.fit_transform(df_train)
    df_test_proc = preprocessor.transform(df_test)

    X_train, y_train = preprocessor.get_X_y(df_train_proc)
    X_test, y_test = preprocessor.get_X_y(df_test_proc)

    if y_train is None:
        raise ValueError(f"Label column not found. Columns: {list(df_train.columns)}")

    p_train = float(y_train.mean())
    p_test = float(y_test.mean())
    print(f"  Train features: {X_train.shape}, Test features: {X_test.shape}")
    print(f"  Train attack rate: {p_train:.3f}  |  Test attack rate: {p_test:.3f}")

    # Stratified validation split from training data
    idx_tr, idx_val = train_test_split(
        np.arange(len(X_train)),
        test_size=VAL_FRACTION,
        stratify=y_train,
        random_state=42,
    )
    p_val = float(y_train[idx_val].mean())
    print(f"  Val split: {len(idx_val)} samples, attack rate: {p_val:.3f}")

    # ── 3. Markov ────────────────────────────────────────────────────────
    print("\n[3/7] Markov Chain State Modeling …")
    markov = MarkovChainModel(smoothing=1e-4, anomaly_percentile=10.0)
    markov.fit(df_train)
    markov_train = markov.score(df_train)
    markov_test = markov.score(df_test)
    print(
        f"  Markov score  — mean: {markov_train.mean():.4f}, "
        f"std: {markov_train.std():.4f}"
    )

    # ── 4. Statistical (with column-name fix) ────────────────────────────
    print("\n[4/7] Statistical + Queue Anomaly Detection …")
    stat_model = StatisticalAnomalyModel(contamination=0.15)
    stat_model.fit(df_train, y_train)
    stat_train = stat_model.score(df_train)
    stat_test = stat_model.score(df_test)
    print(
        f"  Stat score    — mean: {stat_train.mean():.4f}, "
        f"std: {stat_train.std():.4f}"
    )

    # ── 5. Bayesian ──────────────────────────────────────────────────────
    print("\n[5/7] Bayesian Signature Matching …")
    bayes_model = BayesianSignatureModel(smoothing=1.0)
    bayes_model.fit(df_train, y_train)
    bayes_train = bayes_model.score(df_train)
    bayes_test = bayes_model.score(df_test)
    print(
        f"  Bayes score   — mean: {bayes_train.mean():.4f}, "
        f"std: {bayes_train.std():.4f}"
    )

    # ── 6. Logistic Regression Fusion (full 196+3 features) ──────────────
    print("\n[6/7] Logistic Regression Fusion …")
    X_fused_train = build_fusion_features(
        X_train, markov_train, stat_train, bayes_train
    )
    X_fused_test = build_fusion_features(X_test, markov_test, stat_test, bayes_test)
    print(f"  Fused feature matrix shape: {X_fused_train.shape}")

    X_tr = X_fused_train[idx_tr]
    y_tr = y_train[idx_tr]
    X_val = X_fused_train[idx_val]
    y_val = y_train[idx_val]

    fusion = LogisticRegressionFusion(
        lr=0.005,
        n_epochs=LR_EPOCHS,
        batch_size=1024,
        lambda_reg=0.01,
        class_weight="balanced",
    )
    fusion.fit(X_tr, y_tr)
    print(f"  LR training complete. Final loss: {fusion.loss_history_[-1]:.4f}")

    y_val_pred = fusion.predict(X_val, threshold=0.5)
    print(
        f"  Val F1 (threshold=0.5): "
        f"{f1_score(y_val, y_val_pred, zero_division=0):.4f}  "
        f"Acc: {accuracy_score(y_val, y_val_pred):.4f}"
    )

    # ── 6b. Bayesian parameter update ────────────────────────────────────
    print("  Bayesian parameter update on val set …")
    markov.bayesian_update(df_train.iloc[idx_val])
    stat_model.bayesian_update(df_train.iloc[idx_val])
    bayes_model.bayesian_update(df_train.iloc[idx_val], y_val)

    # ── 7. Stochastic Optimization ───────────────────────────────────────
    print("\n[7/7] Stochastic Optimization …")
    print(f"  Val attack rate: {p_val:.3f}")
    init_params = OptParams(
        lr_weights=fusion.weights_.copy(),
        lr_bias=fusion.bias_,
        threshold=0.5,
        stat_threshold=float(stat_model._threshold or 0.5),
        bayes_smooth=1.0,
        markov_smooth=1e-4,
    )

    # Threshold search runs on X_val/y_val only.
    optimizer = StochasticOptimizer(
        n_iter=OPT_N_ITER,
        sigma_init=0.005,
        sigma_thresh=0.04,
        decay=0.997,
        fp_weight=0.3,
        random_seed=42,
    )
    best_params = optimizer.optimize(fusion, X_val, y_val, init_params)
    conv = optimizer.convergence_summary()

    print(f"  Objective BEFORE optimization: {conv['initial_obj']:.4f}")
    print(f"  Objective AFTER  optimization: {conv['best_obj']:.4f}")
    print(f"  Improvement:                   {conv['improvement']:+.4f}")
    print(f"  Optimal threshold: {best_params.threshold:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    print("\nSaving models …")
    components = {
        "preprocessor": preprocessor,
        "markov": markov,
        "stat_model": stat_model,
        "bayes_model": bayes_model,
        "fusion": fusion,
        "optimizer": optimizer,
        "best_params": best_params,
        "opt_threshold": best_params.threshold,
        "feature_cols": preprocessor.feature_cols,
        "p_train": p_train,
        "p_val": p_val,
        "p_test": p_test,
    }
    save_model(components, "ids_components")
    np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_fused_test)
    np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(MODEL_DIR, "X_train.npy"), X_fused_train)
    np.save(os.path.join(MODEL_DIR, "y_train.npy"), y_train)
    print("\nTraining pipeline complete.")
    return components


if __name__ == "__main__":
    train()

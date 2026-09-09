"""
evaluate.py – Evaluation utilities for the Stochastic IDS.

Computes:
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix
  - Before vs after optimization comparison
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')


# ── Metric helpers ────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str = '') -> dict:
    """Compute and print standard classification metrics."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}Classification Metrics")
    print("  " + "-" * 40)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}  ({prec*100:.2f}%)")
    print(f"  Recall    : {rec:.4f}  ({rec*100:.2f}%)")
    print(f"  F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  Confusion Matrix:\n{cm}")

    return {
        'label'    : label,
        'accuracy' : acc,
        'precision': prec,
        'recall'   : rec,
        'f1'       : f1,
        'confusion_matrix': cm,
    }


def compare_before_after(metrics_before: dict, metrics_after: dict) -> None:
    """Print a before vs after optimisation comparison table."""
    print("\n" + "=" * 60)
    print("BEFORE vs AFTER STOCHASTIC OPTIMIZATION")
    print("=" * 60)
    keys = ['accuracy', 'precision', 'recall', 'f1']
    header = f"{'Metric':<15} {'Before':>10} {'After':>10} {'Delta':>10}"
    print(header)
    print("-" * 50)
    for k in keys:
        before = metrics_before.get(k, 0.0)
        after  = metrics_after.get(k, 0.0)
        delta  = after - before
        sign   = '+' if delta >= 0 else ''
        print(f"  {k.capitalize():<13} {before*100:>9.2f}% {after*100:>9.2f}% {sign}{delta*100:>8.2f}%")
    print("=" * 60)


# ── Main evaluation routine ───────────────────────────────────────────────

def evaluate(model_path: str = None) -> dict:
    """
    Load saved model components and evaluate on test set.
    Shows metrics before and after stochastic optimization.
    """
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'ids_components.pkl')

    print("=" * 60)
    print("STOCHASTIC IDS – EVALUATION")
    print("=" * 60)

    # Load models
    with open(model_path, 'rb') as f:
        components = pickle.load(f)

    fusion        = components['fusion']
    opt_threshold = float(components['opt_threshold'])
    best_params   = components['best_params']

    # Load test data
    X_test = np.load(os.path.join(MODEL_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(MODEL_DIR, 'y_test.npy'))

    print(f"\nTest set: {X_test.shape[0]} samples, "
          f"attack rate: {y_test.mean():.3f}")

    # ── Before optimization (default threshold 0.5, original weights) ────
    # We need to temporarily set initial weights. Since we saved best_params
    # and they ARE the optimised weights, we approximate "before" with threshold=0.5.
    print("\n--- BEFORE Stochastic Optimization (threshold=0.5) ---")
    y_proba = fusion.predict_proba(X_test)
    y_pred_before = (y_proba >= 0.5).astype(int)
    metrics_before = compute_metrics(y_test, y_pred_before, label='Before Opt')

    # ── After optimization (optimised threshold) ─────────────────────────
    print("\n--- AFTER Stochastic Optimization "
          f"(threshold={opt_threshold:.4f}) ---")
    y_pred_after = (y_proba >= opt_threshold).astype(int)
    metrics_after = compute_metrics(y_test, y_pred_after, label='After Opt')

    # ── Comparison table ─────────────────────────────────────────────────
    compare_before_after(metrics_before, metrics_after)

    # ── Module-level score analysis ───────────────────────────────────────
    print("\n--- Score contribution analysis ---")
    print("  (Last 3 features of X_test are markov, stat, bayesian scores)")
    if X_test.shape[1] >= 3:
        score_names = ['markov_score', 'stat_score', 'bayes_score']
        for i, name in enumerate(score_names):
            col = X_test[:, -(3 - i)]
            print(f"  {name:20s}: mean={col.mean():.4f}, std={col.std():.4f}")

    return {'before': metrics_before, 'after': metrics_after}


if __name__ == '__main__':
    evaluate()

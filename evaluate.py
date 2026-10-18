"""
QKernel-Lite – Evaluation Metrics
===================================
ROC-AUC, balanced accuracy, F1.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from typing import Dict


def evaluate_decision_scores(
    y_true: np.ndarray,
    decision_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all metrics from decision function scores.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,) – binary labels {0, 1}
    decision_scores : np.ndarray of shape (n,) – SVM decision values

    Returns
    -------
    dict with auc, balanced_accuracy, f1
    """
    # AUC from raw decision scores (no threshold needed)
    try:
        auc = roc_auc_score(y_true, decision_scores)
    except ValueError:
        auc = 0.5  # single class

    # For balanced accuracy and F1, threshold at 0
    y_pred = (decision_scores > 0).astype(int)
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)

    return {
        "auc": float(auc),
        "balanced_accuracy": float(ba),
        "f1": float(f1),
    }


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all metrics from probability scores (for M1 CNN).

    Parameters
    ----------
    y_true : np.ndarray of shape (n,) – binary labels {0, 1}
    probabilities : np.ndarray of shape (n,) – P(class=1)
    threshold : float – classification threshold

    Returns
    -------
    dict with auc, balanced_accuracy, f1
    """
    try:
        auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        auc = 0.5

    y_pred = (probabilities >= threshold).astype(int)
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0.0)

    return {
        "auc": float(auc),
        "balanced_accuracy": float(ba),
        "f1": float(f1),
    }


def print_metrics(metrics: Dict[str, float], model_name: str = ""):
    """Pretty-print metrics."""
    prefix = f"[{model_name}] " if model_name else ""
    print(f"  {prefix}AUC={metrics['auc']:.4f}  "
          f"BalAcc={metrics['balanced_accuracy']:.4f}  "
          f"F1={metrics['f1']:.4f}")

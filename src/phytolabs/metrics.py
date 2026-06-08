"""Image-level evaluation metrics (implemented in NumPy).

All metrics here are for binary image-level classification (0 = healthy,
1 = diseased/rust). We deliberately do not report per-region metrics because the
datasets provide image-level labels only.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """2x2 confusion matrix [[TN, FP], [FN, TP]]."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney U) formulation."""
    y_true = np.asarray(y_true).astype(int).ravel()
    proba = np.asarray(proba, dtype=np.float64).ravel()
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(proba, kind="mergesort")
    ranks = np.empty(len(proba), dtype=np.float64)
    ranks[order] = np.arange(1, len(proba) + 1)
    # Average ranks for ties.
    _, inv, counts = np.unique(proba, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    sum_pos = float(np.sum(ranks[y_true == 1]))
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None) -> Dict[str, float]:
    """Accuracy, precision, recall, F1 (and AUC if probabilities are given)."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    report = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if proba is not None:
        report["roc_auc"] = roc_auc(y_true, proba)
    return report

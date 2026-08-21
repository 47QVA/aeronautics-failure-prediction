"""Evaluation metrics for the RUL regression and failure classification framings.

The RUL scoring function is the standard asymmetric PHM08 / C-MAPSS challenge
score (Saxena et al., 2008): let d = predicted - actual. Under-predicting RUL
(d < 0, a conservative early warning) is penalized gently; over-predicting RUL
(d > 0, telling an operator an engine has more life left than it does) is
penalized much more steeply. Lower total score is better.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def phm_score(y_true, y_pred) -> np.ndarray:
    d = np.asarray(y_pred) - np.asarray(y_true)
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def regression_report(y_true, y_pred) -> dict:
    scores = phm_score(y_true, y_pred)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "phm_score_sum": float(scores.sum()),
        "phm_score_mean": float(scores.mean()),
        "n_samples": len(y_true),
    }


def select_threshold(y_true, y_proba, thresholds=None) -> float:
    """Pick the probability threshold that maximizes F1 on the given (validation) set."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (np.asarray(y_proba) >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def classification_report_dict(y_true, y_pred, y_proba=None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    report = {
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": len(y_true),
        "n_positive": int(np.asarray(y_true).sum()),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        report["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return report


def report_to_frame(reports: dict[str, dict]) -> pd.DataFrame:
    """reports: {split_name: report_dict} -> tidy DataFrame for display."""
    return pd.DataFrame(reports).T

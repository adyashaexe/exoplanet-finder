"""Evaluation helpers. Classes are imbalanced, so PR-AUC and recall matter more than accuracy."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y, p)) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / max(len(y), 1)),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def per_kind(kinds: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """How each signal type is treated: fraction scored as 'planet-like' and mean probability."""
    out = {}
    for k in np.unique(kinds):
        m = kinds == k
        out[str(k)] = {
            "n": int(m.sum()),
            "mean_probability": float(p[m].mean()),
            "flagged_as_planet": float((p[m] >= threshold).mean()),
        }
    return out

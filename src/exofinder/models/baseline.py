"""Gradient-boosted baseline on BLS scalars plus a few light-curve shape statistics.

Fast, needs no torch, and gives the CNN a number to beat. The API falls back to
it automatically when no CNN checkpoint is available.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ..config import AUX_FEATURES

SHAPE_FEATURES = ("local_core_mean", "local_shoulder_mean", "local_vness", "global_scatter")
FEATURE_NAMES = AUX_FEATURES + SHAPE_FEATURES


def shape_features(local_view: np.ndarray, global_view: np.ndarray) -> np.ndarray:
    """Cheap descriptors of transit shape. Views are batched (N, L)."""
    n = local_view.shape[1]
    c = n // 2
    core = local_view[:, c - n // 20 : c + n // 20 + 1].mean(1)
    shoulder = np.concatenate(
        [local_view[:, c - n // 6 : c - n // 12], local_view[:, c + n // 12 : c + n // 6]], axis=1
    ).mean(1)
    m = global_view.shape[1]
    outside = np.concatenate(
        [global_view[:, : m // 2 - m // 40], global_view[:, m // 2 + m // 40 :]], axis=1
    )
    return np.stack([core, shoulder, shoulder - core, outside.std(1)], axis=1).astype(np.float32)


def feature_matrix(aux: np.ndarray, local_view: np.ndarray, global_view: np.ndarray) -> np.ndarray:
    return np.concatenate([aux, shape_features(local_view, global_view)], axis=1)


def fit(x: np.ndarray, y: np.ndarray, seed: int = 0) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=400,
        max_depth=5,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        class_weight="balanced",
        random_state=seed,
    )
    return clf.fit(x, y)


def save(path: str | Path, clf, threshold: float, metrics: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "threshold": threshold, "features": FEATURE_NAMES, "metrics": metrics}, path)


def load(path: str | Path) -> dict:
    return joblib.load(path)


def predict_proba(bundle: dict, aux, local_view, global_view) -> np.ndarray:
    x = feature_matrix(np.atleast_2d(aux), np.atleast_2d(local_view), np.atleast_2d(global_view))
    return bundle["clf"].predict_proba(x)[:, 1]

"""Competition metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from config import TARGET_LABELS


def mean_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
) -> float:
    """Mean ROC-AUC across labels (skips columns with a single class)."""
    labels = labels or TARGET_LABELS
    scores = []
    for i, name in enumerate(labels):
        y = y_true[:, i]
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            continue
        y = y[mask].astype(int)
        p = y_pred[mask, i]
        if len(np.unique(y)) < 2:
            continue
        scores.append(roc_auc_score(y, p))
    if not scores:
        raise ValueError("No valid AUC scores computed")
    return float(np.mean(scores))

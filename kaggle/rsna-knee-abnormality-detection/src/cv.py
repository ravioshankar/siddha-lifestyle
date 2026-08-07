"""Cross-validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import TARGET_LABELS
from src.metrics import mean_auc
from src.text_model import TextBaseline


def _stratify_column(y: np.ndarray) -> np.ndarray:
    """Any-positive multilabel stratification bucket."""
    return (np.nansum(y, axis=1) > 0).astype(int)


def cross_validate_text(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Run stratified CV on labeled studies with reports."""
    y = df[TARGET_LABELS].to_numpy(dtype=float)
    # Fill NaN with 0 for training within CV on labeled rows only
    y_train_full = np.nan_to_num(y, nan=0.0)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_scores = []
    indices = np.arange(len(df))

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(indices, _stratify_column(y_train_full)), start=1
    ):
        train = df.iloc[tr_idx]
        valid = df.iloc[va_idx]
        y_tr = train[TARGET_LABELS].to_numpy(dtype=float)
        y_va = valid[TARGET_LABELS].to_numpy(dtype=float)
        y_tr = np.nan_to_num(y_tr, nan=0.0)

        model = TextBaseline()
        model.fit(train["Report"], y_tr)
        preds = model.predict_proba(valid["Report"])

        # Evaluate only where validation labels exist
        score = mean_auc(y_va, preds)
        fold_scores.append(score)
        print(f"Fold {fold}: mean AUC = {score:.4f}")

    return {
        "fold_scores": fold_scores,
        "mean_auc": float(np.mean(fold_scores)),
        "std_auc": float(np.std(fold_scores)),
    }

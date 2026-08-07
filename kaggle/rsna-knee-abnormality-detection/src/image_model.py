"""Study-level 2D middle-slice image baseline (sklearn; Kaggle may use CNN)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR, TARGET_LABELS, TEST_SERIES_DIR, TRAIN_SERIES_DIR
from src.dicom import load_study_middle_preprocessed, select_preferred_series


def build_feature_matrix(
    study_uids: list[str],
    series_df: pd.DataFrame,
    root: Path,
    size: int = 64,
) -> np.ndarray:
    """Flatten preprocessed middle slices; missing series → zeros."""
    preferred = select_preferred_series(series_df)
    feats = []
    for uid in study_uids:
        series_uid = preferred.get(str(uid))
        if series_uid is None:
            feats.append(np.zeros(size * size, dtype=np.float32))
            continue
        try:
            img = load_study_middle_preprocessed(
                str(uid), str(series_uid), root=root, size=size
            )
            feats.append(img.reshape(-1))
        except FileNotFoundError:
            feats.append(np.zeros(size * size, dtype=np.float32))
    return np.stack(feats, axis=0)


class ImageBaseline:
    """OvR logistic regression on flattened middle-slice pixels (local baseline).

    On Kaggle GPU, replace with EfficientNet — see notebooks/03_kaggle_image_train.md.
    """

    def __init__(self, size: int = 64, C: float = 1.0, random_state: int = 42) -> None:
        self.size = size
        self.labels = TARGET_LABELS
        self.scaler = StandardScaler()
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ImageBaseline":
        Xs = self.scaler.fit_transform(X)
        y = np.nan_to_num(y, nan=0.0)
        self.classifier.fit(Xs, y)
        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        Xs = self.scaler.transform(X)
        return np.column_stack(
            [est.predict_proba(Xs)[:, 1] for est in self.classifier.estimators_]
        )

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR / "image_baseline.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "classifier": self.classifier,
                    "size": self.size,
                    "labels": self.labels,
                },
                f,
            )
        return path

    @classmethod
    def load(cls, path: Path) -> "ImageBaseline":
        with path.open("rb") as f:
            state = pickle.load(f)
        model = cls(size=state["size"])
        model.scaler = state["scaler"]
        model.classifier = state["classifier"]
        model.labels = state["labels"]
        model._fitted = True
        return model


def default_roots() -> tuple[Path, Path]:
    return TRAIN_SERIES_DIR, TEST_SERIES_DIR

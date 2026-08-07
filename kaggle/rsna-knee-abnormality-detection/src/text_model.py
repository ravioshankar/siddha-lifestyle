"""TF-IDF + logistic regression text baseline."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from config import MODELS_DIR, TARGET_LABELS


def _fill_report(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


class TextBaseline:
    """Multilabel classifier on radiology report text."""

    def __init__(
        self,
        max_features: int = 30_000,
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.labels = TARGET_LABELS
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                max_iter=500,
                class_weight="balanced",
                random_state=random_state,
            )
        )
        self._fitted = False

    def fit(self, reports: pd.Series, y: np.ndarray) -> "TextBaseline":
        X = self.vectorizer.fit_transform(_fill_report(reports))
        self.classifier.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, reports: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        X = self.vectorizer.transform(_fill_report(reports))
        probas = []
        for est in self.classifier.estimators_:
            if hasattr(est, "predict_proba"):
                probas.append(est.predict_proba(X)[:, 1])
            else:
                probas.append(est.decision_function(X))
        return np.column_stack(probas)

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR / "text_baseline.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "classifier": self.classifier,
                    "labels": self.labels,
                },
                f,
            )
        return path

    @classmethod
    def load(cls, path: Path) -> "TextBaseline":
        with path.open("rb") as f:
            state = pickle.load(f)
        model = cls()
        model.vectorizer = state["vectorizer"]
        model.classifier = state["classifier"]
        model.labels = state["labels"]
        model._fitted = True
        return model

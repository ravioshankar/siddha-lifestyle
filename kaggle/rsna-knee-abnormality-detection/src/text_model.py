"""TF-IDF text models: baseline OvR and masked per-label logistic regression."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from config import MODELS_DIR, TARGET_LABELS


def _fill_report(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


class TextBaseline:
    """Multilabel classifier on radiology report text (NaNs must be filled by caller)."""

    def __init__(
        self,
        max_features: int = 30_000,
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.labels = TARGET_LABELS
        self.max_features = max_features
        self.C = C
        self.random_state = random_state
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
                    "kind": "baseline",
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


class MaskedTextModel:
    """Per-label logistic regression trained only on non-NaN targets (masked BCE analogue).

    Optionally accepts sample weights (e.g. 1.0 expert, 0.5 weak).
    """

    def __init__(
        self,
        max_features: int = 40_000,
        C: float = 1.0,
        ngram_range: tuple[int, int] = (1, 3),
        random_state: int = 42,
    ) -> None:
        self.labels = TARGET_LABELS
        self.max_features = max_features
        self.C = C
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self.estimators_: list[LogisticRegression | None] = [None] * len(TARGET_LABELS)
        self.prior_: np.ndarray = np.full(len(TARGET_LABELS), 0.5)
        self._fitted = False

    def fit(
        self,
        reports: pd.Series,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "MaskedTextModel":
        """Fit; y may contain NaN. sample_weight shape (n, n_labels) optional."""
        X = self.vectorizer.fit_transform(_fill_report(reports))
        for j in range(y.shape[1]):
            mask = ~np.isnan(y[:, j])
            if mask.sum() == 0 or len(np.unique(y[mask, j])) < 2:
                self.estimators_[j] = None
                self.prior_[j] = float(np.nanmean(y[:, j])) if mask.sum() else 0.5
                if np.isnan(self.prior_[j]):
                    self.prior_[j] = 0.5
                continue
            w = None
            if sample_weight is not None:
                w = sample_weight[mask, j]
            est = LogisticRegression(
                C=self.C,
                max_iter=800,
                class_weight="balanced",
                random_state=self.random_state,
            )
            est.fit(X[mask], y[mask, j].astype(int), sample_weight=w)
            self.estimators_[j] = est
            self.prior_[j] = float(y[mask, j].mean())
        self._fitted = True
        return self

    def predict_proba(self, reports: pd.Series) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        X = self.vectorizer.transform(_fill_report(reports))
        cols = []
        for j, est in enumerate(self.estimators_):
            if est is None:
                cols.append(np.full(X.shape[0], self.prior_[j]))
            else:
                cols.append(est.predict_proba(X)[:, 1])
        return np.column_stack(cols)

    def save(self, path: Path | None = None) -> Path:
        path = path or MODELS_DIR / "text_masked.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "kind": "masked",
                    "vectorizer": self.vectorizer,
                    "estimators": self.estimators_,
                    "prior": self.prior_,
                    "labels": self.labels,
                    "params": {
                        "max_features": self.max_features,
                        "C": self.C,
                        "ngram_range": self.ngram_range,
                        "random_state": self.random_state,
                    },
                },
                f,
            )
        return path

    @classmethod
    def load(cls, path: Path) -> "MaskedTextModel":
        with path.open("rb") as f:
            state = pickle.load(f)
        params = state.get("params", {})
        model = cls(**{k: v for k, v in params.items() if k in ("max_features", "C", "ngram_range", "random_state")})
        model.vectorizer = state["vectorizer"]
        model.estimators_ = state["estimators"]
        model.prior_ = state["prior"]
        model.labels = state["labels"]
        model._fitted = True
        return model

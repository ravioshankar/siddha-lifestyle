# Kaggle submission notebook (Phase 4)

Upload this notebook to the competition (or copy cells into a new Notebook attached to **RSNA Knee Abnormality Detection**).

**Local counterpart:** `predict.py` writes `submissions/submission_text_baseline.csv`.

## Setup on Kaggle

1. Add competition data (Data → Add data → this competition).
2. Upload `models/text_baseline.pkl` as a Kaggle Dataset, **or** paste the small TF-IDF training cells below to train in-notebook from `train.csv`.
3. Set accelerator if needed (CPU is enough for TF-IDF).
4. Ensure the notebook writes `/kaggle/working/submission.csv`.

```python
# Cell 1 — paths
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

COMP = Path("/kaggle/input/rsna-knee-abnormality-detection")
# Fallbacks for nested layouts
if not (COMP / "train.csv").exists():
    candidates = list(COMP.glob("**/train.csv"))
    COMP = candidates[0].parent if candidates else COMP

TARGET_LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
```

```python
# Cell 2 — train text baseline on expert-labeled rows (or load uploaded pickle)
train = pd.read_csv(COMP / "train.csv")
label_cols = [c for c in TARGET_LABELS if c in train.columns]
mask = train[label_cols].notna().any(axis=1)
labeled = train.loc[mask].copy()
y = labeled[TARGET_LABELS].to_numpy(dtype=float)
y = np.nan_to_num(y, nan=0.0)

vectorizer = TfidfVectorizer(
    max_features=30_000, ngram_range=(1, 2), min_df=2,
    strip_accents="unicode", sublinear_tf=True,
)
clf = OneVsRestClassifier(
    LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)
)
X = vectorizer.fit_transform(labeled["Report"].fillna("").astype(str))
clf.fit(X, y)
```

```python
# Cell 3 — predict test (Report may be missing)
test = pd.read_csv(COMP / "test.csv")
sample = pd.read_csv(COMP / "sample_submission.csv")
if "Report" in test.columns:
    reports = test["Report"].fillna("").astype(str)
else:
    reports = pd.Series([""] * len(test), index=test.index)

Xt = vectorizer.transform(reports)
probas = np.column_stack([est.predict_proba(Xt)[:, 1] for est in clf.estimators_])

# If reports are empty, fall back to sample priors (0.5) so image models can replace later
if reports.str.strip().eq("").all():
    print("WARNING: all test reports empty — text model uninformative; use image/fusion.")
    probas = sample[TARGET_LABELS].to_numpy(dtype=float)

sub = sample.copy()
sub[TARGET_LABELS] = probas
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv", len(sub))
```

## After submit

Record public LB in `notes/score_log.md` next to Phase 3 CV (0.9707 on synthetic local data; re-run on real CSVs).

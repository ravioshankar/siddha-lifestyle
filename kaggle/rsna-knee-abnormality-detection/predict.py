"""Generate submission.csv from the text baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from config import MODELS_DIR, SUBMISSIONS_DIR, SUBMISSION_COLUMNS  # noqa: E402
from src.data import load_sample_submission, load_test  # noqa: E402
from src.text_model import MaskedTextModel, TextBaseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "text_baseline.pkl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=SUBMISSIONS_DIR / "submission_text_baseline.csv",
    )
    return p.parse_args()


def _load_any_text_model(path: Path):
    import pickle

    with path.open("rb") as f:
        state = pickle.load(f)
    if state.get("kind") == "masked":
        return MaskedTextModel.load(path)
    return TextBaseline.load(path)


def main() -> None:
    args = parse_args()
    test = load_test()
    sample = load_sample_submission()

    if "Report" in test.columns:
        reports = test["Report"]
    else:
        # Public test CSV may only have StudyInstanceUID; use empty reports
        reports = pd.Series([""] * len(test), index=test.index)

    model = _load_any_text_model(args.model)
    preds = model.predict_proba(reports)

    if reports.fillna("").astype(str).str.strip().eq("").all():
        print(
            "WARNING: all test reports empty — text probs are uninformative. "
            "Prefer fuse_predictions.py (image-dominant) for submission."
        )

    submission = sample.copy()
    submission[SUBMISSION_COLUMNS[1:]] = preds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission):,} rows to {args.output}")


if __name__ == "__main__":
    main()

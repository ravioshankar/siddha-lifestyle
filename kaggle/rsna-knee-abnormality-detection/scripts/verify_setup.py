#!/usr/bin/env python3
"""Verify environment, credentials, and CSV loaders (Phase 0 exit criteria)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

REQUIRED_CSVS = [
    "train.csv",
    "train_series.csv",
    "test.csv",
    "test_series.csv",
    "sample_submission.csv",
]


def main() -> int:
    print("=== Phase 0 setup verification ===")
    print(f"Project: {PROJECT}")

    venv_python = PROJECT / ".venv" / "bin" / "python"
    print(f"uv environment: {'OK' if venv_python.exists() else 'MISSING — run uv venv'}")

    access_token = Path.home() / ".kaggle" / "access_token"
    print(f"Kaggle access token: {'OK' if access_token.exists() else 'MISSING — optional until real download'}")

    from config import TARGET_LABELS, resolve_data_root

    root = resolve_data_root()
    print(f"data root: {root}")

    missing = [name for name in REQUIRED_CSVS if not (root / name).exists()]
    if missing:
        print(f"CSV status: MISSING {missing}")
        print("Fix: uv run bash scripts/download_csvs.sh  OR  uv run python scripts/make_sample_data.py")
        return 1

    from src.data import (
        load_sample_submission,
        load_test,
        load_test_series,
        load_train,
        load_train_series,
    )

    train = load_train()
    series = load_train_series()
    test = load_test()
    test_series = load_test_series()
    sample = load_sample_submission()

    print(f"train.csv rows: {len(train):,}")
    print(f"train_series.csv rows: {len(series):,}")
    print(f"test.csv rows: {len(test):,}")
    print(f"test_series.csv rows: {len(test_series):,}")
    print(f"sample_submission rows: {len(sample):,}")

    for col in ["StudyInstanceUID", "Report"]:
        if col not in train.columns:
            print(f"ERROR: train.csv missing column {col}")
            return 1

    label_cols = [c for c in TARGET_LABELS if c in train.columns]
    print(f"label columns present: {len(label_cols)}/{len(TARGET_LABELS)}")
    if len(label_cols) != len(TARGET_LABELS):
        print(f"ERROR: missing labels {[c for c in TARGET_LABELS if c not in train.columns]}")
        return 1

    print("=== Phase 0 PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

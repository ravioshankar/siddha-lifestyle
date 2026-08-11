#!/usr/bin/env python3
"""Phase 6: train study-level 2D middle-slice image baseline with study-level CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from config import MODELS_DIR, TARGET_LABELS, TRAIN_SERIES_DIR  # noqa: E402
from src.data import load_train, load_train_series  # noqa: E402
from src.image_model import ImageBaseline, build_feature_matrix  # noqa: E402
from src.labels import labeled_mask  # noqa: E402
from src.metrics import mean_auc  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--output", type=Path, default=MODELS_DIR / "image_baseline.pkl")
    p.add_argument("--results", type=Path, default=MODELS_DIR / "phase6_cv_results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train()
    series = load_train_series()
    labeled = train.loc[labeled_mask(train)].reset_index(drop=True)
    if labeled.empty:
        raise SystemExit("No labeled studies")

    print(f"Building image features for {len(labeled)} studies (size={args.size})...")
    uids = labeled["StudyInstanceUID"].astype(str).tolist()
    X = build_feature_matrix(uids, series, TRAIN_SERIES_DIR, size=args.size)
    y = labeled[TARGET_LABELS].to_numpy(dtype=float)
    missing = (X == 0).all(axis=1).sum()
    print(f"Feature matrix: {X.shape}; all-zero rows (missing series): {missing}")

    if missing == len(labeled):
        print("No image files found — generating sample slices...")
        import runpy

        runpy.run_path(str(PROJECT / "scripts" / "make_sample_dicoms.py"), run_name="__main__")
        X = build_feature_matrix(uids, series, TRAIN_SERIES_DIR, size=args.size)

    strat = (np.nansum(np.nan_to_num(y, nan=0.0), axis=1) > 0).astype(int)
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    fold_scores = []
    oof = np.full_like(y, np.nan, dtype=float)

    for fold, (tr, va) in enumerate(skf.split(X, strat), start=1):
        model = ImageBaseline(size=args.size)
        model.fit(X[tr], np.nan_to_num(y[tr], nan=0.0))
        preds = model.predict_proba(X[va])
        oof[va] = preds
        score = mean_auc(y[va], preds)
        fold_scores.append(score)
        print(f"Fold {fold}: mean AUC = {score:.4f}")

    mean_s = float(np.mean(fold_scores))
    std_s = float(np.std(fold_scores))
    print(f"Image CV mean AUC: {mean_s:.4f} (+/- {std_s:.4f})")

    model = ImageBaseline(size=args.size)
    model.fit(X, np.nan_to_num(y, nan=0.0))
    path = model.save(args.output)
    print(f"Saved {path}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / "image_oof_preds.npy", oof)
    labeled[["StudyInstanceUID"]].to_csv(MODELS_DIR / "image_oof_study_uids.csv", index=False)

    payload = {
        "mean_auc": mean_s,
        "std_auc": std_s,
        "fold_scores": fold_scores,
        "n_studies": len(labeled),
        "size": args.size,
    }
    args.results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.results}")


if __name__ == "__main__":
    main()

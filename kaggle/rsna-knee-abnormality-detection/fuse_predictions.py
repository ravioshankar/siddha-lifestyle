#!/usr/bin/env python3
"""Phase 7: late fusion of text + image OOF / test predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from config import (  # noqa: E402
    MODELS_DIR,
    SUBMISSIONS_DIR,
    SUBMISSION_COLUMNS,
    TARGET_LABELS,
    TEST_SERIES_DIR,
    TRAIN_SERIES_DIR,
)
from src.data import load_sample_submission, load_test, load_test_series, load_train, load_train_series  # noqa: E402
from src.image_model import ImageBaseline, build_feature_matrix  # noqa: E402
from src.labels import labeled_mask  # noqa: E402
from src.metrics import mean_auc  # noqa: E402
from src.text_model import MaskedTextModel, TextBaseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--method",
        choices=["average", "weighted"],
        default="average",
        help="average = 0.5/0.5; weighted uses --text-weight",
    )
    p.add_argument("--text-weight", type=float, default=0.4)
    p.add_argument("--text-model", type=Path, default=MODELS_DIR / "text_masked.pkl")
    p.add_argument("--image-model", type=Path, default=MODELS_DIR / "image_baseline.pkl")
    p.add_argument(
        "--output",
        type=Path,
        default=SUBMISSIONS_DIR / "submission_fusion.csv",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=MODELS_DIR / "phase7_fusion_results.json",
    )
    return p.parse_args()


def _load_text(path: Path):
    with path.open("rb") as f:
        import pickle

        state = pickle.load(f)
    kind = state.get("kind", "baseline")
    if kind == "masked":
        return MaskedTextModel.load(path)
    return TextBaseline.load(path)


def fuse(a: np.ndarray, b: np.ndarray, text_weight: float) -> np.ndarray:
    w = float(np.clip(text_weight, 0.0, 1.0))
    return w * a + (1.0 - w) * b


def evaluate_oof_fusion(text_weight: float) -> dict | None:
    """If OOF files exist on overlapping study UIDs, report fused CV AUC."""
    text_oof_path = MODELS_DIR / "text_oof_preds.npy"
    image_oof_path = MODELS_DIR / "image_oof_preds.npy"
    text_uids_path = MODELS_DIR / "text_oof_study_uids.csv"
    image_uids_path = MODELS_DIR / "image_oof_study_uids.csv"
    if not all(p.exists() for p in (text_oof_path, image_oof_path, text_uids_path, image_uids_path)):
        return None

    text_oof = np.load(text_oof_path)
    image_oof = np.load(image_oof_path)
    text_uids = pd.read_csv(text_uids_path)["StudyInstanceUID"].astype(str)
    image_uids = pd.read_csv(image_uids_path)["StudyInstanceUID"].astype(str)

    train = load_train()
    labeled = train.loc[labeled_mask(train)].copy()
    labeled["StudyInstanceUID"] = labeled["StudyInstanceUID"].astype(str)

    # Align on image OOF UIDs (expert-labeled); look up text OOF by UID
    text_map = {u: i for i, u in enumerate(text_uids)}
    rows_t, rows_i, y_rows = [], [], []
    for i, uid in enumerate(image_uids):
        if uid not in text_map:
            continue
        lab_row = labeled.loc[labeled["StudyInstanceUID"] == uid]
        if lab_row.empty:
            continue
        rows_i.append(image_oof[i])
        rows_t.append(text_oof[text_map[uid]])
        y_rows.append(lab_row.iloc[0][TARGET_LABELS].to_numpy(dtype=float))

    if not rows_t:
        return None
    t = np.stack(rows_t)
    im = np.stack(rows_i)
    y = np.stack(y_rows)
    fused = fuse(t, im, text_weight)
    score = mean_auc(y, fused)
    text_score = mean_auc(y, t)
    image_score = mean_auc(y, im)
    return {
        "n_overlap": len(rows_t),
        "text_oof_auc": float(text_score),
        "image_oof_auc": float(image_score),
        "fusion_oof_auc": float(score),
        "text_weight": text_weight,
    }


def main() -> None:
    args = parse_args()
    text_w = 0.5 if args.method == "average" else args.text_weight

    print("=== OOF fusion evaluation (labeled overlap) ===")
    oof_stats = evaluate_oof_fusion(text_w)
    if oof_stats:
        print(
            f"n={oof_stats['n_overlap']}  text={oof_stats['text_oof_auc']:.4f}  "
            f"image={oof_stats['image_oof_auc']:.4f}  "
            f"fusion={oof_stats['fusion_oof_auc']:.4f}  (text_w={text_w})"
        )
    else:
        print("OOF files missing or no overlap — skip CV fusion metric")

    print("\n=== Building test submission ===")
    test = load_test()
    sample = load_sample_submission()
    test_series = load_test_series()

    text_model = _load_text(args.text_model)
    if "Report" in test.columns:
        reports = test["Report"]
    else:
        reports = pd.Series([""] * len(test), index=test.index)
    text_preds = text_model.predict_proba(reports)

    image_model = ImageBaseline.load(args.image_model)
    uids = test["StudyInstanceUID"].astype(str).tolist()
    X = build_feature_matrix(uids, test_series, TEST_SERIES_DIR, size=image_model.size)
    image_preds = image_model.predict_proba(X)

    # If all reports empty, lean on image
    if reports.fillna("").astype(str).str.strip().eq("").all():
        print("All test reports empty — using text_weight=0.15 (image-dominant)")
        text_w = min(text_w, 0.15)

    fused = fuse(text_preds, image_preds, text_w)
    sub = sample.copy()
    sub[TARGET_LABELS] = fused
    # Ensure column order
    cols = [c for c in SUBMISSION_COLUMNS if c in sub.columns]
    sub = sub[cols]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(sub)} rows)")

    payload = {
        "method": args.method,
        "text_weight_used": text_w,
        "oof": oof_stats,
        "submission": str(args.output),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.results}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 5: weak-label QC + masked text model (should beat Phase 3 baseline CV)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from config import MODELS_DIR, TARGET_LABELS  # noqa: E402
from src.cv import cross_validate_masked_text, cross_validate_text  # noqa: E402
from src.data import load_train  # noqa: E402
from src.labels import (  # noqa: E402
    evaluate_weak_labels,
    labeled_mask,
    merge_expert_and_weak,
)
from src.text_model import MaskedTextModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--min-weak-precision", type=float, default=0.7)
    p.add_argument("--output", type=Path, default=MODELS_DIR / "text_masked.pkl")
    p.add_argument(
        "--results",
        type=Path,
        default=MODELS_DIR / "phase5_cv_results.json",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train()
    notes = PROJECT / "notes"
    notes.mkdir(parents=True, exist_ok=True)

    print("=== Weak-label QC (vs expert) ===")
    qc = evaluate_weak_labels(train)
    print(qc.to_string(index=False))
    qc_path = notes / "weak_label_qc.md"
    lines = [
        "# Weak-label QC",
        "",
        "| Label | N | Precision | Recall | Agreement |",
        "|-------|---|-----------|--------|-----------|",
    ]
    for _, row in qc.iterrows():
        lines.append(
            f"| {row['label']} | {int(row['n'])} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['agreement']:.3f} |"
        )
    qc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {qc_path}")

    # Phase 3 baseline on expert-only for comparison
    labeled = train.loc[labeled_mask(train)].copy()
    print("\n=== Phase 3 baseline CV (expert only, NaN→0) ===")
    baseline_stats = cross_validate_text(labeled, n_splits=args.cv)
    print(
        f"Baseline CV mean AUC: {baseline_stats['mean_auc']:.4f} "
        f"(+/- {baseline_stats['std_auc']:.4f})"
    )

    # Merged expert + high-precision weak labels
    merged, y, weights = merge_expert_and_weak(
        train, min_weak_precision=args.min_weak_precision, qc=qc
    )
    # Keep rows that have at least one label
    row_mask = ~np.isnan(y).all(axis=1)
    reports = merged.loc[row_mask, "Report"].reset_index(drop=True)
    y = y[row_mask]
    weights = weights[row_mask]
    print(
        f"\nTraining rows after weak merge: {len(reports):,} "
        f"(expert-labeled was {len(labeled):,})"
    )

    print("\n=== Phase 5 masked + weak CV ===")
    stats = cross_validate_masked_text(
        reports,
        y,
        sample_weight=weights,
        n_splits=args.cv,
        model_kwargs={"max_features": 40_000, "C": 1.0, "ngram_range": (1, 3)},
    )
    print(
        f"Phase 5 CV mean AUC: {stats['mean_auc']:.4f} "
        f"(+/- {stats['std_auc']:.4f})"
    )

    improved = stats["mean_auc"] >= baseline_stats["mean_auc"] - 1e-6
    # On partial labels, also accept if within noise but more data — require non-worse
    print(
        "Beat/match baseline:",
        "YES" if improved else "NO (investigate QC thresholds)",
        f"Δ={stats['mean_auc'] - baseline_stats['mean_auc']:+.4f}",
    )

    print("\nTraining final masked model on all merged labels...")
    model = MaskedTextModel(max_features=40_000, C=1.0, ngram_range=(1, 3))
    model.fit(reports, y, sample_weight=weights)
    path = model.save(args.output)
    print(f"Saved model to {path}")

    # Save OOF for fusion
    oof_path = MODELS_DIR / "text_oof_preds.npy"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(oof_path, stats["oof_preds"])
    uid_path = MODELS_DIR / "text_oof_study_uids.csv"
    merged.loc[row_mask, ["StudyInstanceUID"]].reset_index(drop=True).to_csv(
        uid_path, index=False
    )

    payload = {
        "baseline_mean_auc": baseline_stats["mean_auc"],
        "baseline_std_auc": baseline_stats["std_auc"],
        "phase5_mean_auc": stats["mean_auc"],
        "phase5_std_auc": stats["std_auc"],
        "phase5_fold_scores": stats["fold_scores"],
        "n_train_rows": int(len(reports)),
        "beat_baseline": bool(improved),
        "min_weak_precision": args.min_weak_precision,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.results}")


if __name__ == "__main__":
    main()

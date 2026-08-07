"""Train the report-text baseline on expert-labeled studies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from config import MODELS_DIR, TARGET_LABELS  # noqa: E402
from src.cv import cross_validate_text  # noqa: E402
from src.data import load_train  # noqa: E402
from src.labels import labeled_mask  # noqa: E402
from src.text_model import TextBaseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv", type=int, default=5, help="CV folds (0 to skip)")
    p.add_argument("--max-features", type=int, default=30_000)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument(
        "--output",
        type=Path,
        default=MODELS_DIR / "text_baseline.pkl",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train = load_train()
    mask = labeled_mask(train)
    labeled = train.loc[mask].copy()

    print(f"Total studies: {len(train):,}")
    print(f"Labeled studies: {len(labeled):,}")
    if len(labeled) == 0:
        raise SystemExit(
            "No labeled rows found. Download train.csv and ensure label columns exist."
        )

    per_label = labeled[TARGET_LABELS].notna().sum()
    print("\nLabels per class (non-null):")
    for label, count in per_label.items():
        print(f"  {label:20s} {count:,}")

    if args.cv > 1:
        print(f"\nRunning {args.cv}-fold CV on labeled studies...")
        stats = cross_validate_text(labeled, n_splits=args.cv)
        print(
            f"CV mean AUC: {stats['mean_auc']:.4f} "
            f"(+/- {stats['std_auc']:.4f})"
        )

    y = labeled[TARGET_LABELS].to_numpy(dtype=float)
    y = np.nan_to_num(y, nan=0.0)

    print("\nTraining final model on all labeled studies...")
    model = TextBaseline(max_features=args.max_features, C=args.C)
    model.fit(labeled["Report"], y)
    path = model.save(args.output)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    main()

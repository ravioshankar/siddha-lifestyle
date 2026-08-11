#!/usr/bin/env python3
"""Create a small synthetic DICOM/npy subset for local image-model development.

Writes middle-slice .npy files under data/raw/train_series/<study>/<series>/middle.npy
for labeled studies only (capped). Prefer real DICOMs via scripts/download_dicom_subset.sh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from config import RAW_DATA_DIR, TARGET_LABELS, TRAIN_SERIES_DIR  # noqa: E402
from src.data import load_train, load_train_series  # noqa: E402
from src.dicom import select_preferred_series  # noqa: E402
from src.labels import labeled_mask  # noqa: E402

RNG = np.random.default_rng(7)


def synth_slice(labels: dict[str, float], size: int = 128) -> np.ndarray:
    """Simple pattern so a linear image model can learn something on sample data."""
    yy, xx = np.mgrid[0:size, 0:size]
    img = RNG.normal(0.0, 0.3, size=(size, size)).astype(np.float32)
    # Encode positives as localized blobs
    centers = {
        "ACL": (size // 2, size // 3),
        "MCL": (size // 2, 2 * size // 3),
        "Medial Meniscus": (2 * size // 3, size // 3),
        "Lateral Meniscus": (2 * size // 3, 2 * size // 3),
        "Medial OA": (3 * size // 4, size // 4),
        "Lateral OA": (3 * size // 4, 3 * size // 4),
        "PF OA": (size // 4, size // 2),
        "Effusion": (size // 2, size // 2),
        "Synovitis": (size // 3, size // 2),
        "Baker's": (size - 10, size // 2),
        "Contusion": (size // 2, 10),
        "Fracture": (10, 10),
    }
    for lab, (cy, cx) in centers.items():
        if labels.get(lab, 0) == 1:
            img += 2.5 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 8**2))
    return img


def main(max_studies: int = 84) -> None:
    train = load_train()
    series = load_train_series()
    labeled = train.loc[labeled_mask(train)].head(max_studies)
    preferred = select_preferred_series(series)

    n_written = 0
    for _, row in labeled.iterrows():
        uid = str(row["StudyInstanceUID"])
        series_uid = preferred.get(uid)
        if series_uid is None:
            sub = series.loc[series["StudyInstanceUID"] == uid]
            if sub.empty:
                continue
            series_uid = str(sub.iloc[0]["SeriesInstanceUID"])
        labels = {lab: float(row[lab]) if pd.notna(row[lab]) else 0.0 for lab in TARGET_LABELS}
        arr = synth_slice(labels)
        out_dir = TRAIN_SERIES_DIR / uid / series_uid
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "middle.npy", arr)
        n_written += 1

    # Tiny test set mirrors
    test = pd.read_csv(RAW_DATA_DIR / "test.csv")
    test_series = pd.read_csv(RAW_DATA_DIR / "test_series.csv")
    test_pref = select_preferred_series(test_series)
    test_root = RAW_DATA_DIR / "test_series"
    for uid in test["StudyInstanceUID"].astype(str):
        series_uid = test_pref.get(uid)
        if series_uid is None:
            continue
        out_dir = test_root / uid / series_uid
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "middle.npy", RNG.normal(0, 0.3, size=(128, 128)).astype(np.float32))

    print(f"Wrote {n_written} train middle.npy volumes under {TRAIN_SERIES_DIR}")
    print(f"Wrote {len(test)} test placeholders under {test_root}")
    print("Replace with real DICOMs when ready: bash scripts/download_dicom_subset.sh")


if __name__ == "__main__":
    main()

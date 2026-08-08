#!/usr/bin/env python3
"""Create synthetic competition CSVs for local pipeline development.

Use when ~/.kaggle/access_token is not configured yet, or to smoke-test the
pipeline without downloading the full competition dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from config import RAW_DATA_DIR, TARGET_LABELS  # noqa: E402

RNG = np.random.default_rng(42)

# English report snippets keyed by finding (positive language).
POS_SNIPPETS = {
    "ACL": "There is a complete tear of the ACL.",
    "MCL": "Partial tear of the MCL is present.",
    "Medial Meniscus": "Horizontal tear of the medial meniscus.",
    "Lateral Meniscus": "Complex tear of the lateral meniscus.",
    "Medial OA": "Moderate medial compartment osteoarthritis.",
    "Lateral OA": "Mild lateral osteoarthritis with cartilage loss.",
    "PF OA": "Patellofemoral osteoarthritis is noted.",
    "Effusion": "Moderate joint effusion is present.",
    "Synovitis": "Synovitis with synovial thickening.",
    "Baker's": "A Baker's cyst is identified in the popliteal fossa.",
    "Contusion": "Bone contusion / bone marrow edema at the lateral femoral condyle.",
    "Fracture": "Nondisplaced fracture of the tibial plateau.",
}

NEG_SNIPPETS = {
    "ACL": "The ACL is intact.",
    "MCL": "No MCL tear.",
    "Medial Meniscus": "Medial meniscus is intact without tear.",
    "Lateral Meniscus": "Lateral meniscus unremarkable.",
    "Medial OA": "No medial osteoarthritis.",
    "Lateral OA": "No lateral OA.",
    "PF OA": "Patellofemoral joint is preserved.",
    "Effusion": "No significant joint effusion.",
    "Synovitis": "No synovitis.",
    "Baker's": "No Baker's cyst.",
    "Contusion": "No bone contusion.",
    "Fracture": "No fracture.",
}

# Multilingual flavour for a subset of rows.
OTHER_LANG_PREFIX = [
    "Informe: estudio de rodilla. ",
    "Compte rendu IRM du genou. ",
    "Befund Knie-MRT: ",
]


def _study_uid(i: int) -> str:
    return f"1.2.826.0.1.3680043.8.498.{i:012d}"


def _series_uid(study_i: int, series_i: int) -> str:
    return f"1.2.826.0.1.3680043.8.498.{study_i:012d}.{series_i}"


def _make_report(labels: dict[str, int], multilingual: bool) -> str:
    parts = []
    if multilingual:
        parts.append(OTHER_LANG_PREFIX[RNG.integers(0, len(OTHER_LANG_PREFIX))])
    parts.append("MRI of the knee demonstrates the following. ")
    for label in TARGET_LABELS:
        if labels[label] == 1:
            parts.append(POS_SNIPPETS[label] + " ")
        else:
            # Keep reports realistic: mention negatives sometimes
            if RNG.random() < 0.35:
                parts.append(NEG_SNIPPETS[label] + " ")
    parts.append("Impression: see findings above.")
    return "".join(parts)


def build_train(n_studies: int = 240, labeled_frac: float = 0.35) -> pd.DataFrame:
    rows = []
    n_labeled = int(n_studies * labeled_frac)
    for i in range(n_studies):
        # Correlated prevalence — rare findings less often positive
        base_rates = {
            "ACL": 0.22,
            "MCL": 0.12,
            "Medial Meniscus": 0.35,
            "Lateral Meniscus": 0.28,
            "Medial OA": 0.30,
            "Lateral OA": 0.18,
            "PF OA": 0.20,
            "Effusion": 0.40,
            "Synovitis": 0.15,
            "Baker's": 0.08,
            "Contusion": 0.14,
            "Fracture": 0.05,
        }
        labels = {lab: int(RNG.random() < rate) for lab, rate in base_rates.items()}
        multilingual = RNG.random() < 0.15
        report = _make_report(labels, multilingual)
        row = {
            "StudyInstanceUID": _study_uid(i),
            "PatientSex": RNG.choice(["M", "F"]),
            "Report": report,
        }
        if i < n_labeled:
            row.update(labels)
        else:
            row.update({lab: np.nan for lab in TARGET_LABELS})
        rows.append(row)
    return pd.DataFrame(rows)


def build_series(train: pd.DataFrame) -> pd.DataFrame:
    planes = ["Sagittal", "Coronal", "Axial"]
    rows = []
    for i, uid in enumerate(train["StudyInstanceUID"]):
        n_series = int(RNG.integers(2, 6))
        for s in range(n_series):
            plane = planes[s % len(planes)]
            rows.append(
                {
                    "StudyInstanceUID": uid,
                    "SeriesInstanceUID": _series_uid(i, s),
                    "Fluid_Sensitive": int(plane == "Sagittal" or RNG.random() < 0.4),
                    "Fat_Suppression": int(RNG.random() < 0.5),
                    "Anatomical_Plane": plane,
                }
            )
    return pd.DataFrame(rows)


def build_test(n_test: int = 40) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test CSVs often omit Report — match competition reality."""
    test_rows = []
    series_rows = []
    sub_rows = []
    for i in range(n_test):
        uid = _study_uid(10_000 + i)
        test_rows.append({"StudyInstanceUID": uid})
        # No Report column on purpose
        for s in range(3):
            series_rows.append(
                {
                    "StudyInstanceUID": uid,
                    "SeriesInstanceUID": _series_uid(10_000 + i, s),
                    "Fluid_Sensitive": int(s == 0),
                    "Fat_Suppression": 1,
                    "Anatomical_Plane": ["Sagittal", "Coronal", "Axial"][s],
                }
            )
        sub = {"StudyInstanceUID": uid}
        sub.update({lab: 0.5 for lab in TARGET_LABELS})
        sub_rows.append(sub)
    return pd.DataFrame(test_rows), pd.DataFrame(series_rows), pd.DataFrame(sub_rows)


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train = build_train()
    series = build_series(train)
    test, test_series, sample = build_test()

    train.to_csv(RAW_DATA_DIR / "train.csv", index=False)
    series.to_csv(RAW_DATA_DIR / "train_series.csv", index=False)
    test.to_csv(RAW_DATA_DIR / "test.csv", index=False)
    test_series.to_csv(RAW_DATA_DIR / "test_series.csv", index=False)
    sample.to_csv(RAW_DATA_DIR / "sample_submission.csv", index=False)

    print(f"Wrote synthetic CSVs to {RAW_DATA_DIR}")
    print(f"  train: {len(train)} studies ({train[TARGET_LABELS].notna().any(axis=1).sum()} labeled)")
    print(f"  train_series: {len(series)}")
    print(f"  test: {len(test)} (no Report column — mirrors competition)")
    print("Replace with real data via: uv run bash scripts/download_csvs.sh")


if __name__ == "__main__":
    main()

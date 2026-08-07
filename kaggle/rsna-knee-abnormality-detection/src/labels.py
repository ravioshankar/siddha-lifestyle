"""Label utilities: labeled subset detection and report keyword weak labels."""

import re

import numpy as np
import pandas as pd

from config import TARGET_LABELS

# English keywords for weak supervision from radiology reports.
# Extend with multilingual patterns as you analyze the data.
REPORT_KEYWORDS: dict[str, list[str]] = {
    "ACL": [r"\bacl\b", r"anterior cruciate", r"cruciate ligament"],
    "MCL": [r"\bmcl\b", r"medial collateral"],
    "Medial Meniscus": [r"medial meniscus", r"medial meniscal", r"medial tear"],
    "Lateral Meniscus": [r"lateral meniscus", r"lateral meniscal", r"lateral tear"],
    "Medial OA": [r"medial osteoarthritis", r"medial oa\b", r"medial compartment.*(?:oa|osteoarthritis|degenerative)"],
    "Lateral OA": [r"lateral osteoarthritis", r"lateral oa\b", r"lateral compartment.*(?:oa|osteoarthritis|degenerative)"],
    "PF OA": [r"patellofemoral", r"\bpf oa\b", r"pf joint.*(?:oa|osteoarthritis|degenerative)"],
    "Effusion": [r"effusion", r"joint fluid", r"fluid in (?:the )?joint"],
    "Synovitis": [r"synovitis", r"synovial (?:thickening|inflammation)"],
    "Baker's": [r"baker'?s cyst", r"popliteal cyst"],
    "Contusion": [r"contusion", r"bone bruise", r"bone marrow edema"],
    "Fracture": [r"fracture", r"fractured", r"break in (?:the )?(?:bone|tibia|femur|patella)"],
}

NEGATION = re.compile(
    r"\b(no|without|absence of|negative for|not seen|unremarkable|intact|normal)\b",
    re.I,
)


def labeled_mask(df: pd.DataFrame) -> pd.Series:
    """True for studies with at least one expert label present."""
    label_cols = [c for c in TARGET_LABELS if c in df.columns]
    if not label_cols:
        return pd.Series(False, index=df.index)
    return df[label_cols].notna().any(axis=1)


def get_label_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return (n, 12) float array; NaN where label unknown."""
    return df[TARGET_LABELS].to_numpy(dtype=float)


def weak_labels_from_report(report: str) -> dict[str, float]:
    """Heuristic 0/1 labels from report text. Use for EDA / pseudo-labeling only."""
    if not isinstance(report, str) or not report.strip():
        return {label: np.nan for label in TARGET_LABELS}

    text = report.lower()
    out: dict[str, float] = {}
    for label, patterns in REPORT_KEYWORDS.items():
        found = False
        for pat in patterns:
            for match in re.finditer(pat, text, re.I):
                start = max(0, match.start() - 40)
                window = text[start : match.start()]
                if NEGATION.search(window):
                    continue
                found = True
                break
            if found:
                break
        out[label] = 1.0 if found else 0.0
    return out


def add_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add weak_* columns from Report text."""
    weak = df["Report"].apply(weak_labels_from_report).apply(pd.Series)
    weak = weak.add_prefix("weak_")
    return pd.concat([df, weak], axis=1)

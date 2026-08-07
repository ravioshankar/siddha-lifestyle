"""Label utilities: labeled subset, weak keywords, QC, and pseudo-label merge."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from config import TARGET_LABELS

# English + multilingual keywords for weak supervision from radiology reports.
REPORT_KEYWORDS: dict[str, list[str]] = {
    "ACL": [
        r"\bacl\b",
        r"anterior cruciate",
        r"cruciate ligament",
        r"ligamento cruzado anterior",
        r"ligament crois[eé] ant[eé]rieur",
        r"vorderes kreuzband",
    ],
    "MCL": [
        r"\bmcl\b",
        r"medial collateral",
        r"ligamento colateral medial",
        r"ligament collat[eé]ral m[eé]dial",
        r"innenband",
    ],
    "Medial Meniscus": [
        r"medial meniscus",
        r"medial meniscal",
        r"medial tear",
        r"menisco medial",
        r"m[eé]nisque m[eé]dial",
        r"innenmeniskus",
    ],
    "Lateral Meniscus": [
        r"lateral meniscus",
        r"lateral meniscal",
        r"lateral tear",
        r"menisco lateral",
        r"m[eé]nisque lat[eé]ral",
        r"aussenmeniskus|außenmeniskus",
    ],
    "Medial OA": [
        r"medial osteoarthritis",
        r"medial oa\b",
        r"medial compartment.*(?:oa|osteoarthritis|degenerative)",
        r"artrosis medial",
        r"arthrose m[eé]diale",
    ],
    "Lateral OA": [
        r"lateral osteoarthritis",
        r"lateral oa\b",
        r"lateral compartment.*(?:oa|osteoarthritis|degenerative)",
        r"artrosis lateral",
        r"arthrose lat[eé]rale",
    ],
    "PF OA": [
        r"patellofemoral",
        r"\bpf oa\b",
        r"pf joint.*(?:oa|osteoarthritis|degenerative)",
        r"femoropatellar",
        r"f[eé]moro-?patellaire",
    ],
    "Effusion": [
        r"effusion",
        r"joint fluid",
        r"fluid in (?:the )?joint",
        r"derrame",
        r"\bepanchement\b",
        r"gelenkerguss",
    ],
    "Synovitis": [
        r"synovitis",
        r"synovial (?:thickening|inflammation)",
        r"sinovitis",
        r"synovite",
    ],
    "Baker's": [
        r"baker'?s cyst",
        r"popliteal cyst",
        r"quiste de baker",
        r"kyste de baker",
        r"bakerzyste",
    ],
    "Contusion": [
        r"contusion",
        r"bone bruise",
        r"bone marrow edema",
        r"contusi[oó]n [oó]sea",
        r"contusion osseuse",
        r"knochenmark[oö]dem|bone bruise",
    ],
    "Fracture": [
        r"fracture",
        r"fractured",
        r"break in (?:the )?(?:bone|tibia|femur|patella)",
        r"fractura",
        r"fraktur",
    ],
}

NEGATION = re.compile(
    r"\b(no|without|absence of|negative for|not seen|unremarkable|intact|normal|"
    r"sin|sans|ohne|kein)\b",
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


def evaluate_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Precision / recall of keyword weak labels vs expert labels (labeled rows)."""
    labeled = df.loc[labeled_mask(df)].copy()
    if labeled.empty:
        return pd.DataFrame()
    weak_df = add_weak_labels(labeled)
    rows = []
    for lab in TARGET_LABELS:
        y = weak_df[lab]
        m = y.notna()
        if m.sum() == 0 or y[m].nunique() < 2:
            rows.append(
                {
                    "label": lab,
                    "n": int(m.sum()),
                    "precision": np.nan,
                    "recall": np.nan,
                    "agreement": np.nan,
                }
            )
            continue
        yt = y[m].astype(int)
        yp = weak_df.loc[m, f"weak_{lab}"].astype(int)
        rows.append(
            {
                "label": lab,
                "n": int(m.sum()),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "agreement": float((yt == yp).mean()),
            }
        )
    return pd.DataFrame(rows)


def merge_expert_and_weak(
    df: pd.DataFrame,
    *,
    min_weak_precision: float = 0.7,
    qc: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build training frame with expert labels preferred and high-precision weak fills.

    Returns
    -------
    frame : DataFrame with Report and TARGET_LABELS (may still contain NaN)
    y : (n, 12) float label matrix (NaN = missing)
    source_weight : (n, 12) weight matrix — 1.0 expert, 0.5 weak, 0 missing
    """
    frame = df.copy()
    if "Report" not in frame.columns:
        raise ValueError("Report column required for weak labels")
    weak_df = add_weak_labels(frame)
    if qc is None:
        qc = evaluate_weak_labels(frame)
    allow_weak = {
        row["label"]: (row["precision"] >= min_weak_precision)
        for _, row in qc.iterrows()
        if pd.notna(row.get("precision"))
    }

    y = np.full((len(frame), len(TARGET_LABELS)), np.nan, dtype=float)
    w = np.zeros((len(frame), len(TARGET_LABELS)), dtype=float)

    for j, lab in enumerate(TARGET_LABELS):
        expert = frame[lab].to_numpy(dtype=float) if lab in frame.columns else np.full(len(frame), np.nan)
        weak = weak_df[f"weak_{lab}"].to_numpy(dtype=float)
        for i in range(len(frame)):
            if not np.isnan(expert[i]):
                y[i, j] = expert[i]
                w[i, j] = 1.0
            elif allow_weak.get(lab, False) and not np.isnan(weak[i]):
                y[i, j] = weak[i]
                w[i, j] = 0.5
    out = frame[["StudyInstanceUID", "Report"]].copy() if "StudyInstanceUID" in frame.columns else frame[["Report"]].copy()
    for j, lab in enumerate(TARGET_LABELS):
        out[lab] = y[:, j]
    return out, y, w

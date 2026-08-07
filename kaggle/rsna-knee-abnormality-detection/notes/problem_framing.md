# Problem Framing — RSNA Knee Abnormality Detection

## Task

Multilabel binary classification: for each knee MRI **study**, predict a probability for each of **12 clinically important findings**.

| Label | Finding |
|-------|---------|
| ACL | Anterior cruciate ligament injury |
| MCL | Medial collateral ligament injury |
| Medial Meniscus | Medial meniscus tear |
| Lateral Meniscus | Lateral meniscus tear |
| Medial OA | Medial tibiofemoral osteoarthritis |
| Lateral OA | Lateral tibiofemoral osteoarthritis |
| PF OA | Patellofemoral osteoarthritis |
| Effusion | Joint effusion |
| Synovitis | Joint lining inflammation |
| Baker's | Baker's cyst |
| Contusion | Bone contusion / bruise |
| Fracture | Fracture |

## Metric

**Mean ROC-AUC** across the 12 targets (equal weight). Implementation: `src/metrics.py` → `mean_auc`.

- Columns with a single class in a fold are skipped.
- NaN labels are treated as **missing** at evaluation time (masked), not as negatives.

## Inputs

| Modality | Train | Test (expected) |
|----------|-------|-----------------|
| Radiology report text (`Report`) | Usually present (multilingual) | May be **absent** |
| MRI DICOM series | `train_series/<Study>/<Series>/*.dcm` | Hidden test series at scoring |
| Series metadata | plane, fluid-sensitive, fat suppression | Same schema |
| Patient sex | Available on train | May be limited |

## Labels

- Only a **subset** of training studies have expert per-condition labels.
- Remaining studies can be weakly labeled from reports (keywords / NLP) for semi-supervised learning.
- Partial label matrices: a row may have some labels filled and others NaN.

## Constraints

- **Code competition**: inference must run in a Kaggle Notebook on the platform.
- Dataset size if fully downloaded: ~570 GB DICOMs — use **CSV-first**, then a **small DICOM subset** locally; train heavy models on Kaggle GPU.
- Submission format: `StudyInstanceUID` + 12 probability columns (see `sample_submission.csv`).

## Leakage / validation rules

1. Split at **study** level (never put series from the same study in train and valid).
2. Do not use test reports if they appear only in public sample and not in hidden test.
3. Weak labels from reports must not be treated as equal to expert labels without QC.
4. Image model must stand alone — text may be unavailable at test.

## Success criteria (v1)

| Gate | Definition |
|------|------------|
| Phase 0 | CSVs load via `scripts/verify_setup.py` |
| Phase 3 | 5-fold CV mean AUC recorded for text baseline |
| Phase 4 | Accepted Kaggle submission (or local submission CSV when offline) |
| Later | Image CV + fusion beats text-only CV when reports missing |

## Critical insight

**Test may lack reports.** A text-only model will collapse toward chance on hidden test if `Report` is empty. An image (or multimodal with image fallback) path is required for a competitive score.

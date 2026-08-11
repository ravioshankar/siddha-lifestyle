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

| Modality | Train | Test |
|----------|-------|------|
| Radiology report text (`Report`) | Present for all 4,407 studies | **Absent** from `test.csv` |
| MRI DICOM series | `train_series/<Study>/<Series>/*.dcm` | Hidden test series at scoring |
| Series metadata | Plane, fluid-sensitive, fat suppression | Same schema |

The CSV schema contains no patient-sex field.

## Labels

- Only **58 / 4,407 studies (1.32%)** have expert labels for all 12 targets.
- The remaining **4,349 studies** have reports but no target labels.
- There are no partially labeled rows in the current training CSV.
- Remaining studies can be weakly labeled from reports (keywords / NLP) for semi-supervised learning.
- Keep label masking in the pipeline even though the current release is all-or-none; it remains safe if the schema changes.

## Verified CSV snapshot

_Checked from `data/raw` on 2026-08-07._

| File | Rows | Key observation |
|------|-----:|-----------------|
| `train.csv` | 4,407 | Unique study IDs; all reports present |
| `train_series.csv` | 24,371 | 3–14 series/study; median 5 |
| `test.csv` | 3 | Example test set; study ID only |
| `test_series.csv` | 15 | Every example test study has series metadata |
| `sample_submission.csv` | 3 | IDs and target order match `test.csv` |

All train/test studies have series metadata, with no orphan series rows. The three-row
test set is only the local example; Kaggle replaces it during code-competition scoring.

## Constraints

- **Code competition**: inference must run in a Kaggle Notebook on the platform.
- Dataset size if fully downloaded: ~570 GB DICOMs — use **CSV-first**, then a **small DICOM subset** locally; train heavy models on Kaggle GPU.
- Submission format: `StudyInstanceUID` + 12 probability columns (see `sample_submission.csv`).

## Leakage / validation rules

1. Split at **study** level (never put series from the same study in train and valid).
2. Do not use test reports if they appear only in public sample and not in hidden test.
3. Weak labels from reports must not be treated as equal to expert labels without QC.
4. Image model must stand alone — text may be unavailable at test.

## Modeling and validation decisions

1. **Unit of prediction:** one row per `StudyInstanceUID`.
2. **Primary validation set:** stratified study-level folds over the 58 expert-labeled studies.
3. **Text role:** weak-label generation and representation learning, not final-test inference.
4. **Image role:** primary inference modality because hidden test reports are unavailable.
5. **Model selection metric:** macro mean ROC-AUC over targets that contain both classes in a fold.
6. **Reproducibility:** persist fold assignments and out-of-fold predictions before fusion.
7. **Submission contract:** preserve sample-submission column order and clip probabilities to `[0, 1]`.

## Success criteria (v1)

| Gate | Definition |
|------|------------|
| Phase 0 | CSVs load via `scripts/verify_setup.py` |
| Phase 3 | 5-fold CV mean AUC recorded for text baseline |
| Phase 4 | Accepted Kaggle submission (or local submission CSV when offline) |
| Later | Image CV + fusion beats text-only CV when reports missing |

## Critical insight

**Test lacks reports.** A text-only model cannot provide study-specific hidden-test
predictions. Images must be the primary inference path; text can still help exploit
the 4,349 report-only training studies.

## Phase 1 exit checklist

- [x] Prediction unit and 12 targets documented
- [x] Metric and missing-label behavior documented
- [x] Train/test modalities and schema verified
- [x] Leakage boundary fixed at study level
- [x] Validation and submission contracts defined
- [x] Main risk identified: only 58 expert-labeled studies

# RSNA Knee Abnormality Detection

Multimodal ML challenge to detect 12 clinically important knee abnormalities from MRI scans and radiology reports.

- **Competition:** [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- **Metric:** Mean AUC across 12 multilabel targets
- **Type:** Code competition (submit via Kaggle Notebooks)
- **Dataset size:** ~570 GB, ~820k DICOM files
- **Deadline:** October 22, 2026

## Task

Predict per-study probabilities for 12 knee findings from multimodal data (DICOM MRI series + free-text radiology reports):

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

Only a **small subset** of training studies have per-condition labels. The rest can be labeled from the multilingual radiology reports (`Report` column).

## Dataset files

| File | Description |
|------|-------------|
| `train.csv` | One row per study: UID, sex, report text, 12 labels |
| `train_series.csv` | One row per MRI series: plane, fluid-sensitive, fat suppression |
| `train_series/` | DICOMs at `train_series/<StudyUID>/<SeriesUID>/*.dcm` |
| `test.csv` | ~1,300 test study IDs (example set for local dev) |
| `test_series.csv` | Series metadata for test studies |
| `test_series/` | Example test DICOMs (replaced during scoring) |
| `sample_submission.csv` | Template with all labels set to 0.5 |

**DICOM notes:** Mixed transfer syntaxes (JPEG lossless, JPEG 2000, etc.). Series typically have 20–45 slices (median 30). Intensity, orientation, and resolution vary across sites.
## Setup

### 1. Create the virtual environment with `uv`

```bash
cd kaggle/rsna-knee-abnormality-detection
uv venv
uv pip install -r requirements.txt
uv run kaggle --version
```

Use `uv run <command>` throughout the project; activating `.venv` is not required. These commands are intended for openSUSE and other Linux distributions.

### 2. Configure Kaggle API credentials

1. Sign in at [kaggle.com](https://www.kaggle.com) and accept the [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).
2. Go to **Settings → API → Generate New Token** and copy the token.
3. Save the token with owner-only permissions:

```bash
mkdir -p ~/.kaggle
printf '%s\n' 'YOUR_NEW_KAGGLE_TOKEN' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Never commit or paste the real token into documentation, source code, or chat.

### 3. Download data

**Start with CSVs only (~few MB)** — enough for text baseline + EDA:

```bash
uv run bash scripts/download_csvs.sh
```

**No Kaggle token yet?** Generate local sample CSVs (pipeline smoke test):

```bash
uv run python scripts/make_sample_data.py
uv run python scripts/verify_setup.py
```

Full DICOMs (~570 GB) — prefer Kaggle Notebooks; locally use a small subset:

```bash
bash scripts/download_dicom_subset.sh   # falls back to synthetic slices if no token
```

## Standard DS workflow (Phases 0–7)

Follow `notes/problem_framing.md` and log scores in `notes/score_log.md`.

```
CSVs → EDA → text baseline → Kaggle submit
         → weak labels + masked text → image subset → late fusion
```

**openSUSE / Linux quickstart (no Kaggle token yet — sample data):**

```bash
cd kaggle/rsna-knee-abnormality-detection
uv venv
uv pip install -r requirements.txt
uv run python scripts/make_sample_data.py          # or: uv run bash scripts/download_csvs.sh
uv run python scripts/verify_setup.py
uv run python scripts/run_eda.py                   # → notes/eda_summary.md
uv run python train_text_baseline.py --cv 5        # Phase 3
uv run python predict.py                           # Phase 4 local CSV
uv run python train_text_improved.py --cv 5        # Phase 5
uv run python scripts/make_sample_dicoms.py        # or download_dicom_subset.sh
uv run python train_image_baseline.py --cv 5       # Phase 6
uv run python fuse_predictions.py                  # Phase 7 → submissions/submission_fusion.csv
```

Submit on Kaggle using `notebooks/02_kaggle_submit.md` (text) and `notebooks/03_kaggle_image_train.md` (CNN).

### Project layout

```
rsna-knee-abnormality-detection/
├── config.py
├── train_text_baseline.py      # Phase 3
├── train_text_improved.py      # Phase 5
├── train_image_baseline.py     # Phase 6
├── fuse_predictions.py         # Phase 7
├── predict.py
├── notes/                      # framing, EDA, QC, score log
├── src/
│   ├── data.py, labels.py, metrics.py, cv.py
│   ├── text_model.py           # TextBaseline + MaskedTextModel
│   ├── image_model.py          # middle-slice sklearn baseline
│   └── dicom.py                # load / preprocess / prefer series
├── scripts/
│   ├── download_csvs.sh
│   ├── make_sample_data.py / make_sample_dicoms.py
│   ├── verify_setup.py / run_eda.py
│   └── download_dicom_subset.sh
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_kaggle_submit.md
│   └── 03_kaggle_image_train.md
├── data/raw/
├── models/
└── submissions/
```

## Next steps

Do these in order. Local Phases 0–7 already run on **synthetic** sample data; the goal now is real competition data and a leaderboard score.

### 1. Connect Kaggle and pull real CSVs

1. Accept the [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).
2. Create an API token and save it as `~/.kaggle/access_token` with permissions `600`.
3. Download CSVs only (do **not** pull the full ~570 GB DICOM dump yet):

```bash
uv run bash scripts/download_csvs.sh
uv run python scripts/verify_setup.py
```

### 2. Re-run the text path on real data

```bash
uv run python scripts/run_eda.py
uv run python train_text_baseline.py --cv 5          # Phase 3 — log CV in notes/score_log.md
uv run python predict.py                             # local submission CSV
uv run python train_text_improved.py --cv 5          # Phase 5 — must beat Phase 3 CV
```

### 3. First Kaggle submission (close the loop)

1. Open a new Notebook attached to this competition.
2. Follow [`notebooks/02_kaggle_submit.md`](notebooks/02_kaggle_submit.md) (train or upload the text model; write `/kaggle/working/submission.csv`).
3. Submit and record **public LB** next to local CV in [`notes/score_log.md`](notes/score_log.md).
4. Expect a CV→LB drop if hidden test has no `Report` — that is expected for text-only.

### 4. Image model (small subset / Kaggle GPU)

1. Prefer training on Kaggle GPU with a capped study list — see [`notebooks/03_kaggle_image_train.md`](notebooks/03_kaggle_image_train.md).
2. Locally you can smoke-test with `bash scripts/download_dicom_subset.sh` then `uv run python train_image_baseline.py --cv 5`.
3. Save study-level OOF predictions for fusion.

### 5. Late fusion and iterate

```bash
uv run python fuse_predictions.py                    # → submissions/submission_fusion.csv
```

1. Submit the fused (image-dominant when reports are empty) predictions via Notebook.
2. Compare CV vs LB in `notes/score_log.md`.
3. Only after that: more slices, better series selection, EfficientNet/3D, efficiency track.

### Checklist

| Done? | Step |
|-------|------|
| | Kaggle rules accepted + `~/.kaggle/access_token` installed |
| | Real CSVs downloaded; `verify_setup.py` passes |
| | Phase 3 CV logged on real data |
| | First Kaggle submission accepted; LB recorded |
| | Phase 5 beats Phase 3 on real data |
| | Image OOF on Kaggle; fusion submitted |

## Key dates

| Date | Milestone |
|------|-----------|
| Jul 30, 2026 | Start |
| Oct 15, 2026 | Entry & team merger deadline |
| Oct 22, 2026 | Final submission deadline |

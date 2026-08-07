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

### 1. Create and activate the virtual environment

**Windows — easiest (no PowerShell policy change):**

Use **Command Prompt** instead of PowerShell:

```cmd
cd kaggle\rsna-knee-abnormality-detection
.venv\Scripts\activate.bat
kaggle --version
```

Or skip activation and call tools directly:

```powershell
.\.venv\Scripts\kaggle.exe --version
.\scripts\download-data.bat
```

**Windows (PowerShell)** — if activation is blocked, allow scripts once for your user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
cd kaggle/rsna-knee-abnormality-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> The `kaggle` command only works after activating the venv (or using the full path to `.venv\Scripts\kaggle.exe` on Windows).

### 2. Configure Kaggle API credentials

1. Sign in at [kaggle.com](https://www.kaggle.com) and accept the [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).
2. Go to **Account → Create New Token** (downloads `kaggle.json`).
3. Place the file at:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - macOS / Linux: `~/.kaggle/kaggle.json`

### 3. Download data

**Start with CSVs only (~few MB)** — enough for text baseline + EDA:

```bash
# macOS / Linux
bash scripts/download_csvs.sh

# Windows
scripts\download-csvs.bat
```

**No Kaggle token yet?** Generate local sample CSVs (pipeline smoke test):

```bash
.venv/bin/python scripts/make_sample_data.py
.venv/bin/python scripts/verify_setup.py
```

Full DICOMs (~570 GB) — prefer Kaggle Notebooks; locally use a small subset:

```bash
bash scripts/download_dicom_subset.sh   # falls back to synthetic slices if no token
# Windows full dump (large): scripts\download-data.bat
```

## Standard DS workflow (Phases 0–7)

Follow `notes/problem_framing.md` and log scores in `notes/score_log.md`.

```
CSVs → EDA → text baseline → Kaggle submit
         → weak labels + masked text → image subset → late fusion
```

**macOS / Linux quickstart (no Kaggle token yet — sample data):**

```bash
cd kaggle/rsna-knee-abnormality-detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_sample_data.py          # or: bash scripts/download_csvs.sh
python scripts/verify_setup.py
python scripts/run_eda.py                   # → notes/eda_summary.md
python train_text_baseline.py --cv 5        # Phase 3
python predict.py                           # Phase 4 local CSV
python train_text_improved.py --cv 5        # Phase 5
python scripts/make_sample_dicoms.py        # or download_dicom_subset.sh
python train_image_baseline.py --cv 5       # Phase 6
python fuse_predictions.py                  # Phase 7 → submissions/submission_fusion.csv
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
│   ├── download_csvs.sh / download-csvs.bat
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
2. Create an API token and save it as `~/.kaggle/kaggle.json` (Windows: `%USERPROFILE%\.kaggle\kaggle.json`).
3. Download CSVs only (do **not** pull the full ~570 GB DICOM dump yet):

```bash
bash scripts/download_csvs.sh
# Windows: scripts\download-csvs.bat
python scripts/verify_setup.py
```

### 2. Re-run the text path on real data

```bash
python scripts/run_eda.py
python train_text_baseline.py --cv 5          # Phase 3 — log CV in notes/score_log.md
python predict.py                             # local submission CSV
python train_text_improved.py --cv 5          # Phase 5 — must beat Phase 3 CV
```

### 3. First Kaggle submission (close the loop)

1. Open a new Notebook attached to this competition.
2. Follow [`notebooks/02_kaggle_submit.md`](notebooks/02_kaggle_submit.md) (train or upload the text model; write `/kaggle/working/submission.csv`).
3. Submit and record **public LB** next to local CV in [`notes/score_log.md`](notes/score_log.md).
4. Expect a CV→LB drop if hidden test has no `Report` — that is expected for text-only.

### 4. Image model (small subset / Kaggle GPU)

1. Prefer training on Kaggle GPU with a capped study list — see [`notebooks/03_kaggle_image_train.md`](notebooks/03_kaggle_image_train.md).
2. Locally you can smoke-test with `bash scripts/download_dicom_subset.sh` then `python train_image_baseline.py --cv 5`.
3. Save study-level OOF predictions for fusion.

### 5. Late fusion and iterate

```bash
python fuse_predictions.py                    # → submissions/submission_fusion.csv
```

1. Submit the fused (image-dominant when reports are empty) predictions via Notebook.
2. Compare CV vs LB in `notes/score_log.md`.
3. Only after that: more slices, better series selection, EfficientNet/3D, efficiency track.

### Checklist

| Done? | Step |
|-------|------|
| | Kaggle rules accepted + `kaggle.json` installed |
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

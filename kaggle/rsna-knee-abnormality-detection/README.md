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

**Option A — batch script (recommended on Windows):**

```cmd
scripts\download-data.bat
```

**Option B — with venv activated:**

```powershell
kaggle competitions download -c rsna-knee-abnormality-detection -p data/raw
```

**Option C — without activating venv:**

```powershell
.\.venv\Scripts\kaggle.exe competitions download -c rsna-knee-abnormality-detection -p data/raw
```

Or place files manually under `data/raw/`.

**Start with CSVs only (~few MB)** — enough to train the text baseline:

```cmd
scripts\download-csvs.bat
```

Download full DICOMs (~570 GB) later when ready for image models:

```cmd
scripts\download-data.bat
```

## Build & run

### Pipeline overview

```
CSVs  →  EDA notebook  →  text baseline (TF-IDF + logistic regression)
                              ↓
DICOMs (later)  →  image model  →  multimodal fusion  →  submission
```

### 1. Explore the data

```cmd
.\.venv\Scripts\jupyter.exe notebook notebooks\01_eda.ipynb
```

### 2. Train text baseline

Trains on expert-labeled studies using the `Report` column. Runs 5-fold CV by default.

```cmd
scripts\train.bat
scripts\train.bat --cv 0          REM skip CV, train only
```

### 3. Generate submission

```cmd
scripts\predict.bat
REM output: submissions\submission_text_baseline.csv
```

### Project layout

```
rsna-knee-abnormality-detection/
├── config.py                 # paths, labels, constants
├── train_text_baseline.py    # train script
├── predict.py                # submission generator
├── src/
│   ├── data.py               # load CSVs
│   ├── labels.py             # labeled mask, report keyword weak labels
│   ├── metrics.py            # mean AUC
│   ├── text_model.py         # TF-IDF baseline
│   ├── cv.py                 # cross-validation
│   └── dicom.py              # DICOM loaders (for image stage)
├── scripts/
│   ├── download-csvs.bat     # fast CSV download
│   ├── download-data.bat     # full dataset
│   ├── train.bat
│   └── predict.bat
├── notebooks/01_eda.ipynb
├── data/raw/
├── models/
└── submissions/
```

## Next steps (image / multimodal)

1. Download `train_series/` DICOMs
2. Add a 2D/3D CNN on middle-slice or volume (`src/dicom.py` ready)
3. Fuse image + text predictions (average or learned stacking)
4. Use weak report labels (`src/labels.py`) to pseudo-label unlabeled studies
5. Submit via Kaggle Notebook for code competition scoring

## Key dates

| Date | Milestone |
|------|-----------|
| Jul 30, 2026 | Start |
| Oct 15, 2026 | Entry & team merger deadline |
| Oct 22, 2026 | Final submission deadline |

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_DIR / "models"
SUBMISSIONS_DIR = PROJECT_DIR / "submissions"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"
SRC_DIR = PROJECT_DIR / "src"

COMPETITION_SLUG = "rsna-knee-abnormality-detection"
COMPETITION_URL = f"https://www.kaggle.com/competitions/{COMPETITION_SLUG}"

# --- Labels (12 binary targets, mean AUC metric) ---

TARGET_LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

SUBMISSION_COLUMNS = ["StudyInstanceUID", *TARGET_LABELS]

# --- Dataset files (after download to data/raw/) ---

TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TRAIN_SERIES_CSV = RAW_DATA_DIR / "train_series.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
TEST_SERIES_CSV = RAW_DATA_DIR / "test_series.csv"
SAMPLE_SUBMISSION_CSV = RAW_DATA_DIR / "sample_submission.csv"

TRAIN_SERIES_DIR = RAW_DATA_DIR / "train_series"
TEST_SERIES_DIR = RAW_DATA_DIR / "test_series"


def resolve_data_root() -> Path:
    """Find CSV root whether files are flat or inside a nested folder."""
    if TRAIN_CSV.exists():
        return RAW_DATA_DIR
    for child in RAW_DATA_DIR.iterdir() if RAW_DATA_DIR.exists() else []:
        if child.is_dir() and (child / "train.csv").exists():
            return child
    return RAW_DATA_DIR


def data_path(name: str) -> Path:
    return resolve_data_root() / name

# train.csv columns
TRAIN_META_COLUMNS = ["StudyInstanceUID", "PatientSex", "Report"]
TRAIN_SERIES_COLUMNS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
]

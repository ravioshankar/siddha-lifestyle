"""Load competition CSV metadata."""

from pathlib import Path

import pandas as pd

from config import RAW_DATA_DIR, data_path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Download data first:\n"
            f"  .\\.venv\\Scripts\\kaggle.exe competitions download "
            f"-c rsna-knee-abnormality-detection -p {RAW_DATA_DIR.as_posix()}"
        )
    return pd.read_csv(path)


def load_train() -> pd.DataFrame:
    return _read_csv(data_path("train.csv"))


def load_train_series() -> pd.DataFrame:
    return _read_csv(data_path("train_series.csv"))


def load_test() -> pd.DataFrame:
    return _read_csv(data_path("test.csv"))


def load_test_series() -> pd.DataFrame:
    return _read_csv(data_path("test_series.csv"))


def load_sample_submission() -> pd.DataFrame:
    return _read_csv(data_path("sample_submission.csv"))

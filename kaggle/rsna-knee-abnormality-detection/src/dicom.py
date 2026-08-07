"""DICOM loading helpers for train_series/ and test_series/."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import pydicom
except ImportError as exc:
    pydicom = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from config import TRAIN_SERIES_DIR


def _require_pydicom() -> None:
    if pydicom is None:
        raise ImportError(
            "pydicom is required for DICOM loading. "
            "Install with: pip install pydicom"
        ) from _IMPORT_ERROR


def series_dir(
    study_uid: str,
    series_uid: str,
    root: Path | None = None,
) -> Path:
    root = root or TRAIN_SERIES_DIR
    return root / study_uid / series_uid


def list_dicom_paths(
    study_uid: str,
    series_uid: str,
    root: Path | None = None,
) -> list[Path]:
    folder = series_dir(study_uid, series_uid, root)
    if not folder.exists():
        return []
    return sorted(folder.glob("*.dcm"))


def load_dicom_slice(path: Path) -> np.ndarray:
    _require_pydicom()
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    return arr * slope + intercept


def load_series_volume(
    study_uid: str,
    series_uid: str,
    root: Path | None = None,
) -> np.ndarray:
    """Load all slices in a series as (slices, H, W)."""
    paths = list_dicom_paths(study_uid, series_uid, root)
    if not paths:
        raise FileNotFoundError(
            f"No DICOMs for study={study_uid} series={series_uid}"
        )
    slices = [load_dicom_slice(p) for p in paths]
    return np.stack(slices, axis=0)


def middle_slice(
    study_uid: str,
    series_uid: str,
    root: Path | None = None,
) -> np.ndarray:
    vol = load_series_volume(study_uid, series_uid, root)
    return vol[len(vol) // 2]

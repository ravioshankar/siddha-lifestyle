"""DICOM loading, preprocessing, and preferred-series selection."""

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
    paths = sorted(folder.glob("*.dcm"))
    if paths:
        return paths
    # Sample-data fallback: middle.npy
    return sorted(folder.glob("*.npy"))


def load_dicom_slice(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
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
    # Single middle.npy file → treat as 1-slice volume
    if len(paths) == 1 and paths[0].suffix == ".npy":
        arr = np.load(paths[0]).astype(np.float32)
        if arr.ndim == 2:
            return arr[None, ...]
        return arr
    slices = [load_dicom_slice(p) for p in paths]
    return np.stack(slices, axis=0)


def middle_slice(
    study_uid: str,
    series_uid: str,
    root: Path | None = None,
) -> np.ndarray:
    vol = load_series_volume(study_uid, series_uid, root)
    return vol[len(vol) // 2]


def normalize_slice(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Percentile clip + z-score → float32 in roughly [-3, 3]."""
    x = arr.astype(np.float32)
    lo, hi = np.percentile(x, (1, 99))
    x = np.clip(x, lo, hi)
    mu, sigma = float(x.mean()), float(x.std())
    return (x - mu) / (sigma + eps)


def resize_slice(arr: np.ndarray, size: int = 224) -> np.ndarray:
    """Nearest-neighbor resize without torchvision (keeps deps light)."""
    h, w = arr.shape[:2]
    ys = (np.linspace(0, h - 1, size)).astype(np.int32)
    xs = (np.linspace(0, w - 1, size)).astype(np.int32)
    return arr[ys][:, xs]


def preprocess_slice(arr: np.ndarray, size: int = 224) -> np.ndarray:
    return resize_slice(normalize_slice(arr), size=size)


def select_preferred_series(series_df) -> dict[str, str]:
    """Map StudyInstanceUID → SeriesInstanceUID preferring fluid-sensitive sagittal."""
    import pandas as pd

    if not isinstance(series_df, pd.DataFrame):
        raise TypeError("series_df must be a DataFrame")

    preferred: dict[str, str] = {}
    for study_uid, grp in series_df.groupby("StudyInstanceUID"):
        score = np.zeros(len(grp), dtype=float)
        if "Fluid_Sensitive" in grp.columns:
            score += grp["Fluid_Sensitive"].fillna(0).to_numpy(dtype=float) * 2.0
        if "Anatomical_Plane" in grp.columns:
            plane = grp["Anatomical_Plane"].fillna("").astype(str).str.lower()
            score += plane.str.contains("sagittal").to_numpy(dtype=float) * 1.5
            score += plane.str.contains("coronal").to_numpy(dtype=float) * 0.5
        if "Fat_Suppression" in grp.columns:
            score += grp["Fat_Suppression"].fillna(0).to_numpy(dtype=float) * 0.25
        idx = int(np.argmax(score))
        preferred[str(study_uid)] = str(grp.iloc[idx]["SeriesInstanceUID"])
    return preferred


def load_study_middle_preprocessed(
    study_uid: str,
    series_uid: str,
    root: Path | None = None,
    size: int = 224,
) -> np.ndarray:
    """Return (size, size) float32 middle slice ready for a 2D model."""
    return preprocess_slice(middle_slice(study_uid, series_uid, root=root), size=size)

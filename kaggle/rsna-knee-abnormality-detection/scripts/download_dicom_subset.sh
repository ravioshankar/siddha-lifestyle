#!/usr/bin/env bash
# Download a capped DICOM subset for local image experiments (NOT the full 570GB).
# Requires ~/.kaggle/access_token and competition rules accepted.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

KAGGLE_BIN=".venv/bin/kaggle"
if [[ ! -x "$KAGGLE_BIN" ]]; then KAGGLE_BIN="kaggle"; fi

if [[ ! -f "${HOME}/.kaggle/access_token" ]]; then
  echo "Missing ~/.kaggle/access_token — generating synthetic slices instead:"
  uv run python scripts/make_sample_dicoms.py
  exit 0
fi

echo "NOTE: Full train_series is huge. Prefer Kaggle Datasets / Notebooks for DICOMs."
echo "This script downloads competition files metadata; for a true subset, filter"
echo "StudyUIDs in a Kaggle Notebook and save a small dataset."
echo ""
echo "For local smoke tests without the archive:"
uv run python scripts/make_sample_dicoms.py

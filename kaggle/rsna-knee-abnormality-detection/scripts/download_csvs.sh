#!/usr/bin/env bash
# Download competition CSVs only (~few MB). Requires ~/.kaggle/kaggle.json
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

KAGGLE_BIN="${KAGGLE_BIN:-}"
if [[ -z "$KAGGLE_BIN" ]]; then
  if [[ -x .venv/bin/kaggle ]]; then
    KAGGLE_BIN=".venv/bin/kaggle"
  else
    KAGGLE_BIN="kaggle"
  fi
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "ERROR: Missing ${HOME}/.kaggle/kaggle.json"
  echo "1. Accept competition rules on Kaggle"
  echo "2. Account → Create New Token → place kaggle.json in ~/.kaggle/"
  echo "Meanwhile you can generate local sample CSVs:"
  echo "  .venv/bin/python scripts/make_sample_data.py"
  exit 1
fi

echo "Downloading CSV files only..."
for f in train.csv train_series.csv test.csv test_series.csv sample_submission.csv; do
  "$KAGGLE_BIN" competitions download -c rsna-knee-abnormality-detection -f "$f" -p data/raw
done

# Unzip any .zip sidecars Kaggle may produce
shopt -s nullglob
for z in data/raw/*.zip; do
  unzip -o "$z" -d data/raw
  rm -f "$z"
done

echo "Done. CSV files are in data/raw/"
.venv/bin/python scripts/verify_setup.py

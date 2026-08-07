@echo off
setlocal
cd /d "%~dp0.."
if not exist "data\raw" mkdir "data\raw"
echo Downloading CSV files only (fast, ~few MB)...
".venv\Scripts\kaggle.exe" competitions download -c rsna-knee-abnormality-detection -f train.csv -p data/raw
".venv\Scripts\kaggle.exe" competitions download -c rsna-knee-abnormality-detection -f train_series.csv -p data/raw
".venv\Scripts\kaggle.exe" competitions download -c rsna-knee-abnormality-detection -f test.csv -p data/raw
".venv\Scripts\kaggle.exe" competitions download -c rsna-knee-abnormality-detection -f test_series.csv -p data/raw
".venv\Scripts\kaggle.exe" competitions download -c rsna-knee-abnormality-detection -f sample_submission.csv -p data/raw
echo Done. CSV files are in data\raw\
endlocal

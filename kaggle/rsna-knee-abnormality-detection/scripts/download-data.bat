@echo off
setlocal
cd /d "%~dp0.."
if not exist "data\raw" mkdir "data\raw"
".venv\Scripts\kaggle.exe" competitions download -c rsna-knee-abnormality-detection -p data/raw
endlocal

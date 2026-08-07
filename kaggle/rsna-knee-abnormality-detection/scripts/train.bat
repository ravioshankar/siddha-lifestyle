@echo off
setlocal
cd /d "%~dp0.."
".venv\Scripts\python.exe" train_text_baseline.py %*
endlocal

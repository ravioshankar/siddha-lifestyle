@echo off
REM Open CMD with the project venv activated (no PowerShell execution policy needed)
cd /d "%~dp0.."
call .venv\Scripts\activate.bat
cmd /k

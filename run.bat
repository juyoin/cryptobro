@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating virtual environment...
  py -3 -m venv .venv
  if errorlevel 1 (
    echo [error] Failed to create virtual environment.
    exit /b 1
  )
)

echo [setup] Installing/updating dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [error] Failed to install dependencies.
  exit /b 1
)

echo [run] Starting Crypto Oracle GUI...
".venv\Scripts\python.exe" -m src.gui.main_app

endlocal

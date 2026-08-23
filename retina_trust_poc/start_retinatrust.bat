@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :venv_ready

where py >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_LAUNCHER=py -3"
) else (
  set "PYTHON_LAUNCHER=python"
)

echo Creating isolated RetinaTrust environment...
%PYTHON_LAUNCHER% -m venv ".venv"
if errorlevel 1 (
  echo Python 3.11 or newer is required.
  pause
  exit /b 1
)

:venv_ready
set "PYTHON_CMD=%CD%\.venv\Scripts\python.exe"
set "REQ_FILE=requirements.txt"
if exist "requirements.lock" set "REQ_FILE=requirements.lock"

"%PYTHON_CMD%" -c "import numpy,pandas,PIL,scipy,sklearn,joblib; assert sklearn.__version__ == '1.8.0'" >nul 2>nul
if errorlevel 1 (
  echo Installing verified Python packages...
  "%PYTHON_CMD%" -m pip install -r "%REQ_FILE%"
  if errorlevel 1 (
    echo Installation failed. Check Python and internet access.
    pause
    exit /b 1
  )
)

echo Starting RetinaTrust...
"%PYTHON_CMD%" app.py
pause

@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if exist "%cd%\VC_redist.x64.exe" (
    echo Installing Microsoft Visual C++ runtime if required...
    start /wait "" "%cd%\VC_redist.x64.exe" /install /passive /norestart
)

if not exist ".venv" (
    python -m venv .venv
)

call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip setuptools wheel

if exist "requirements.txt" (
    python -m pip install -r requirements.txt
) else (
    echo requirements.txt not found.
    pause
    exit /b 1
)

python -m streamlit run code/Home.py --server.port 8510

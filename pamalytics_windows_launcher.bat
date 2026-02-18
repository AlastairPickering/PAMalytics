@echo off
setlocal enabledelayedexpansion

rem Change to the directory of this script
cd /d "%~dp0"

rem Check if uv is installed
where uv >nul 2>nul
if %errorlevel% neq 0 (
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    rem Add uv to PATH for the current session
    if exist "%LOCALAPPDATA%\uv\uv.exe" (
        set "PATH=%LOCALAPPDATA%\uv;%PATH%"
    ) else if exist "%USERPROFILE%\.cargo\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    )
)

rem Install dependencies
uv sync

rem Run the app
uv run streamlit run src/pamalytics/app.py --server.port 8510

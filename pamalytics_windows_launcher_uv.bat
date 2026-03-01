@echo off
setlocal enabledelayedexpansion

rem Change to the directory of this script
cd /d "%~dp0"

rem Bootstrap uv locally
set "UV_DIR=%cd%\.uv-bin"
set "UV_EXE=%UV_DIR%\uv.exe"
set "UV_EXE_ALT=%UV_DIR%\bin\uv.exe"

if not exist "%UV_EXE%" (
  if not exist "%UV_EXE_ALT%" (
    if not exist "%UV_DIR%" mkdir "%UV_DIR%" >nul 2>&1

    powershell -NoProfile -ExecutionPolicy ByPass -Command ^
      "$ErrorActionPreference='Stop';" ^
      "$env:UV_INSTALL_DIR='%UV_DIR%';" ^
      "$env:UV_NO_MODIFY_PATH='1';" ^
      "irm https://astral.sh/uv/install.ps1 | iex"
  )
)

if exist "%UV_EXE_ALT%" set "UV_EXE=%UV_EXE_ALT%"

if not exist "%UV_EXE%" (
  echo "Error: uv did not install correctly (expected: %UV_EXE%)."
  exit /b 1
)

set "VENV_DIR=%cd%\.venv"
set "PY_EXE=%VENV_DIR%\Scripts\python.exe"

if exist "%VENV_DIR%" (
  if not exist "%PY_EXE%" (
    set "VENV_DIR=%cd%\.venv-win"
    set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
  )
)

rem If venv exists, ensure it's a supported Python (3.9-3.12); otherwise recreate
if exist "%PY_EXE%" (
  for /f "usebackq delims=" %%V in (`"%PY_EXE%" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul`) do set "PY_MINOR=%%V"
  set "SUPPORTED=0"
  if "%PY_MINOR%"=="3.9"  set "SUPPORTED=1"
  if "%PY_MINOR%"=="3.10" set "SUPPORTED=1"
  if "%PY_MINOR%"=="3.11" set "SUPPORTED=1"
  if "%PY_MINOR%"=="3.12" set "SUPPORTED=1"

  if not "!SUPPORTED!"=="1" (
    rmdir /s /q "%VENV_DIR%"
  )
)

rem Create venv if missing. Prefer existing system Python 3.12-3.9, else install 3.12
if not exist "%PY_EXE%" (
  set "SYS_PY="

  for %%V in (3.12 3.11 3.10 3.9) do (
    if not defined SYS_PY (
      for /f "usebackq delims=" %%P in (`"%UV_EXE%" python find --system --no-python-downloads %%V 2^>nul`) do set "SYS_PY=%%P"
    )
  )

  if defined SYS_PY (
    "%UV_EXE%" venv --python "%SYS_PY%" "%VENV_DIR%"
  ) else (
    "%UV_EXE%" venv --python 3.12 "%VENV_DIR%"
  )

  set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
)

rem Install dependencies into the venv
if exist "requirements.txt" (
  "%UV_EXE%" pip install --python "%PY_EXE%" -r requirements.txt
) else (
  "%UV_EXE%" pip install --python "%PY_EXE%" streamlit pydantic python-dateutil streamlit-extras
)

if exist "%cd%\VC_redist.x64.exe" (
  start /wait "" "%cd%\VC_redist.x64.exe" /install /passive /norestart
)

rem Launch Streamlit app
"%PY_EXE%" -m streamlit run code/Home.py --server.port 8510

endlocal
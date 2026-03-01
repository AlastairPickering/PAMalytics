@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

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
  echo uv not found: %UV_EXE%
  goto :err
)

set "VENV_DIR=%cd%\.venv"
set "PY_EXE=%VENV_DIR%\Scripts\python.exe"

if exist "%VENV_DIR%" (
  if not exist "%PY_EXE%" (
    set "VENV_DIR=%cd%\.venv-win"
    set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
  )
)

if not exist "%PY_EXE%" (
  set "SYS_PY="
  for %%V in (3.11 3.10 3.9) do (
    if not defined SYS_PY (
      for /f "usebackq delims=" %%P in (`"%UV_EXE%" python find --system --no-python-downloads %%V 2^>nul`) do set "SYS_PY=%%P"
    )
  )
  if defined SYS_PY (
    "%UV_EXE%" venv --python "%SYS_PY%" "%VENV_DIR%" || goto :err
  ) else (
    "%UV_EXE%" venv --python 3.11 "%VENV_DIR%" || goto :err
  )
  set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
)

if exist "%cd%\VC_redist.x64.exe" (
  start /wait "" "%cd%\VC_redist.x64.exe" /install /passive /norestart
)

"%PY_EXE%" -m pip --version >nul 2>&1
if errorlevel 1 (
  "%PY_EXE%" -m ensurepip --upgrade || goto :err
)

set "REQ_FILE=%cd%\requirements.txt"
if exist "%cd%\code\scripts\requirements.txt" set "REQ_FILE=%cd%\code\scripts\requirements.txt"

"%UV_EXE%" pip install --python "%PY_EXE%" -r "%REQ_FILE%" || goto :err

"%PY_EXE%" -c "import librosa" >nul 2>&1
if errorlevel 1 (
  for /f "usebackq delims=" %%V in (`"%PY_EXE%" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul`) do set "PY_MINOR=%%V"
  if "%PY_MINOR%"=="3.12" (
    set "VENV_DIR=%cd%\.venv-win311"
    set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
    if not exist "%PY_EXE%" (
      "%UV_EXE%" venv --python 3.11 "%VENV_DIR%" || goto :err
      set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
    )
    "%PY_EXE%" -m pip --version >nul 2>&1
    if errorlevel 1 (
      "%PY_EXE%" -m ensurepip --upgrade || goto :err
    )
    "%UV_EXE%" pip install --python "%PY_EXE%" -r "%REQ_FILE%" || goto :err
    "%PY_EXE%" -c "import librosa" >nul 2>&1 || goto :err
  ) else (
    goto :err
  )
)

"%PY_EXE%" -m streamlit run code/Home.py --server.port 8510
exit /b 0

:err
echo.
echo FAILED. The error is above.
pause
exit /b 1
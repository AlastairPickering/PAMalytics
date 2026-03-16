@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "APP_DIR=%cd%"
set "UV_DIR=%APP_DIR%\.uv-bin"
set "UV_EXE=%UV_DIR%\uv.exe"
set "UV_EXE_ALT=%UV_DIR%\bin\uv.exe"
set "UV_SOURCE="
set "BOOTSTRAP_ERR="

rem Resolve uv

if exist "%UV_EXE%" (
  set "UV_SOURCE=local"
  goto :uv_ready
)

if exist "%UV_EXE_ALT%" (
  set "UV_EXE=%UV_EXE_ALT%"
  set "UV_SOURCE=local"
  goto :uv_ready
)

for /f "usebackq delims=" %%P in (`where uv 2^>nul`) do (
  if not defined UV_SOURCE (
    set "UV_EXE=%%P"
    set "UV_SOURCE=path"
  )
)

if defined UV_SOURCE goto :uv_ready

where winget >nul 2>&1
if not errorlevel 1 (
  echo Installing uv via WinGet...
  winget install -e --id Astral-sh.uv --accept-source-agreements --accept-package-agreements >nul 2>&1

  rem First try PATH in the current session
  for /f "usebackq delims=" %%P in (`where uv 2^>nul`) do (
    if not defined UV_SOURCE (
      set "UV_EXE=%%P"
      set "UV_SOURCE=winget"
    )
  )

  rem Then try common install locations in case PATH has not refreshed yet
  if not defined UV_SOURCE if exist "%LocalAppData%\Microsoft\WinGet\Links\uv.exe" (
    set "UV_EXE=%LocalAppData%\Microsoft\WinGet\Links\uv.exe"
    set "UV_SOURCE=winget"
  )

  if not defined UV_SOURCE if exist "%UserProfile%\.local\bin\uv.exe" (
    set "UV_EXE=%UserProfile%\.local\bin\uv.exe"
    set "UV_SOURCE=winget"
  )
)

if defined UV_SOURCE goto :uv_ready

set "PS_MODE="
for /f "usebackq delims=" %%M in (`powershell -NoProfile -Command "$ExecutionContext.SessionState.LanguageMode" 2^>nul`) do (
  if not defined PS_MODE set "PS_MODE=%%M"
)

if /i "%PS_MODE%"=="ConstrainedLanguage" (
  set "BOOTSTRAP_ERR=PowerShell Constrained Language Mode blocks bootstrap"
  goto :uv_fail
)

if not exist "%UV_DIR%" mkdir "%UV_DIR%" >nul 2>&1

echo Installing uv via PowerShell bootstrap...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$env:UV_INSTALL_DIR='%UV_DIR%';" ^
  "$env:UV_NO_MODIFY_PATH='1';" ^
  "irm https://astral.sh/uv/install.ps1 | iex"

if exist "%UV_EXE_ALT%" set "UV_EXE=%UV_EXE_ALT%"
if exist "%UV_EXE%" (
  set "UV_SOURCE=bootstrap"
  goto :uv_ready
)

set "BOOTSTRAP_ERR=uv bootstrap did not produce uv.exe"
goto :uv_fail

:uv_ready
echo Using uv from !UV_SOURCE!: !UV_EXE!

rem Resolve venv / python

set "VENV_DIR=%APP_DIR%\.venv"
set "PY_EXE=%VENV_DIR%\Scripts\python.exe"

if exist "%VENV_DIR%" (
  if not exist "%PY_EXE%" (
    set "VENV_DIR=%APP_DIR%\.venv-win"
    set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
  )
)

if not exist "%PY_EXE%" (
  set "SYS_PY="
  for %%V in (3.11 3.10 3.9) do (
    if not defined SYS_PY (
      for /f "usebackq delims=" %%P in (`"%UV_EXE%" python find --system --no-python-downloads %%V 2^>nul`) do (
        if not defined SYS_PY set "SYS_PY=%%P"
      )
    )
  )

  if defined SYS_PY (
    echo Creating venv from system Python: !SYS_PY!
    "%UV_EXE%" venv --python "!SYS_PY!" "%VENV_DIR%" || goto :err
  ) else (
    echo Creating venv with Python 3.11...
    "%UV_EXE%" venv --python 3.11 "%VENV_DIR%" || goto :err
  )

  set "PY_EXE=%VENV_DIR%\Scripts\python.exe"
)

if not exist "!PY_EXE!" (
  echo Python executable not found after venv creation: !PY_EXE!
  goto :err
)

rem Optional VC runtime

if exist "%APP_DIR%\VC_redist.x64.exe" (
  echo Installing VC++ runtime...
  start /wait "" "%APP_DIR%\VC_redist.x64.exe" /install /passive /norestart
)

rem Ensure pip

"%PY_EXE%" -m pip --version >nul 2>&1
if errorlevel 1 (
  echo Bootstrapping pip...
  "%PY_EXE%" -m ensurepip --upgrade || goto :err
)

rem Install requirements

set "REQ_FILE=%APP_DIR%\requirements.txt"
if exist "%APP_DIR%\code\scripts\requirements.txt" set "REQ_FILE=%APP_DIR%\code\scripts\requirements.txt"

if not exist "%REQ_FILE%" (
  echo Requirements file not found: %REQ_FILE%
  goto :err
)

echo Installing dependencies from:
echo   %REQ_FILE%
"%UV_EXE%" pip install --python "%PY_EXE%" -r "%REQ_FILE%" || goto :err

rem librosa / Python 3.12 fallback

"%PY_EXE%" -c "import librosa" >nul 2>&1
if errorlevel 1 (
  set "PY_MINOR="
  for /f "usebackq delims=" %%V in (`"%PY_EXE%" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul`) do (
    set "PY_MINOR=%%V"
  )

  if "!PY_MINOR!"=="3.12" (
    echo librosa failed on Python 3.12, retrying with Python 3.11...
    set "VENV_DIR=%APP_DIR%\.venv-win311"
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
    echo librosa import failed on Python !PY_MINOR!.
    goto :err
  )
)

rem Launch app

if not exist "%APP_DIR%\code\Home.py" (
  echo Streamlit entry point not found: %APP_DIR%\code\Home.py
  goto :err
)

echo Launching PAMalytics...
"%PY_EXE%" -m streamlit run "%APP_DIR%\code\Home.py" --server.port 8510
exit /b 0

:uv_fail
echo.
echo FAILED to set up uv.
if defined BOOTSTRAP_ERR echo Reason: !BOOTSTRAP_ERR!
echo.
echo Recommended fixes:
echo   1. Bundle uv.exe with the app in .uv-bin
echo   2. Or install uv system-wide
echo   3. Or use a less restricted machine policy
goto :err

:err
echo.
echo FAILED. The error is above.
pause
exit /b 1
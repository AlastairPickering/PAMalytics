@echo off
setlocal

:: Move to root directory
pushd "%~dp0\.."

:: Start the launcher
start /b pamalytics_windows_launcher.bat > scripts\smoke_output.log 2>&1

echo Waiting for Streamlit to initialize...
set "SUCCESS=false"

for /L %%i in (1,1,60) do (
    findstr /C:"You can now view your Streamlit app in your browser." scripts\smoke_output.log >nul
    if %errorlevel%==0 (
        echo ✅ Streamlit started successfully!
        set "SUCCESS=true"
        goto :cleanup
    )
    timeout /t 1 >nul
)

:cleanup
:: Kill Python processes to stop the server
taskkill /F /IM python.exe /T >nul 2>&1
popd

if "%SUCCESS%"=="false" (
    echo ❌ ERROR: Timeout reached. Full logs below:
    type scripts\smoke_output.log
    exit /b 1
)

exit /b 0

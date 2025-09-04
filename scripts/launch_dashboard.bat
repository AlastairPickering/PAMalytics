@echo off
cd /d "%~dp0"
REM Run the Python bootstrapper (it creates venv, installs requirements, and launches Streamlit)
py -3 launch_dashboard.py || python launch_dashboard.py
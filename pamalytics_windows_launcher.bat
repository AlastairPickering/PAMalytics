@echo off
setlocal enabledelayedexpansion

rem Change to the directory of this script
cd /d "%~dp0"

rem Create virtual environment if it does not exist
if not exist ".venv" (
    python -m venv .venv
)

rem Activate virtual environment
call ".venv\Scripts\activate.bat"

rem Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install streamlit pydantic python-dateutil streamlit-extras

rem Launch Streamlit app
python -m streamlit run code/Home.py --server.port 8510

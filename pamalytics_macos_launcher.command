#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install streamlit pydantic python-dateutil streamlit-extras

python -m streamlit run code/Home.py --server.port 8510
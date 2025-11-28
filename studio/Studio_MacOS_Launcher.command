#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install streamlit pydantic python-dateutil streamlit-extras

# Start via app root so Streamlit sees ./pages/*
python -m streamlit run studio/Home.py --server.port 8510

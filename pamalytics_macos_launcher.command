#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3.12 -m venv .venv || python3 -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

REQ_FILE=""
if [ -f "requirements.txt" ]; then
  REQ_FILE="requirements.txt"
elif [ -f "code/scripts/requirements.txt" ]; then
  REQ_FILE="code/scripts/requirements.txt"
fi

if [ -n "$REQ_FILE" ]; then
  python -m pip install -r "$REQ_FILE"
else
  python -m pip install streamlit librosa soundfile
fi

python -m streamlit run code/Home.py --server.port 8510

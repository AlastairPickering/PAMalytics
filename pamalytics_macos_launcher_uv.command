#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Bootstrap uv locally
UV_DIR="${PWD}/.uv-bin"
UV_BIN="${UV_DIR}/uv"

if [ ! -x "$UV_BIN" ]; then
  mkdir -p "$UV_DIR"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_DIR" sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$UV_DIR" sh
  else
    echo "Error: Need either 'curl' or 'wget' to download uv."
    exit 1
  fi
fi

if [ ! -x "$UV_BIN" ]; then
  echo "Error: uv did not install correctly (expected at: $UV_BIN)."
  exit 1
fi

VENV_DIR="${PWD}/.venv"
PY_BIN="${VENV_DIR}/bin/python"

# If venv exists, ensure it's supported (3.9-3.12); otherwise recreate
if [ -x "$PY_BIN" ]; then
  PY_MINOR="$("$PY_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  case "$PY_MINOR" in
    3.9|3.10|3.11|3.12) : ;;
    *) rm -rf "$VENV_DIR" ;;
  esac
fi

# Create venv if missing. Prefer existing system Python 3.12->3.9, else install 3.12
if [ ! -x "$PY_BIN" ]; then
  SYS_PY=""

  for v in 3.12 3.11 3.10 3.9; do
    SYS_PY="$("$UV_BIN" python find --system --no-python-downloads "$v" 2>/dev/null || true)"
    if [ -n "$SYS_PY" ]; then
      break
    fi
  done

  if [ -n "$SYS_PY" ]; then
    "$UV_BIN" venv --python "$SYS_PY" "$VENV_DIR"
  else
    "$UV_BIN" venv --python 3.12 "$VENV_DIR"
  fi

  PY_BIN="${VENV_DIR}/bin/python"
fi

# Install dependencies into the venv
if [ -f "requirements.txt" ]; then
  "$UV_BIN" pip install --python "$PY_BIN" -r requirements.txt
else
  echo "Error: requirements.txt not found."
  exit 1
fi

# Launch PAMalytics (Streamlit)
exec "$PY_BIN" -m streamlit run code/Home.py --server.port 8510
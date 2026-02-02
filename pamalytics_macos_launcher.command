#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Install uv if not installed
if ! command -v uv &>/dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for the current session
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    elif [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

# Install dependencies
uv sync

# Run the app
uv run streamlit run src/pamalytics/app.py --server.port 8510

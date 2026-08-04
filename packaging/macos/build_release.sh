#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERSION="${1:-}"
PYTHON_BIN="${PAMALYTICS_PYTHON:-python3.12}"

if [[ -z "$VERSION" ]]; then
  echo "Usage: bash packaging/macos/build_release.sh <version>" >&2
  echo "Example: bash packaging/macos/build_release.sh v1.0.0" >&2
  exit 1
fi

cd "$ROOT"

"$PYTHON_BIN" --version
"$PYTHON_BIN" packaging/macos/build_app.py
bash packaging/macos/build_dmg.sh "$VERSION"

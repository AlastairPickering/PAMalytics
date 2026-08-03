from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
BUILD = ROOT / "build"


def main() -> int:
    for path in (DIST, BUILD):
        if path.exists():
            shutil.rmtree(path)
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        "PAMalytics",
        "--windowed",
        "--clean",
        "--noconfirm",
        "--icon",
        str(ROOT / "packaging" / "macos" / "PAMalytics.icns"),
        "--add-data",
        f"{ROOT / 'code'}:code",
        "--collect-all",
        "streamlit",
        "--collect-all",
        "altair",
        "--collect-all",
        "plotly",
        "--collect-all",
        "pydeck",
        "--collect-all",
        "matplotlib",
        "--collect-all",
        "librosa",
        "--collect-all",
        "soundfile",
        "--collect-all",
        "audioread",
        "--collect-all",
        "soxr",
        "--collect-all",
        "numba",
        "--collect-all",
        "llvmlite",
        "--collect-all",
        "scipy",
        "--collect-all",
        "sklearn",
        "--collect-all",
        "pyproj",
        "--collect-all",
        "pandas",
        "--collect-all",
        "numpy",
        "--collect-all",
        "openpyxl",
        "--collect-all",
        "tqdm",
        "--copy-metadata",
        "streamlit",
        "--copy-metadata",
        "altair",
        "--copy-metadata",
        "plotly",
        "--copy-metadata",
        "pydeck",
        "--copy-metadata",
        "pandas",
        "--copy-metadata",
        "numpy",
        "--copy-metadata",
        "librosa",
        "--copy-metadata",
        "soundfile",
        "--copy-metadata",
        "audioread",
        "--copy-metadata",
        "soxr",
        "--copy-metadata",
        "numba",
        "--copy-metadata",
        "llvmlite",
        "--copy-metadata",
        "scipy",
        "--copy-metadata",
        "scikit-learn",
        "--copy-metadata",
        "matplotlib",
        "--copy-metadata",
        "pyproj",
        "--copy-metadata",
        "openpyxl",
        "--copy-metadata",
        "tqdm",
        "--hidden-import",
        "streamlit.web.cli",
        "--hidden-import",
        "streamlit.runtime.scriptrunner",
        str(ROOT / "packaging" / "macos" / "launcher.py"),
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())

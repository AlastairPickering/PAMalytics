from __future__ import annotations

import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
BUILD = ROOT / "build"
APP = DIST / "PAMalytics.app"
INFO_PLIST = APP / "Contents" / "Info.plist"
RESOURCES = APP / "Contents" / "Resources"


def _require_python_312() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "The macOS app must be built with Python 3.12. "
            f"Current interpreter: {sys.version.split()[0]} at {sys.executable}"
        )


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=str(ROOT), check=True)


def _validate_app_bundle() -> None:
    if not APP.is_dir():
        raise RuntimeError(f"PyInstaller did not create the app bundle: {APP}")
    if not INFO_PLIST.is_file():
        raise RuntimeError(f"The app bundle has no Info.plist: {INFO_PLIST}")

    with INFO_PLIST.open("rb") as handle:
        info = plistlib.load(handle)

    icon_name = str(info.get("CFBundleIconFile", "")).strip()
    if not icon_name:
        raise RuntimeError("The built app has no CFBundleIconFile entry.")

    icon_path = RESOURCES / icon_name
    if not icon_path.suffix:
        icon_path = icon_path.with_suffix(".icns")
    if not icon_path.is_file():
        raise RuntimeError(
            f"Info.plist refers to {icon_name!r}, but the icon is missing from "
            f"{RESOURCES}."
        )

    plutil = shutil.which("plutil")
    if plutil:
        _run([plutil, "-lint", str(INFO_PLIST)])

    codesign = shutil.which("codesign")
    if not codesign:
        raise RuntimeError("codesign was not found; this build must run on macOS.")
    _run([codesign, "--verify", "--deep", "--strict", "--verbose=2", str(APP)])

    print(f"Validated app bundle: {APP}")
    print(f"Validated app icon: {icon_path}")
    print("Validated macOS code signature.")



def _validate_source_imports() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "from code.scripts.dashboard import render_dashboard; "
            "assert callable(render_dashboard); "
            "print('Validated dashboard renderer import.')"
        ),
    ]
    _run(command)


def _validate_arrow_allocator() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import os; "
            "os.environ['ARROW_DEFAULT_MEMORY_POOL'] = 'system'; "
            "import pyarrow as pa; "
            "backend = pa.default_memory_pool().backend_name; "
            "assert backend == 'system', "
            "f'Expected Arrow system allocator, got {backend!r}'; "
            "print('Validated Arrow system memory allocator.')"
        ),
    ]
    _run(command)

def main() -> int:
    _require_python_312()
    _validate_source_imports()
    _validate_arrow_allocator()

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
        "--hidden-import",
        "AppKit",
        "--hidden-import",
        "Foundation",
        "--hidden-import",
        "PyObjCTools",
        str(ROOT / "packaging" / "macos" / "launcher.py"),
    ]

    _run(cmd)
    _validate_app_bundle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

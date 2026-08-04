from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "dist"
BUILD = ROOT / "build"
APP_DIR = DIST / "PAMalytics"
EXE = APP_DIR / "PAMalytics.exe"
ICON_PNG = ROOT / "packaging" / "macos" / "PAMalytics-icon.png"
ICON_ICO = ROOT / "packaging" / "windows" / "PAMalytics.ico"


def _require_windows_python_312() -> None:
    if os.name != "nt":
        raise RuntimeError("The Windows application must be built on Windows.")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "The Windows application must be built with Python 3.12. "
            f"Current interpreter: {sys.version.split()[0]} at {sys.executable}"
        )


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=str(ROOT), check=True)


def _validate_source_imports() -> None:
    _run(
        [
            sys.executable,
            "-c",
            (
                "from code.scripts.dashboard import render_dashboard; "
                "assert callable(render_dashboard); "
                "print('Validated dashboard renderer import.')"
            ),
        ]
    )


def _create_windows_icon() -> None:
    if ICON_ICO.is_file():
        return
    if not ICON_PNG.is_file():
        raise RuntimeError(f"Missing source icon: {ICON_PNG}")

    from PIL import Image

    image = Image.open(ICON_PNG).convert("RGBA")
    image.save(
        ICON_ICO,
        format="ICO",
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


def _installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _append_collect_all(command: list[str], module_name: str) -> None:
    if _installed(module_name):
        command.extend(["--collect-all", module_name])


def _append_metadata(command: list[str], distribution_name: str, module_name: str | None = None) -> None:
    if _installed(module_name or distribution_name.replace("-", "_")):
        command.extend(["--copy-metadata", distribution_name])


def _validate_build() -> None:
    if not EXE.is_file():
        raise RuntimeError(f"PyInstaller did not create the executable: {EXE}")

    required = [
        APP_DIR / "_internal" / "code" / "Home.py",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "The packaged application is missing required files: "
            + ", ".join(str(path) for path in missing)
        )

    print(f"Validated Windows application: {EXE}")


def main() -> int:
    _require_windows_python_312()
    _validate_source_imports()
    _create_windows_icon()

    for path in (DIST, BUILD):
        if path.exists():
            shutil.rmtree(path)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name",
        "PAMalytics",
        "--windowed",
        "--onedir",
        "--clean",
        "--noconfirm",
        "--icon",
        str(ICON_ICO),
        "--add-data",
        f"{ROOT / 'code'}{os.pathsep}code",
    ]

    for module_name in [
        "streamlit",
        "altair",
        "plotly",
        "pydeck",
        "matplotlib",
        "librosa",
        "soundfile",
        "audioread",
        "soxr",
        "numba",
        "llvmlite",
        "scipy",
        "sklearn",
        "pyproj",
        "pandas",
        "numpy",
        "openpyxl",
        "tqdm",
        "pyarrow",
    ]:
        _append_collect_all(command, module_name)

    for distribution_name, module_name in [
        ("streamlit", "streamlit"),
        ("altair", "altair"),
        ("plotly", "plotly"),
        ("pydeck", "pydeck"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("audioread", "audioread"),
        ("soxr", "soxr"),
        ("numba", "numba"),
        ("llvmlite", "llvmlite"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
        ("pyproj", "pyproj"),
        ("openpyxl", "openpyxl"),
        ("tqdm", "tqdm"),
        ("pyarrow", "pyarrow"),
    ]:
        _append_metadata(command, distribution_name, module_name)

    command.extend(
        [
            "--hidden-import",
            "streamlit.web.cli",
            "--hidden-import",
            "streamlit.runtime.scriptrunner",
            "--hidden-import",
            "tkinter",
            str(ROOT / "packaging" / "windows" / "launcher.py"),
        ]
    )

    _run(command)
    _validate_build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

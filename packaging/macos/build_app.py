from __future__ import annotations

import plistlib
import os
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
ENTITLEMENTS = ROOT / "packaging" / "macos" / "PAMalytics.entitlements"
DEFAULT_SIGNING_IDENTITY = "Developer ID Application: Alastair Pickering (FKXA4C7S9Z)"


def _require_python_312() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "The macOS app must be built with Python 3.12. "
            f"Current interpreter: {sys.version.split()[0]} at {sys.executable}"
        )


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=str(ROOT), check=True)


def _signing_identity() -> str:
    identity = os.environ.get("PAMALYTICS_CODESIGN_IDENTITY", DEFAULT_SIGNING_IDENTITY).strip()
    if not identity:
        raise RuntimeError("PAMALYTICS_CODESIGN_IDENTITY is empty.")

    result = subprocess.run(
        ["security", "find-identity", "-v", "-p", "codesigning"],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    if identity not in result.stdout:
        raise RuntimeError(
            "Developer ID signing identity was not found in the Keychain: "
            f"{identity}"
        )
    return identity


def _validate_app_bundle(signing_identity: str) -> None:
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

    details = subprocess.run(
        [codesign, "--display", "--verbose=4", str(APP)],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    signature_output = details.stdout + details.stderr
    if signing_identity.split(" (")[0].replace("Developer ID Application: ", "") not in signature_output:
        raise RuntimeError("The app bundle was not signed with the expected Developer ID identity.")
    if "runtime" not in signature_output:
        raise RuntimeError("The app signature does not have hardened runtime enabled.")

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
    signing_identity = _signing_identity()

    if not ENTITLEMENTS.is_file():
        raise RuntimeError(f"Missing macOS entitlements file: {ENTITLEMENTS}")

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
        "--codesign-identity",
        signing_identity,
        "--osx-entitlements-file",
        str(ENTITLEMENTS),
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
    _validate_app_bundle(signing_identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

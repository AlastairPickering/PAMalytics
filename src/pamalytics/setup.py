import os as _os
import platform as _platform
import subprocess as _sp
import sys as _sys
import venv as _venv
from pathlib import Path
from pathlib import Path as _Path
import importlib.util as _iu

# Paths
STUDIO_ROOT = Path(__file__).resolve().parent  # code/
REPO_ROOT = STUDIO_ROOT.parent  # repo root
SCRIPTS_DIR = STUDIO_ROOT / "scripts"  # code/scripts


_STUDIO_FILE = _Path(__file__).resolve()  # code/Home.py
_STUDIO_ROOT = _STUDIO_FILE.parent  # code/
_REPO_ROOT = _STUDIO_ROOT.parent  # repo root
_SCRIPTS_DIR = _STUDIO_ROOT / "scripts"  # code/scripts

_REQS_FILE = _SCRIPTS_DIR / "requirements.txt"
if not _REQS_FILE.exists() and (_REPO_ROOT / "requirements.txt").exists():
    _REQS_FILE = _REPO_ROOT / "requirements.txt"


def _default_venv_dir() -> _Path:
    if _os.name == "nt":
        base = _Path(
            _os.environ.get("LOCALAPPDATA", _Path.home() / "AppData" / "Local")
        )
        return base / "PileatedGibbonDashboard" / ".venv"
    return _Path.home() / ".pileated_gibbon_dashboard" / ".venv"


_VENV_DIR = _Path(_os.environ.get("PG_VENV_DIR", _default_venv_dir()))
_PY_EXE = _VENV_DIR / ("Scripts/python.exe" if _os.name == "nt" else "bin/python")


def _run(cmd, **kw):
    print(">", " ".join(map(str, cmd)))
    _sp.check_call(cmd, **kw)


def _ensure_venv():
    if _VENV_DIR.exists():
        return
    print(f"[setup] Creating virtual environment at: {_VENV_DIR}")
    _VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
    _venv.EnvBuilder(with_pip=True).create(str(_VENV_DIR))


def _torch_import_ok(py_exe: str) -> bool:
    try:
        _run([py_exe, "-c", "import torch; print(torch.__version__)"])
        return True
    except _sp.CalledProcessError:
        return False


def _install_torch(py_exe: str):
    print("[setup] Installing PyTorch…")
    if _platform.system() == "Windows":
        _run(
            [
                py_exe,
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                "torch",
                "torchvision",
                "torchaudio",
            ]
        )
    else:
        _run([py_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
    _run(
        [
            py_exe,
            "-c",
            "import torch, torchaudio; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__)",
        ]
    )


def _needs_bootstrap() -> bool:
    in_managed = _sys.executable and str(_sys.executable).startswith(str(_VENV_DIR))
    try:

        have_librosa = _iu.find_spec("librosa") is not None
        have_streamlit = _iu.find_spec("streamlit") is not None
        have_sf = _iu.find_spec("soundfile") is not None
    except Exception:
        have_librosa = False
        have_streamlit = False
        have_sf = False
    return (
        (not in_managed) or (not have_streamlit) or (not have_librosa) or (not have_sf)
    )


def _update_pip():
    _run(
        [
            str(_PY_EXE),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
        ]
    )


def _install_packages():
    if not _REQS_FILE.exists():
        raise FileNotFoundError(f"requirements.txt not found at {_REQS_FILE}")
    print("[setup] Installing requirements.txt …")
    _run([str(_PY_EXE), "-m", "pip", "install", "-r", str(_REQS_FILE)])


def _check_torch():
    try:
        if not _torch_import_ok(str(_PY_EXE)):
            _install_torch(str(_PY_EXE))
    except Exception as _e:
        print(f"[setup] PyTorch check/install skipped or failed: {_e}")


def check_environment():
    _ensure_venv()
    _update_pip()
    _install_packages()
    _check_torch()

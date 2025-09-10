# create venv + install requirements + install/verify PyTorch
import os
import sys
import subprocess
import venv
from pathlib import Path
import platform
import webbrowser

HERE = Path(__file__).resolve().parent           # repo folder (contains Dashboard.py)
REQS = HERE / "requirements.txt"
DASH = HERE / "Dashboard.py"

def default_venv_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "PileatedGibbonDashboard" / ".venv"
    else:
        return Path.home() / ".pileated_gibbon_dashboard" / ".venv"

VENV = Path(os.environ.get("PG_VENV_DIR", default_venv_dir()))

def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

def run(cmd, **kw):
    print(">", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, **kw)

def ensure_venv():
    if VENV.exists():
        return
    print(f"[setup] Creating virtual environment at: {VENV}")
    VENV.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True).create(str(VENV))

def install_or_verify_torch(py_exe: str):
    """
    Ensure torch/torchvision/torchaudio are importable in this venv.
    On Windows, force CPU wheels from the official PyTorch index to avoid DLL issues.
    If WinError 126 occurs, prompt for the VC++ runtime.
    """
    try:
        import importlib
        importlib.import_module("torch")
        print("[setup] PyTorch already present.")
        return
    except Exception:
        pass

    print("[setup] Installing PyTorch…")
    is_windows = platform.system() == "Windows"
    if is_windows:
        # Force CPU wheels; override any prior build
        cmd = [
            py_exe, "-m", "pip", "install",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "--upgrade", "--force-reinstall",
            "torch", "torchvision", "torchaudio",
        ]
    else:
        cmd = [py_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]

    try:
        run(cmd)
    except subprocess.CalledProcessError as e:
        print("[setup][ERROR] PyTorch install command failed.")
        raise

    # Verify import and catch common Windows DLL error
    try:
        import torch  # noqa
        print("[setup] PyTorch OK.")
    except OSError as e:
        # Classic missing MSVC runtime on Windows
        winerr = getattr(e, "winerror", None)
        if is_windows and winerr == 126:
            msg = (
                "[setup][ERROR] PyTorch failed to load (WinError 126). "
                "Please install the Microsoft Visual C++ 2015–2022 Redistributable (x64) "
                "and relaunch:\n  https://aka.ms/vs/17/release/vc_redist.x64.exe"
            )
            print(msg)
            try:
                webbrowser.open("https://aka.ms/vs/17/release/vc_redist.x64.exe")
            except Exception:
                pass
        raise

def main():
    os.chdir(str(HERE))
    ensure_venv()
    py = str(venv_python())

    print(f"✅ Using venv: {VENV}")
    run([py, "-V"])

    # Upgrade pip tooling
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Install from requirements.txt
    if not REQS.exists():
        raise FileNotFoundError(f"requirements.txt not found at {REQS}")
    print("[setup] Installing requirements.txt …")
    run([py, "-m", "pip", "install", "-r", str(REQS)])

    # Ensure PyTorch is correct for this platform 
    install_or_verify_torch(py)

    # Sanity print: where Streamlit is installed
    run([py, "-c", "import streamlit; print('✅ Streamlit', streamlit.__version__, 'in', streamlit.__file__)"])

    # Launch Streamlit with watcher disabled
    env = os.environ.copy()
    env["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
    env["STREAMLIT_LOG_LEVEL"] = "error"
    print("[run] Launching Streamlit…")
    run([
        py, "-m", "streamlit", "run", str(DASH),
        "--server.fileWatcherType", "none",
        "--logger.level", "error"
    ], env=env)

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"\nLaunch failed (exit {e.returncode}).")
        input("Press Enter to exit…")
    except Exception as e:
        print(f"\nError: {e}")
        input("Press Enter to exit…")

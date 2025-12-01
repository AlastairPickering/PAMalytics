# launch_dashboard.py — venv + deps + minimal PyTorch check
import os
import subprocess
import venv
from pathlib import Path
import platform
import webbrowser

# ---- Optional Studio project handoff (backwards compatible) ----
import json

try:
    params = st.experimental_get_query_params()
    proj_param = params.get("project", [None])[0]
except Exception:
    proj_param = None

if proj_param:
    proj = Path(proj_param)
    bridge = proj / "exports" / "pa_bridge" / "pa_bridge.json"
    if bridge.exists():
        cfg = json.loads(bridge.read_text(encoding="utf-8"))
        DETECTIONS_PATH = cfg.get("detections_path")
        AUDIO_MAP_PATH  = cfg.get("audio_map_path")
        # If your existing pages use other variables/paths, set them here too.
        # Everything else continues to work as before.
    else:
        st.warning(f"No PA bridge found at: {bridge}. Falling back to default data.")


HERE = Path(__file__).resolve().parent      # repo/scripts/
REQS = HERE / "requirements.txt"
DASH = HERE / "Dashboard.py"

def default_venv_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "PileatedGibbonDashboard" / ".venv"
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

def torch_import_ok(py_exe: str) -> bool:
    """Return True if 'import torch' succeeds in the venv."""
    try:
        run([py_exe, "-c", "import torch; print(torch.__version__)"])
        return True
    except subprocess.CalledProcessError:
        return False

def install_torch(py_exe: str):
    """Install PyTorch once (CPU wheels on Windows), then verify import."""
    print("[setup] Installing PyTorch…")
    if platform.system() == "Windows":
        # Use CPU wheels to avoid GPU/DLL issues
        run([py_exe, "-m", "pip", "install",
             "--index-url", "https://download.pytorch.org/whl/cpu",
             "torch", "torchvision", "torchaudio"])
    else:
        run([py_exe, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])

    # Verify once
    try:
        run([py_exe, "-c", "import torch, torchaudio; print('torch', torch.__version__, 'torchaudio', torchaudio.__version__)"])
        print("[setup] PyTorch OK.")
    except subprocess.CalledProcessError as e:
        if platform.system() == "Windows":
            # Common case: missing MSVC runtime: WinError 126
            print("[setup][ERROR] PyTorch failed to import. On Windows, install the Microsoft Visual C++ 2015–2022 Redistributable (x64).")
            print("               Opening the download page…")
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

    # Pip tooling
    run([py, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # requirements.txt (should NOT contain torch/torchaudio/torchvision)
    if not REQS.exists():
        raise FileNotFoundError(f"requirements.txt not found at {REQS}")
    print("[setup] Installing requirements.txt …")
    run([py, "-m", "pip", "install", "-r", str(REQS)])

    # Minimal torch check + install if needed
    if not torch_import_ok(py):
        install_torch(py)

    # Locate and confirm Streamlit
    run([py, "-c", "import streamlit; print('✅ Streamlit', streamlit.__version__, 'in', streamlit.__file__)"])

    # Launch Streamlit
    env = os.environ.copy()
    env["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
    env["STREAMLIT_LOG_LEVEL"] = "error"
    print("[run] Launching Streamlit…")
    run([
        py, "-m", "streamlit", "run", str(DASH),
        "--server.fileWatcherType", "none",
        "--logger.level", "error",
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

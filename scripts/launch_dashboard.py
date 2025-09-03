# launch_dashboard.py 
# create venv + installs requirements 
import os
import sys
import subprocess
import venv
from pathlib import Path
import platform

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
    print(f"Creating virtual environment at: {VENV}")
    VENV.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True).create(str(VENV))

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
    run([py, "-m", "pip", "install", "-r", str(REQS)])

    # Sanity print: where Streamlit is installed
    run([py, "-c", "import streamlit; print('✅ Streamlit', streamlit.__version__, 'in', streamlit.__file__)"])

    # Launch Streamlit with watcher disabled
    env = os.environ.copy()
    env["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
    env["STREAMLIT_LOG_LEVEL"] = "error"
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

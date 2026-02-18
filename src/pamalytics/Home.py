# code/Home.py

from pathlib import Path
import sys

# environment bootstrap
import os as _os
from pamalytics.setup import check_environment, _needs_bootstrap, _PY_EXE, _STUDIO_FILE
from pamalytics.app import *

# Paths
STUDIO_ROOT = Path(__file__).resolve().parent      # code/
REPO_ROOT   = STUDIO_ROOT.parent                   # repo root
SCRIPTS_DIR = STUDIO_ROOT / "scripts"              # code/scripts

# Ensure scripts/ is importable
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


if _os.environ.get("PA_STUDIO_BOOTSTRAPPED", "0") != "1" and _needs_bootstrap():
    check_environment()

    env = _os.environ.copy()
    env["PA_STUDIO_BOOTSTRAPPED"] = "1"
    env["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"
    env["STREAMLIT_LOG_LEVEL"] = "error"
    PORT = str(_os.environ.get("PA_STUDIO_PORT", "8510"))
    args = [
        str(_PY_EXE), "-m", "streamlit", "run", str(_STUDIO_FILE),
        "--server.headless", "true",
        "--server.port", PORT,
        "--server.fileWatcherType", "none",
        "--logger.level", "error",
    ]
    _os.execvpe(str(_PY_EXE), args, env)

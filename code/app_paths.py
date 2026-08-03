from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Dict

APP_NAME = "PAMalytics"
CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent


def _default_user_root() -> Path:
    override = os.environ.get("PAMALYTICS_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / "Documents" / APP_NAME


def user_root() -> Path:
    path = _default_user_root()
    path.mkdir(parents=True, exist_ok=True)
    return path


USER_ROOT = user_root()
PROJECTS_ROOT = USER_ROOT / "projects"
AUTH_FILE = USER_ROOT / ".auth.json"
LOGS_ROOT = USER_ROOT / "logs"
CACHE_ROOT = USER_ROOT / "cache"

for _path in (PROJECTS_ROOT, LOGS_ROOT, CACHE_ROOT):
    _path.mkdir(parents=True, exist_ok=True)


def runtime_summary() -> Dict[str, str]:
    return {
        "app_name": APP_NAME,
        "user_root": str(USER_ROOT),
        "projects_root": str(PROJECTS_ROOT),
        "auth_file": str(AUTH_FILE),
        "logs_root": str(LOGS_ROOT),
        "cache_root": str(CACHE_ROOT),
        "code_root": str(CODE_ROOT),
        "repo_root": str(REPO_ROOT),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pamalytics_home_env": os.environ.get("PAMALYTICS_HOME", ""),
    }

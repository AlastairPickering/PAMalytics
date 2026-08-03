from __future__ import annotations

from pathlib import Path
import importlib
import os
import sqlite3
import sys
import tempfile

assert (3, 12) <= sys.version_info < (3, 13), "Python 3.12 required"

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"

if not CODE_ROOT.exists():
    raise SystemExit(f"code/ folder not found at {CODE_ROOT}")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

os.environ.setdefault("PAMALYTICS_HOME", tempfile.mkdtemp(prefix="pamalytics_smoke_"))

MODULES = [
    "code.app_paths",
    "code.Home",
    "code.core.pa_data",
    "code.core.project",
    "code.core.ui",
    "code.pages.40_Dashboard",
    "code.pages.41_Validation",
    "code.pages.43_Recalculate",
    "code.scripts.adapters.batdetect2",
    "code.scripts.adapters.birdnet",
    "code.scripts.dashboard",
    "code.scripts.schema",
]

failures = []
for modname in MODULES:
    try:
        importlib.import_module(modname)
    except Exception as e:
        failures.append((modname, repr(e)))

if failures:
    detail = "\n".join(f"- {m}: {err}" for m, err in failures)
    raise SystemExit(f"Smoke test import failures:\n{detail}")

from code.app_paths import AUTH_FILE, PROJECTS_ROOT, USER_ROOT

if CODE_ROOT in USER_ROOT.parents or USER_ROOT == CODE_ROOT:
    raise SystemExit(f"User data root must not be inside code/: {USER_ROOT}")

PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
AUTH_FILE.write_text('{"remember": false, "user": ""}', encoding="utf-8")

with sqlite3.connect(":memory:") as conn:
    conn.execute("CREATE TABLE audio_files (filename_lc TEXT, stem_lc TEXT, path TEXT, path_lc TEXT, rel_lc TEXT)")
    conn.execute("CREATE TABLE audio_index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

print("PAMalytics smoke test passed")
print(f"Python: {sys.version.split()[0]}")
print(f"User root: {USER_ROOT}")

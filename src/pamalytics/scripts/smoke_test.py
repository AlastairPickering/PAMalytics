# code/scripts/smoke_test.py

from __future__ import annotations
from pathlib import Path
import importlib
import sys

# Sanity-check Python version
assert (3, 9) <= sys.version_info < (3, 13), "Python 3.9–3.12 required"

# run this from repo root: `python -m code.scripts.smoke_test`
REPO_ROOT = Path(__file__).resolve().parents[2]   # .../<repo>
CODE_ROOT = REPO_ROOT / "code"

if not CODE_ROOT.exists():
    raise SystemExit(f"code/ folder not found at {CODE_ROOT}")

MODULES = [
    "code.Home",
    "code.core.pa_data",
    "code.core.project",
    "code.core.ui",
    "code.pages.40_Dashboard",
    "code.pages.41_Validation",
    "code.pages.43_Recalculate",
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

# light-touch structural checks
projects_dir = CODE_ROOT / "projects"
if not projects_dir.exists():
    print(f"Note: projects directory missing at {projects_dir} (may be fine in CI)")

print("PAMalytics smoke test passed on", sys.version)

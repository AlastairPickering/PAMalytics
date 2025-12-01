from pathlib import Path
import importlib.util
import sys

assert (3, 9) <= sys.version_info < (3, 13), "Python 3.9-3.12 required"

# repo root
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

FILES = [
    "preprocessing.py",
    "prepare_occupancy.py",
    "pipeline.py",
]

def load_module_from_path(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module) 
    return module

failures = []
for fname in FILES:
    path = SCRIPTS_DIR / fname
    if not path.exists():
        failures.append((fname, f"File not found: {path}"))
        continue
    try:
        # give each temp module a unique, throwaway name
        load_module_from_path(f"pamalytics_{fname[:-3]}", path)
    except Exception as e:
        failures.append((fname, repr(e)))

if failures:
    detail = "\n".join(f"- {f}: {err}" for f, err in failures)
    raise SystemExit(f"Smoke test import failures:\n{detail}")

# Non-fatal sanity signal if these folders are absent in CI workspaces
expected = ["audio", "results", "models"]
missing = [d for d in expected if not (ROOT / d).exists()]
if missing:
    print(f"Note: missing expected folders (may be fine in CI): {missing}")

print("Smoke test passed on", sys.version)

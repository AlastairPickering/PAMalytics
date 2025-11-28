# studio/pages/43_PAM_Occupancy.py
from pathlib import Path
import sys
import importlib.util
import streamlit as st
import pandas as pd

# ---------- repo paths ----------
STUDIO_DIR  = Path(__file__).resolve().parents[1]
REPO_ROOT   = STUDIO_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
PAGES_DIR   = SCRIPTS_DIR / "pages"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ---------- get data helpers from studio/utils ----------
from studio.utils import (  # type: ignore
    build_analysis_dataset,
    project_path,
)

def _load_callable(preferred_file: str, fallbacks_glob: str, func_name: str):
    # Try exact file first (since '4_Occupancy.py' isn't a valid module name for importlib.import_module)
    exact = PAGES_DIR / preferred_file
    if exact.exists():
        spec = importlib.util.spec_from_file_location(exact.stem, exact)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            fn = getattr(mod, func_name, None)
            if callable(fn):
                return fn

    # Glob fallback across scripts/pages and scripts/
    for p in list(PAGES_DIR.glob(fallbacks_glob)) + list(SCRIPTS_DIR.glob(fallbacks_glob)):
        spec = importlib.util.spec_from_file_location(p.stem, p)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore
            fn = getattr(mod, func_name, None)
            if callable(fn):
                return fn

    return None

# The real file is 4_Occupancy.py and must expose render_occupancy(df, sources)
render_fn = _load_callable(
    preferred_file="4_Occupancy.py",
    fallbacks_glob="*Occupancy*.py",
    func_name="render_occupancy",
)
if not callable(render_fn):
    raise ImportError(
        "Could not locate a callable render_occupancy(df, sources) in scripts/pages/4_Occupancy.py (or any *Occupancy*.py)."
    )

# IMPORTANT: do NOT call st.set_page_config here.
# The real page sets its own layout/title. Calling it twice can raise a StreamlitAPIException.

# Ensure a project is active (dashboard shim should have set this)
proj = st.session_state.get("current_project")
if not proj:
    st.error("No active project. Open a project first.")
    st.stop()

proj_path = Path(proj)

# Build the dashboard dataset
df_det, _notes = build_analysis_dataset(proj_path, use_stem_fallback=True)
if df_det is None or df_det.empty:
    st.error("No matched detections with audio. Complete Import → Audio mapping → Metadata mapping first.")
    st.stop()

# For Occupancy we don't need audio clips; pass minimal context
sources = {"project": str(proj_path)}

# Call the real page
render_fn(df_det, sources)

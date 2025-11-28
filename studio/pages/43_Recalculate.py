# studio/pages/43_Calculate.py
from __future__ import annotations
from pathlib import Path
import importlib
import importlib.util
import streamlit as st

st.set_page_config(page_title="PAMalytics — Recalculate", layout="wide", initial_sidebar_state="expanded")

def _load_recalc_renderer():
    """
    Prefer importing as a module: scripts.pages.5_Recalculate
    Fallback to loading by file path relative to this file.
    Returns a callable: render_recalculate(df, sources) or render_settings(df, sources).
    """
    # Try normal package import first
    for modname in ("scripts.pages.5_Recalculate", "scripts.pages.Recalculate", "scripts.pages.recalculate"):
        try:
            mod = importlib.import_module(modname)
            if hasattr(mod, "render_recalculate"):
                return getattr(mod, "render_recalculate")
            if hasattr(mod, "render_settings"):
                return getattr(mod, "render_settings")
        except Exception:
            pass

    # Fallback: resolve by file path near repo root
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "scripts" / "pages" / "5_Recalculate.py",
        here.parents[1] / "scripts" / "pages" / "5_Recalculate.py",
        here.parents[3] / "scripts" / "pages" / "5_Recalculate.py",
    ]
    for p in candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("recalc_page", str(p))
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            if hasattr(mod, "render_recalculate"):
                return getattr(mod, "render_recalculate")
            if hasattr(mod, "render_settings"):
                return getattr(mod, "render_settings")

    raise ImportError("Could not locate 5_Recalculate.py with render_recalculate()/render_settings().")

# require an active project & prepared dataset
proj = st.session_state.get("current_project")
if not proj:
    st.error("No active project. Open a project first.")
    st.stop()

pa_ready = bool(st.session_state.get("pa_ready"))
df_det = st.session_state.get("pa_df_det")
sources = st.session_state.get("pa_sources") or {}

# Ensure the actual project path is passed through to the page
sources.setdefault("project", str(Path(proj)))
sources.setdefault("project_root", str(Path(proj)))

if not pa_ready or df_det is None or getattr(df_det, "empty", True):
    st.error("Project data is not prepared yet. Complete the import/mapping steps first.")
    st.stop()

# top bar 
tb_l, tb_sp = st.columns([1, 9])
with tb_l:
    if st.button("◀ Back to Project hub"):
        st.session_state["route"] = "overview" 
        st.rerun()

# hand off to recalc page
try:
    render_page = _load_recalc_renderer()
except Exception as e:
    st.error(f"Failed to load recalculate page: {e}")
    st.stop()

render_page(df_det, sources)

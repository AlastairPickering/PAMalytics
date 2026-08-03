# code/pages/43_Recalculate.py
from __future__ import annotations
from pathlib import Path
import sys
import importlib.util
import json

import streamlit as st

HERE = Path(__file__).resolve()
STUDIO_ROOT = HERE.parents[1]
if str(STUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(STUDIO_ROOT))
from app_paths import AUTH_FILE


def _has_streamlit_context() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def sidebar_signout_button(label: str = "Sign out") -> None:
    with st.sidebar:
        st.markdown("---")
        if st.button(label, key="pa_signout_sidebar_recalc"):
            for k in list(st.session_state.keys()):
                if str(k) in {"auth_user", "route", "current_project", "pa_page"} or str(k).startswith((
                    "bd2_", "manual_", "import_", "audio_", "metadata_", "ds_", "dataset_",
                    "val_", "filters_", "pa_"
                )):
                    st.session_state.pop(k, None)

            try:
                AUTH_FILE.write_text(
                    json.dumps({"remember": False, "user": ""}),
                    encoding="utf-8",
                )
            except Exception:
                pass

            st.session_state["route"] = "login"
            st.switch_page("Home.py")


def _load_recalc_renderer():
    here = Path(__file__).resolve()
    studio_root = here.parents[1]
    page_path = studio_root / "core" / "page_impl" / "5_Recalculate.py"

    if not page_path.exists():
        raise FileNotFoundError(f"Recalculate page not found at {page_path}")

    spec = importlib.util.spec_from_file_location("page_impl_recalculate", str(page_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {page_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]

    if hasattr(mod, "render_recalculate"):
        return getattr(mod, "render_recalculate")
    if hasattr(mod, "render_settings"):
        return getattr(mod, "render_settings")

    raise ImportError(
        f"{page_path} does not define render_recalculate() or render_settings()"
    )


def render_page() -> None:
    st.set_page_config(
        page_title="PAMalytics - Recalculate",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    sidebar_signout_button()

    proj = st.session_state.get("current_project")
    if not proj:
        st.error("No active project. Open a project first.")
        st.stop()

    pa_ready = bool(st.session_state.get("pa_ready"))
    df_det = st.session_state.get("pa_df_det")
    sources = st.session_state.get("pa_sources") or {}

    sources.setdefault("project", str(Path(proj)))
    sources.setdefault("project_root", str(Path(proj)))

    if not pa_ready or df_det is None or getattr(df_det, "empty", True):
        st.error("Project data is not prepared yet. Complete the import/mapping steps first.")
        st.stop()

    tb_l, tb_sp = st.columns([1, 9])
    with tb_l:
        if st.button("◀ Back to Project hub"):
            st.session_state["route"] = "overview"
            st.rerun()

    try:
        render_recalculate_page = _load_recalc_renderer()
    except Exception as e:
        st.error(f"Failed to load recalculate page: {e}")
        st.stop()

    render_recalculate_page(df_det, sources)


if _has_streamlit_context():
    render_page()

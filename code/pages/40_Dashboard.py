# code/pages/40_Dashboard.py
from pathlib import Path
import sys
import json

import streamlit as st
from code.scripts.dashboard import render_dashboard
from code.utils import build_analysis_dataset as build_ds

THIS_FILE = Path(__file__).resolve()
STUDIO_ROOT = THIS_FILE.parent.parent
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
        if st.button(label, key="pa_signout_sidebar"):
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


def render_page() -> None:
    st.set_page_config(
        page_title="PAMalytics — Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    proj = st.session_state.get("current_project")
    if not proj:
        st.error("No active project. Open a project first.")
        st.stop()

    proj_path = Path(proj)
    df_det, _notes = build_ds(proj_path, use_stem_fallback=True)
    if df_det is None or df_det.empty:
        st.error("No matched detections with audio. Complete Import → Metadata mapping first.")
        st.stop()

    sources = {"project": str(proj_path)}
    st.session_state["pa_df_det"] = df_det
    st.session_state["pa_sources"] = sources
    st.session_state["pa_ready"] = True

    sidebar_signout_button()
    render_dashboard(df_det, sources)


if _has_streamlit_context():
    render_page()

# studio/pages/43_Calculate.py
from __future__ import annotations
from pathlib import Path
import json

import streamlit as st
from pamalytics.core.page_impl.Recalculate import render_recalculate

st.set_page_config(
    page_title="PAMalytics - Recalculate",
    layout="wide",
    initial_sidebar_state="expanded",
)

HERE = Path(__file__).resolve()
STUDIO_ROOT = HERE.parents[1]         
AUTH_FILE = STUDIO_ROOT / ".auth.json"


def sidebar_signout_button(label: str = "Sign out") -> None:
    """
    Show a sign-out button in the sidebar and log the user out when clicked.
    """
    with st.sidebar:
        st.markdown("---")
        if st.button(label, key="pa_signout_sidebar_recalc"):
            # Clear auth- and project-related session state
            for k in list(st.session_state.keys()):
                if str(k) in {"auth_user", "route", "current_project", "pa_page"} or str(k).startswith((
                    "bd2_", "manual_", "import_", "audio_", "metadata_", "ds_", "dataset_",
                    "val_", "filters_", "pa_"
                )):
                    st.session_state.pop(k, None)

            # Clear "remember me" file
            try:
                AUTH_FILE.write_text(
                    json.dumps({"remember": False, "user": ""}),
                    encoding="utf-8",
                )
            except Exception:
                pass

            # Route back to main entrypoint
            st.session_state["route"] = "login"
            st.switch_page("Home.py")


# Sidebar logout
sidebar_signout_button()

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


render_recalculate(df_det, sources)  # type: ignore

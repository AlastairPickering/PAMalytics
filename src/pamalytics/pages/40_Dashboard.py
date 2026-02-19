# code/pages/40_Dashboard.py
from pathlib import Path
import json

import streamlit as st
from pamalytics.scripts.dashboard import render_dashboard
from pamalytics.utils import build_analysis_dataset as build_ds

st.set_page_config(
    page_title="PAMalytics — Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Paths for auth file
THIS_FILE = Path(__file__).resolve()
STUDIO_ROOT = THIS_FILE.parent.parent        # code/
AUTH_FILE = STUDIO_ROOT / ".auth.json"


def sidebar_signout_button(label: str = "Sign out") -> None:
    """
    Show a sign-out button in the sidebar and log the user out when clicked.

    """
    with st.sidebar:
        st.markdown("---")
        if st.button(label, key="pa_signout_sidebar"):
            # Clear auth- and project-related session state
            for k in list(st.session_state.keys()):
                if str(k) in {"auth_user", "route", "current_project", "pa_page"} or str(k).startswith((
                    "bd2_", "manual_", "import_", "audio_", "metadata_", "ds_", "dataset_",
                    "val_", "filters_", "pa_"
                )):
                    st.session_state.pop(k, None)

            # Clear "remember me"
            try:
                AUTH_FILE.write_text(
                    json.dumps({"remember": False, "user": ""}),
                    encoding="utf-8",
                )
            except Exception:
                pass

            st.session_state["route"] = "login"
            st.switch_page("app.py")


# Page start
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

# Sidebar logout 
sidebar_signout_button()

# Render dashboard
render_dashboard(df_det, sources)

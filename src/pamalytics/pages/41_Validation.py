# 41_PAM_Validation.py
# Streamlit page shim 
# Expects a callable: render_validation(df, sources)

from pathlib import Path
import json
from pamalytics.core.page_impl.Validate import render_validation

import streamlit as st

THIS_FILE    = Path(__file__).resolve()        
STUDIO_ROOT  = THIS_FILE.parents[1]            # .../code

AUTH_FILE = STUDIO_ROOT / ".auth.json"


def sidebar_signout_button(label: str = "Sign out") -> None:
    """
    Show a sign-out button in the sidebar and log the user out when clicked.

    """
    with st.sidebar:
        st.markdown("---")
        if st.button(label, key="pa_signout_sidebar_validate"):
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

            # Route back to the main Home.py login entrypoint
            st.session_state["route"] = "login"
            st.switch_page("Home.py")


st.set_page_config(page_title="PAMalytics — Validation", layout="wide", initial_sidebar_state="expanded")

# Sidebar sign out
sidebar_signout_button()

# Load analysis data prepared by the Dashboard launcher
def _pull_from_state(*candidates):
    for k in candidates:
        if k in st.session_state and st.session_state[k] is not None:
            return st.session_state[k]
    return None

df_det   = _pull_from_state("pa_df_det", "df_det", "analysis_df", "detections_df")
sources  = _pull_from_state("pa_sources", "sources")

if df_det is None:
    st.error("Validation cannot start because the analysis dataset is not initialised. "
             "Open the PAM Dashboard first so it can prepare the data.")
    st.stop()

render_validation(df_det, sources)

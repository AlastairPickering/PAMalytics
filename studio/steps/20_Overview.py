import streamlit as st
from pathlib import Path
from studio.core.ui import hide_chrome, require_login
from studio.core.project import load_project

st.set_page_config(
    page_title="Overview • PAMalytics Studio",
    layout="wide",
    initial_sidebar_state="collapsed",
)

hide_chrome(hide_sidebar=True, hide_header=True)
require_login()

proj_path_str = st.session_state.get("current_project")
if not proj_path_str:
    st.warning("No project selected. Return to Project Hub.")
    if st.button("Back to Project Hub"):
        try:
            st.switch_page("studio/pages/10_Project_Hub.py")
        except Exception:
            st.experimental_rerun()
    st.stop()

proj_path = Path(proj_path_str)
data = load_project(proj_path)

st.title("Overview")
st.caption(f"Project: **{data['name']}** • Use case: `{data['use_case']}` • Timezone: `{data.get('tz','UTC')}`")

st.info("Next step for Use case 1: Import results (we’ll add the no-code importer on `30_Data_Import.py`).")

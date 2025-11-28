# studio/pages/40_Dashboard.py
from pathlib import Path
import streamlit as st
from scripts.dashboard import render_dashboard

st.set_page_config(
    page_title="PAMalytics — Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

from studio.utils import build_analysis_dataset as build_ds, project_path as project_path_func

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

# Render dashboard
render_dashboard(df_det, sources)

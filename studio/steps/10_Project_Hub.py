import streamlit as st
from pathlib import Path
from studio.core.ui import hide_chrome, require_login
from studio.core.project import list_projects, create_project, load_project, touch_last_opened

st.set_page_config(page_title="Project Hub • PAMalytics Studio", layout="wide", initial_sidebar_state="collapsed")
hide_chrome(hide_sidebar=True, hide_header=True)
require_login()  # will bounce to pages/00_Login.py if not logged in

st.title("Project Hub")

with st.expander("Create a new project", expanded=True):
    name = st.text_input("Project name", placeholder="e.g. Sabah 2024 – external results")
    use_case = st.radio(
        "Use case",
        options=["external_results", "pipeline"],
        index=0,
        format_func=lambda x: "I already have classifier results" if x == "external_results" else "I have audio; I want to run a classifier",
        horizontal=True,
    )
    if st.button("Create project", type="primary", disabled=(not name.strip())):
        folder = create_project(name.strip(), use_case, created_by=st.session_state.get("auth_user"))
        touch_last_opened(folder)
        st.session_state.current_project = str(folder)
        st.success(f"Created project: `{folder.name}`")
        try:
            st.switch_page("pages/20_Overview.py")
        except Exception:
            try:
                st.rerun()
            except Exception:
                pass

st.subheader("Recent projects")
projects = list_projects()
if not projects:
    st.caption("No projects yet. Create one above.")
else:
    for p in projects:
        data = load_project(p)
        cols = st.columns([3, 2, 2, 1])
        cols[0].markdown(f"**{data.get('name','(unnamed)')}**  \n`{p.name}`")
        cols[1].write(f"Use case: `{data.get('use_case')}`")
        cols[2].write(f"Timezone: `{data.get('tz','UTC')}`")
        if cols[3].button("Open", key=f"open_{p.name}"):
            st.session_state.current_project = str(p)
            touch_last_opened(p)
            try:
                st.switch_page("pages/20_Overview.py")
            except Exception:
                try:
                    st.rerun()
                except Exception:
                    pass

if st.session_state.get("current_project"):
    st.success(f"Active project: `{Path(st.session_state.current_project).name}`")

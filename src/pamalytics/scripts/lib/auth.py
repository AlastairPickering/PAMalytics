# scripts/ui/auth.py
from __future__ import annotations
import streamlit as st
from pathlib import Path

AUTH_KEYS = ("user_id", "user_name", "username")

def is_logged_in() -> bool:
    return any(st.session_state.get(k) for k in AUTH_KEYS)

def require_login() -> None:
    if is_logged_in():
        return
    try:
        st.switch_page("Home.py")  # Streamlit >=1.32
    except Exception:
        st.info("Please sign in on the Home page.")
        st.stop()

def logout_and_return() -> None:
    # Clear only auth and volatile UI state. Never touch user dataframes.
    for k in list(st.session_state.keys()):
        if str(k).startswith(("user_", "username", "active_dataset", "dataset_", "ds_", "val_", "filters_", "pa_")):
            st.session_state.pop(k, None)
    try:
        st.switch_page("Home.py")
    except Exception:
        st.rerun()

def account_sidebar() -> None:
    """Call on pages that have a sidebar (e.g., Dashboard)."""
    with st.sidebar:
        st.markdown("### Account")
        name = st.session_state.get("user_name") or st.session_state.get("username") or "User"
        st.caption(f"Signed in as **{name}**")
        if st.button("Sign out", use_container_width=True, key="logout_sidebar"):
            logout_and_return()

def top_right_logout() -> None:
    """Call on ingestion pages (no sidebar). Renders a top-right Sign out."""
    _, col_btn = st.columns([1, 0.18])
    with col_btn:
        st.write("")  # spacer for alignment
        if st.button("Sign out", key="logout_inline"):
            logout_and_return()

# Logos
def find_logo(project_root: str | Path, names=("pamalytics_logo.png","pamalytics_logo.svg",
                                               "logo.png","logo.svg")) -> Path | None:
    roots = [Path(project_root)/"assets", Path(project_root), Path.cwd()]
    for r in roots:
        for n in names:
            p = r / n
            if p.exists():
                return p
    return None

def render_logo(project_root: str | Path, max_width: int = 320) -> None:
    p = find_logo(project_root)
    if p:
        st.image(str(p), width=max_width)

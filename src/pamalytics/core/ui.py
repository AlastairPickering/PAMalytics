import streamlit as st

def hide_chrome(hide_sidebar: bool = True, hide_header: bool = True) -> None:
    css = ["<style>"]
    if hide_sidebar:
        css += [
            '[data-testid="stSidebar"] { display: none !important; }',
            '[data-testid="stSidebarNav"] { display: none !important; }',
            '@media (min-width: 0px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }',
        ]
    if hide_header:
        css += ['header { visibility: hidden; }', 'footer { visibility: hidden; }', '#MainMenu { visibility: hidden; }']
    css += ["</style>"]
    st.markdown("\n".join(css), unsafe_allow_html=True)

def require_login(redirect_page: str = "pages/00_Login.py") -> None:
    if not st.session_state.get("auth_user"):
        try:
            st.switch_page(redirect_page)
        except Exception:
            st.warning("Please log in to continue.")
            st.stop()

import streamlit as st
from studio.core.ui import hide_chrome
from streamlit_extras.switch_page_button import switch_page   # <-- add this

st.set_page_config(page_title="Login • PAMalytics Studio", layout="wide", initial_sidebar_state="collapsed")
hide_chrome(hide_sidebar=True, hide_header=True)

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

st.title("Login")

with st.form("login_form", clear_on_submit=False):
    user = st.text_input("Username")
    pin = st.text_input("PIN (optional)", type="password")
    submit = st.form_submit_button("Sign in")

if submit:
    if not user.strip():
        st.error("Please enter a username.")
    else:
        st.session_state.auth_user = user.strip()
        # Navigate to page by its filename stem ("10_Project_Hub")
        switch_page("10_Project_Hub")      # <-- this reliably switches pages

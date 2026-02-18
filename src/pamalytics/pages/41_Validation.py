# 41_PAM_Validation.py
# Streamlit page shim 
# Expects a callable: render_validation(df, sources)

from pathlib import Path
import sys
import importlib.util
import json

import streamlit as st

THIS_FILE    = Path(__file__).resolve()        
STUDIO_ROOT  = THIS_FILE.parents[1]            # .../code
REPO_ROOT    = STUDIO_ROOT.parent              # repo root

SCRIPTS_DIR   = STUDIO_ROOT / "scripts"        # .../code/scripts
CORE_DIR      = STUDIO_ROOT / "core"           # .../code/core
PAGE_IMPL_DIR = CORE_DIR / "page_impl"         # .../code/core/page_impl

REAL_PAGE = PAGE_IMPL_DIR / "1_Validate.py"    

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


# Make sure relevant dirs are importable
for p in (SCRIPTS_DIR, CORE_DIR, PAGE_IMPL_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

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

# Dynamically import scripts/pages/1_Validate.py and call render_validation
if not REAL_PAGE.exists():
    st.error(f"Could not find the Validation page at:\n`{REAL_PAGE}`")
    st.stop()

spec = importlib.util.spec_from_file_location("page_validate", REAL_PAGE)
if spec is None or spec.loader is None:
    st.error("Could not load the Validation page module (spec/loader not available).")
    st.stop()

mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)  # type: ignore
except Exception as e:
    st.error(f"Import error in 1_Validate.py: {e}")
    st.stop()

# Prefer an explicitly named entrypoint; fail if absent.
render_fn = getattr(mod, "render_validation", None)

if render_fn is None or not callable(render_fn):
    st.error(
        "The Validation page (`scripts/pages/1_Validate.py`) must expose a callable "
        "`render_validation(df, sources)`. Please add that function."
    )
    st.stop()

try:
    render_fn(df_det, sources)
except Exception as e:
    st.error(f"Error while rendering Validation: {e}")

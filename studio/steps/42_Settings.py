# 42_PAM_Settings.py
# Streamlit page shim that loads your real Settings page from scripts/pages/3_Settings.py
# Expects a callable: render_settings(df, sources)

from pathlib import Path
import sys
import importlib.util
import streamlit as st

# --- Locate repo roots ---
THIS_FILE   = Path(__file__).resolve()
STUDIO_DIR  = THIS_FILE.parents[1]          # <repo>/studio
REPO_ROOT   = STUDIO_DIR.parent             # <repo>
SCRIPTS_DIR = REPO_ROOT / "scripts"
REAL_PAGE   = SCRIPTS_DIR / "pages" / "3_Settings.py"

# Make sure scripts/ is importable (for any intra-module imports your page does)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Page chrome — let the main dashboard control visibility.
st.set_page_config(page_title="PAMalytics — Settings", layout="wide", initial_sidebar_state="expanded")

# ---- Load analysis data prepared by the Dashboard launcher (from session_state) ----
def _pull_from_state(*candidates):
    for k in candidates:
        if k in st.session_state and st.session_state[k] is not None:
            return st.session_state[k]
    return None

df_det   = _pull_from_state("pa_df_det", "df_det", "analysis_df", "detections_df")
sources  = _pull_from_state("pa_sources", "sources")

if df_det is None:
    st.error("Settings cannot start because the analysis dataset is not initialised. "
             "Open the PAM Dashboard first so it can prepare the data.")
    st.stop()

# Dynamically import scripts/pages/3_Settings.py and call render_settings
if not REAL_PAGE.exists():
    st.error(f"Could not find the Settings page at:\n`{REAL_PAGE}`")
    st.stop()

spec = importlib.util.spec_from_file_location("page_settings", REAL_PAGE)
if spec is None or spec.loader is None:
    st.error("Could not load the Settings page module (spec/loader not available).")
    st.stop()

mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)  # type: ignore
except Exception as e:
    st.error(f"Import error in 3_Settings.py: {e}")
    st.stop()

# Prefer an explicitly named entrypoint; fail loudly if absent.
render_fn = getattr(mod, "render_settings", None)

if render_fn is None or not callable(render_fn):
    st.error(
        "The Settings page (`scripts/pages/3_Settings.py`) must expose a callable "
        "`render_settings(df, sources)`. Please add that function."
    )
    st.stop()

# ---- Render Settings page ----
try:
    render_fn(df_det, sources)
except Exception as e:
    st.error(f"Error while rendering Settings: {e}")

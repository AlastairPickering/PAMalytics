# 41_PAM_Validation.py
# Streamlit page shim that loads your real Validation page from scripts/pages/1_Validate.py
# Expects a callable: render_validation(df, sources)

from pathlib import Path
import sys
import importlib.util
import streamlit as st

# --- Locate repo roots ---
THIS_FILE   = Path(__file__).resolve()
STUDIO_DIR  = THIS_FILE.parents[1]          # <repo>/studio
REPO_ROOT   = STUDIO_DIR.parent             # <repo>
SCRIPTS_DIR = REPO_ROOT / "scripts"
REAL_PAGE   = SCRIPTS_DIR / "pages" / "1_Validate.py"

# Make sure scripts/ is importable (for any intra-module imports your page does)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Page chrome — let the main dashboard control visibility; we just ensure expanded here.
st.set_page_config(page_title="PAMalytics — Validation", layout="wide", initial_sidebar_state="expanded")

# ---- Load analysis data prepared by the Dashboard launcher (from session_state) ----
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

# ---- Dynamically import scripts/pages/1_Validate.py and call render_validation ----
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

# Prefer an explicitly named entrypoint; fail loudly if absent.
render_fn = getattr(mod, "render_validation", None)

if render_fn is None or not callable(render_fn):
    st.error(
        "The Validation page (`scripts/pages/1_Validate.py`) must expose a callable "
        "`render_validation(df, sources)`. Please add that function."
    )
    st.stop()

# ---- Render your real Validation page ----
try:
    render_fn(df_det, sources)
except Exception as e:
    st.error(f"Error while rendering Validation: {e}")

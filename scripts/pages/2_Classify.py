# pages/2_Classify.py
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional
import streamlit as st

# Page config
st.set_page_config(layout="wide", page_title="Pipeline")
st.title("Launch classifier")

# Paths
SCRIPTS_DIR = Path(__file__).resolve().parent.parent       # …/scripts
REPO_ROOT   = SCRIPTS_DIR.parent                           # repo root
DEFAULT_PIPELINE = SCRIPTS_DIR / "pipeline.py"             # pipeline is under scripts/
DEFAULT_AUDIO_DIR = (REPO_ROOT / "audio").resolve()        # default audio folder = repo_root/audio

# Try to import shared dirs; otherwise fall back
try:
    from config import RESULTS_DIR  # type: ignore
except Exception:
    RESULTS_DIR = REPO_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = RESULTS_DIR / "app_state.json"  # persists choices (e.g. audio folder)

# Tiny persistence
def load_app_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_app_state(data: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass

# Session state
ss = st.session_state

# Persistent choices
if "pipeline_script" not in ss:
    ss["pipeline_script"] = str(DEFAULT_PIPELINE)

saved_audio = load_app_state().get("audio_base_dir", "").strip()
if "audio_base_dir" not in ss or not str(ss["audio_base_dir"]).strip():
    ss["audio_base_dir"] = saved_audio or str(DEFAULT_AUDIO_DIR)

if "extra_args" not in ss:
    ss["extra_args"] = ""

if "pipeline_log" not in ss:
    ss["pipeline_log"] = str((RESULTS_DIR / "pipeline.log").resolve())

if "pipeline_status" not in ss:
    ss["pipeline_status"] = str((RESULTS_DIR / "pipeline_status.json").resolve())

# Process handle + UI state
ss.setdefault("proc", None)                 # subprocess.Popen handle
ss.setdefault("last_start", None)
ss.setdefault("auto_refresh", True)         # default: on
ss.setdefault("refresh_seconds", 2)         # refresh every 2s by default

# Helpers
def is_running(p: Optional[subprocess.Popen]) -> bool:
    try:
        return p is not None and p.poll() is None
    except Exception:
        return False

def tail(path: Path, n: int = 800) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""

def read_status(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _safe_output_path(wanted: Path, fallback: Path, kind: str) -> Path:
    try:
        wanted = wanted.resolve()
    except Exception:
        return fallback.resolve()
    pages_dir = (SCRIPTS_DIR / "pages").resolve()
    bad_ext = wanted.suffix.lower() in {".py", ".pyc", ".pyo"}
    under_pages = pages_dir in wanted.parents or wanted == pages_dir
    if bad_ext or under_pages:
        st.warning(f"{kind} path adjusted to a safe location: {fallback}")
        return fallback.resolve()
    return wanted

# Start/stop
def start_pipeline():
    if is_running(ss.get("proc")):
        st.warning("Pipeline is already running.")
        return

    script = Path(ss["pipeline_script"])
    if not script.exists():
        st.error(f"Pipeline script not found: {script}")
        return

    # Build command
    cmd = [sys.executable, "-u", str(script)]
    xargs = ss["extra_args"].strip().split() if ss["extra_args"].strip() else []

    # Safe output paths
    safe_log    = _safe_output_path(Path(ss["pipeline_log"]),    RESULTS_DIR / "pipeline.log",    "Log file")
    safe_status = _safe_output_path(Path(ss["pipeline_status"]), RESULTS_DIR / "pipeline_status.json", "Status file")

    # Enforce audio folder from the UI 
    if "--audio-dir" in xargs:
        toks, skip = [], False
        for t in xargs:
            if skip:
                skip = False
                continue
            if t == "--audio-dir":
                skip = True
                continue
            toks.append(t)
        xargs = toks
    if ss.get("audio_base_dir"):
        xargs += ["--audio-dir", ss["audio_base_dir"]]

    # Provide defaults for status/log/results if not present
    if "--status-file" not in xargs:
        xargs += ["--status-file", str(safe_status)]
    if "--results-dir" not in xargs:
        xargs += ["--results-dir", str(RESULTS_DIR)]

    cmd += xargs

    # Overwrite log each run
    safe_log.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(safe_log, "w", buffering=1, encoding="utf-8")
    log_fh.write(f"=== Launch {datetime.now().isoformat()} ===\n")
    log_fh.write(f"CMD: {' '.join(cmd)}\n")

    # Run from scripts
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    ss["proc"] = subprocess.Popen(
        cmd,
        cwd=str(SCRIPTS_DIR),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    ss["last_start"] = time.time()
    st.success("Pipeline started.")

def stop_pipeline():
    p = ss.get("proc")
    if not is_running(p):
        st.info("Pipeline is not running.")
        return
    p.terminate()
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()
    ss["proc"] = None
    st.success("Pipeline stopped.")

# Controls
left, mid, right = st.columns([1.5, 1.5, 1.2])

with left:
    st.text_input("Pipeline script", key="pipeline_script", help="Path to scripts/pipeline.py")
    st.text_input(
        "Audio folder",
        key="audio_base_dir",
        help="Always passed to the pipeline as --audio-dir.",
    )
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Save as default"):
            state = load_app_state()
            state["audio_base_dir"] = ss["audio_base_dir"]
            save_app_state(state)
            st.success("Audio folder saved as default.")
    with colB:
        if st.button("Reset to default"):
            ss["audio_base_dir"] = str(DEFAULT_AUDIO_DIR)
            st.success("Audio folder reset to default.")
    # Quick existence check
    audio_path = Path(ss["audio_base_dir"]) if ss.get("audio_base_dir") else None
    exists = bool(audio_path and audio_path.exists() and audio_path.is_dir())
    st.caption(("✅ Folder exists" if exists else ("⚠️ Folder not found" if ss.get("audio_base_dir") else "— No folder set")))

with mid:
    st.text_input(
        "Extra args (optional)",
        key="extra_args",
        help="Other flags, e.g. --tau 0.5 --kn 2,3."
    )
    st.text_input("Status JSON", key="pipeline_status", help="Your pipeline writes progress here.")

with right:
    st.text_input("Log file", key="pipeline_log")
    run_c1, run_c2 = st.columns(2)
    with run_c1:
        if st.button("▶️ Start", use_container_width=True):
            start_pipeline()
    with run_c2:
        if st.button("⏹ Stop", use_container_width=True):
            stop_pipeline()

# Status
running = is_running(ss.get("proc"))
status_path = Path(ss["pipeline_status"])
status_data = read_status(status_path)

st.subheader("Status")
st.caption(f"Status file: {status_path}")
if status_data:
    prog = float(status_data.get("progress", 0.0))
    st.progress(min(max(prog, 0.0), 1.0), text=status_data.get("message", ""))
    cols = st.columns(5)
    cols[0].metric("State", status_data.get("state", "—"))
    cols[1].metric("Done", status_data.get("done", "—"))
    cols[2].metric("Total", status_data.get("total", "—"))
    cols[3].metric("Current", status_data.get("current", "—"))
    cols[4].metric("Started", status_data.get("started", "—"))
elif running:
    st.info("Running… (no status file detected yet).")
else:
    st.info("Idle.")

# Logs
st.subheader("Logs")
log_path = Path(ss["pipeline_log"])
st.caption(f"Log file: {log_path}")

ctrl = st.columns([1, 1, 2, 2])
with ctrl[0]:
    if st.button("Refresh now"):
        pass  # Streamlit reruns on click
with ctrl[1]:
    if st.button("Clear log"):
        try:
            log_path.write_text("", encoding="utf-8")
            st.success("Log cleared.")
        except Exception as e:
            st.error(f"Could not clear log: {e}")
with ctrl[2]:
    st.toggle("Auto refresh", key="auto_refresh", help="Automatically refresh status & logs while running.")
with ctrl[3]:
    st.slider("Interval (seconds)", 1, 10, key="refresh_seconds")

st.code(tail(log_path) or "(no logs yet)", language="bash")

# Auto refresh loop
if ss.get("auto_refresh", False):
    time.sleep(int(ss.get("refresh_seconds", 2)))
    st.rerun()

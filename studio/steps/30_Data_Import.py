import streamlit as st
from pathlib import Path
import pandas as pd
import json

from core.ui import hide_chrome, require_login
from core.project import load_project

st.set_page_config(
    page_title="Import results • PAMalytics Studio",
    layout="wide",
    initial_sidebar_state="collapsed",
)

hide_chrome(hide_sidebar=True, hide_header=True)
require_login()

proj_path_str = st.session_state.get("current_project")
if not proj_path_str:
    st.warning("No project selected. Return to Project Hub.")
    st.stop()

proj_path = Path(proj_path_str)
proj = load_project(proj_path)

st.title("Import results (Use case 1)")
st.caption("Load an external classifier results table, map its columns, preview, and save a normalised copy for the rest of the workflow.")

st.subheader("1) Load your results file")
src = st.file_uploader("CSV or Parquet", type=["csv", "tsv", "parquet"])

if "import_df" not in st.session_state:
    st.session_state.import_df = None

if src is not None:
    try:
        if src.name.endswith(".parquet"):
            df = pd.read_parquet(src)
        else:
            try:
                df = pd.read_csv(src)
            except Exception:
                src.seek(0)
                df = pd.read_csv(src, sep="\t")
        st.session_state.import_df = df
    except Exception as e:
        st.error(f"Could not read file: {e}")

df = st.session_state.import_df
if df is None:
    st.stop()

st.write("Preview (first 20 rows):")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("2) Map columns to the minimal schema")
required = {
    "source_file": None,
    "start_s": None,
    "end_s": None,
    "label": None,
    "score": None,  # optional if upstream thresholded
}
optional = {"timestamp_utc": None}

cols = list(df.columns)

def pick(label, candidates: set[str]):
    guess = next((c for c in cols if c.lower() in candidates), None)
    return st.selectbox(label, ["—"] + cols, index=(cols.index(guess) + 1) if guess in cols else 0)

st.markdown("**Required fields**")
source_file_col = pick("File path / name → `source_file`", {"filename","file","filepath","path","source_file"})
start_col       = pick("Start time (seconds) → `start_s`", {"start","start_s","begin","onset","start_sec","start_time"})
end_col         = pick("End time (seconds) → `end_s`", {"end","end_s","offset","end_sec","end_time"})
label_col       = pick("Label / species → `label`", {"label","species","class","prediction"})
score_col       = pick("Confidence [0,1] → `score` (leave '—' if not provided)", {"score","confidence","prob","probability","p_hat"})

st.markdown("**Optional**")
ts_col          = pick("Timestamp (UTC) → `timestamp_utc` (optional)", {"timestamp_utc","timestamp","datetime","utc"})

missing = []
if source_file_col == "—": missing.append("source_file")
if start_col == "—":       missing.append("start_s")
if end_col == "—":         missing.append("end_s")
if label_col == "—":       missing.append("label")

st.subheader("3) Build the normalised table")
convert_ms = st.checkbox("My start/end appear to be in milliseconds (convert to seconds)", value=False)
fill_default_label = st.text_input("If `label` is missing or generic, set a default (optional)", value="")

disabled = bool(missing)
if disabled:
    st.warning("Please map required fields: " + ", ".join(missing))

if st.button("Create normalised copy", type="primary", disabled=disabled):
    norm = pd.DataFrame()
    norm["source_file"] = df[source_file_col].astype(str)

    def to_seconds(series):
        s = pd.to_numeric(series, errors="coerce")
        if convert_ms:
            s = s / 1000.0
        return s

    norm["start_s"] = to_seconds(df[start_col])
    norm["end_s"]   = to_seconds(df[end_col])

    lbl = df[label_col].astype(str)
    if fill_default_label:
        mask = lbl.isna() | (lbl.str.strip() == "") | (lbl.str.lower().isin(["nan","none"]))
        if mask.any():
            lbl = lbl.mask(mask, fill_default_label)
    norm["label"] = lbl

    if score_col != "—":
        norm["score"] = pd.to_numeric(df[score_col], errors="coerce")
    else:
        norm["score"] = pd.NA

    if ts_col != "—":
        norm["timestamp_utc"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

    issues = []
    if (norm["end_s"] < norm["start_s"]).any():
        issues.append("Some rows have end_s < start_s.")
    if norm["start_s"].isna().any() or norm["end_s"].isna().any():
        issues.append("Some rows have invalid start/end times.")
    if issues:
        for msg in issues: st.error(msg)
        st.stop()

    out_dir = Path(proj["paths"]["data_normalised"])
    out_dir = (Path(proj_path) / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "detections_normalised.csv"
    norm.to_csv(out_csv, index=False)

    manifest = {
        "source_file": source_file_col,
        "start_s": start_col,
        "end_s": end_col,
        "label": label_col,
        "score": None if score_col == "—" else score_col,
        "timestamp_utc": None if ts_col == "—" else ts_col,
        "convert_ms": bool(convert_ms),
        "default_label": fill_default_label or None,
        "rows": int(len(norm)),
    }
    ws_dir = (Path(proj_path) / "workspace")
    ws_dir.mkdir(parents=True, exist_ok=True)
    (ws_dir / "ingest_mapping.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    st.success(f"Saved normalised detections to: `{out_csv}`")
    st.caption("Next: Audio path resolver, Metadata joins, then Dashboard/Validate/Post-processing/Occupancy.")

# pages/1_Validate.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import math
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from pandas.errors import DtypeWarning
import warnings
from typing import Optional, Tuple
from config import RAW_AUDIO_DIR, RESULTS_DIR

st.set_page_config(layout="wide", page_title="Validate")

# Session defaults 
AUDIO_DEFAULT_DIR = RAW_AUDIO_DIR / "processed" / "present"
if "audio_base_dir" not in st.session_state:
    st.session_state["audio_base_dir"] = str(AUDIO_DEFAULT_DIR)

if "filename_level_path" not in st.session_state:
    st.session_state["filename_level_path"] = str(RESULTS_DIR / "filename_level.csv")

# Prefer a user-set segment file; else try cleaned.xlsx, else csv
if "segment_results_path" not in st.session_state:
    cleaned = RESULTS_DIR / "merged_classification_results_cleaned.xlsx"
    csv     = RESULTS_DIR / "merged_classification_results.csv"
    st.session_state["segment_results_path"] = str(cleaned if cleaned.exists() else csv)

# UserLabel helpers
def ensure_userlabel(df_in: pd.DataFrame) -> pd.DataFrame:
    """Ensure an editable UserLabel column exists and uses empty strings (not NaN/'nan')."""
    df = df_in.copy()
    if "UserLabel" not in df.columns:
        df["UserLabel"] = ""
    else:
        df["UserLabel"] = df["UserLabel"].fillna("").replace({"nan": ""})
    return df

def with_effective_labels(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Add FinalLabelEffective (UserLabel if set, else FinalLabel),
    is_present from FinalLabelEffective, and Changed flag.
    Robust to NaN/'nan' in UserLabel.
    """
    df = ensure_userlabel(df_in)
    if "FinalLabel" not in df.columns:
        st.error("Missing 'FinalLabel' in filename-level data.")
        st.stop()

    u = (df["UserLabel"].fillna("").astype(str).replace({"nan": ""}).str.strip().str.lower())
    m = (df["FinalLabel"].astype(str).str.strip().str.lower())

    eff = np.where(u != "", u, m)
    df = df.copy()
    df["FinalLabelEffective"] = eff
    df["is_present"] = (eff == "present").astype(int)
    df["Changed"] = (u != "") & (u != m)
    return df

def to_stem_lower(x: str) -> str:
    return Path(str(x)).stem.lower()

def load_filename_level() -> pd.DataFrame:
    """Load active filename-level file chosen in Settings; ensure UserLabel is clean."""
    path = Path(st.session_state.get("filename_level_path", RESULTS_DIR / "filename_level.csv"))
    if not path.exists():
        st.error(f"Filename-level file not found: {path}")
        st.stop()
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            df = pd.read_excel(path, sheet_name="filename_level")
        except Exception:
            df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return ensure_userlabel(df)

def save_filename_level(path: Path, df: pd.DataFrame) -> None:
    """Save back to the same file type (CSV/XLSX). Only UserLabel is edited by this page."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, sheet_name="filename_level", index=False)
        except Exception:
            df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)

# Segment loading
def find_probability_column(seg: pd.DataFrame) -> Optional[str]:
    direct = [c for c in seg.columns if c.lower() in
              ("probability", "prob", "pred_prob", "pred_probability", "score", "p")]
    if direct:
        return direct[0]
    for c in seg.columns:
        if "prob" in c.lower():
            return c
    return None

def find_filename_like_column(seg: pd.DataFrame) -> Optional[str]:
    candidates = ["filename", "audio_file", "file", "clip_id", "clip", "path"]
    for c in candidates:
        if c in seg.columns:
            return c
    for c in seg.columns:
        if seg[c].dtype == "object":
            return c
    return None

def load_segments() -> Optional[pd.DataFrame]:
    """Load segment-level results; be tolerant of mixed dtypes."""
    path = Path(st.session_state.get("segment_results_path", RESULTS_DIR / "merged_classification_results.csv"))
    if not path.exists():
        st.warning(f"Segment file not found: {path}")
        return None
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            seg = pd.read_excel(path)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DtypeWarning)
                seg = pd.read_csv(path, low_memory=False)
    except Exception as e:
        st.warning(f"Failed to read segments: {e}")
        return None
    if seg.empty:
        st.warning("Segment file is empty.")
        return None
    return seg

def make_clip_prob(seg: pd.DataFrame) -> pd.DataFrame:
    """
    From segment-level results, compute per-file clip_prob = max(probability) across segments.
    Normalises probability to 0–1 if needed. Joins on filename_stem.
    """
    seg = seg.copy()

    fname_col = find_filename_like_column(seg)
    prob_col  = find_probability_column(seg)

    if fname_col is None or prob_col is None:
        st.warning("Could not detect filename/probability columns in the segment file.")
        return pd.DataFrame(columns=["filename_stem", "clip_prob"])

    seg["filename_stem"] = seg[fname_col].astype(str).map(to_stem_lower)
    seg[prob_col] = pd.to_numeric(seg[prob_col], errors="coerce")
    seg = seg.dropna(subset=[prob_col, "filename_stem"])

    max_val = seg[prob_col].max()
    if pd.notna(max_val) and max_val > 1:
        if max_val <= 100:
            seg[prob_col] = seg[prob_col] / 100.0
        else:
            seg[prob_col] = seg[prob_col].clip(0, 1)

    clipprob = (
        seg.groupby("filename_stem")[prob_col]
        .max()
        .reset_index()
        .rename(columns={prob_col: "clip_prob"})
    )
    clipprob["clip_prob"] = clipprob["clip_prob"].clip(0, 1)
    return clipprob

# Audio helpers
def _figure_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

@st.cache_data(show_spinner=False)
def make_thumbnail_and_audio(audio_path: str, preview_seconds: int) -> Tuple[Optional[bytes], Optional[bytes]]:
    """Return (thumbnail_png_bytes, audio_wav_bytes). Thumbnail uses first N seconds; audio is full file."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        y_thumb = y[: int(preview_seconds * sr)]
        fig, ax = plt.subplots(figsize=(3.0, 1.6), dpi=140)  # crisp but compact
        S = librosa.feature.melspectrogram(y=y_thumb, sr=sr, n_fft=1024, hop_length=512)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None, ax=ax)
        ax.set_axis_off()
        thumb_png = _figure_to_png(fig)

        abuf = io.BytesIO()
        sf.write(abuf, y, sr, format="WAV")
        abuf.seek(0)
        return thumb_png, abuf.read()
    except Exception:
        return None, None

def find_audio_path_by_stem(filename_stem: str) -> Optional[Path]:
    base = Path(st.session_state.get("audio_base_dir", str(AUDIO_DEFAULT_DIR)))
    if not filename_stem or not base.exists():
        return None
    for ext in (".wav", ".WAV", ".flac", ".FLAC"):
        p = base / f"{Path(filename_stem).stem}{ext}"
        if p.exists():
            return p
    lower_stem = Path(filename_stem).stem.lower()
    try:
        for p in base.iterdir():
            if p.is_file() and p.stem.lower() == lower_stem:
                return p
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def list_present_stems(base_dir_str: str) -> set:
    """
    Return a set of lowercase filename stems that actually exist
    in the current audio folder (present directory by default).
    """
    stems = set()
    try:
        base = Path(base_dir_str)
        if not base.exists():
            return stems
        for q in base.iterdir():
            if q.is_file() and q.suffix.lower() in {".wav", ".flac"}:
                stems.add(q.stem.lower())
    except Exception:
        pass
    return stems

# Load data
df_master = load_filename_level()
required_cols = ["filename", "FinalLabel"]
missing = [c for c in required_cols if c not in df_master.columns]
if missing:
    st.error(f"Missing required columns in filename-level data: {missing}")
    st.stop()

df = df_master.copy()
df["filename_stem"] = df["filename"].astype(str).map(to_stem_lower)
df = with_effective_labels(df)  # adds FinalLabelEffective, is_present, Changed

# Merge in clip_prob (best segment prob per file)
seg = load_segments()
if seg is not None:
    clipprob = make_clip_prob(seg)
    df = df.merge(clipprob, on="filename_stem", how="left")
else:
    df["clip_prob"] = np.nan
    st.caption("No segment file loaded — probabilities unavailable.")

# UI controls
st.header("Validate predictions")

top1, top2, top3, top4 = st.columns([1.0, 1.0, 1.0, 1.0])
with top1:
    show_label = st.selectbox("Show clips labelled", ["present", "absent", "all"], index=0)
with top2:
    min_prob = st.slider("Min clip probability", 0.0, 1.0, 0.0, 0.01)
with top3:
    sort_by = st.selectbox("Sort by", ["clip_prob", "filename"], index=0)
with top4:
    # spacer so the checkbox aligns with the other controls' labels
    st.markdown("<div style='height:1.95em'></div>", unsafe_allow_html=True)
    changed_only = st.checkbox("Changed only", value=False)

# Filter rows
df_view = df.copy()

# label filter
if show_label in ("present", "absent"):
    df_view = df_view[df_view["FinalLabelEffective"] == show_label]

# changed filter
if changed_only and "Changed" in df_view.columns:
    df_view = df_view[df_view["Changed"]]

# prob filter & sort key
df_view["clip_prob_f"] = pd.to_numeric(df_view["clip_prob"], errors="coerce").fillna(0.0)
df_view = df_view[df_view["clip_prob_f"] >= float(min_prob)]

# keep only files that actually exist in the current audio (present) folder
present_stems = list_present_stems(st.session_state.get("audio_base_dir", str(AUDIO_DEFAULT_DIR)))
if present_stems:
    df_view = df_view[df_view["filename_stem"].isin(present_stems)]
else:
    st.info("No audio files found in the selected audio folder. Nothing to display.")

# sorting
if sort_by == "clip_prob":
    df_view = df_view.sort_values(["clip_prob_f", "filename_stem"], ascending=[False, True])
else:
    df_view = df_view.sort_values("filename_stem")

# Grid controls
c1, c2, c3, c4 = st.columns(4)
with c1:
    NUM_PER_PAGE = st.number_input("Clips per page", min_value=6, max_value=60, value=12, step=6)
with c2:
    COLS_PER_ROW = st.slider("Columns per row", min_value=2, max_value=5, value=3)
with c3:
    THUMB_SEC = st.slider("Spectrogram preview (seconds)", min_value=5, max_value=60, value=60,
                          help="Thumbnail only; audio = full file")
with c4:
    PAGE = st.number_input("Page", min_value=1, value=1, step=1)

total = len(df_view)
start = (int(PAGE)-1) * int(NUM_PER_PAGE)
end = start + int(NUM_PER_PAGE)
page_df = df_view.iloc[start:end].reset_index(drop=True)

st.caption(f"Showing {len(page_df)} of {total} clips (page {PAGE})")

# Grid + proposed updates
updates = []
n_rows = math.ceil(len(page_df) / int(COLS_PER_ROW))

if page_df.empty:
    st.info("No clips match the current filters.")
else:
    page_df = page_df.assign(_stem=page_df["filename"].astype(str).map(to_stem_lower))

    for r in range(n_rows):
        cols = st.columns(int(COLS_PER_ROW))
        for c in range(int(COLS_PER_ROW)):
            i = r * int(COLS_PER_ROW) + c
            if i >= len(page_df):
                break
            row = page_df.iloc[i]
            with cols[c]:
                u = str(row.get("UserLabel", "")).strip().lower()
                m = str(row.get("FinalLabel", "")).strip().lower()
                badge = " 🟠 changed" if (u != "" and u != m) else ""
                prob_txt = f" · p={row['clip_prob_f']:.2f}" if pd.notna(row["clip_prob_f"]) else ""
                st.caption(f"**{row['filename']}**{prob_txt}{badge}")

                apath = find_audio_path_by_stem(row["_stem"])
                if apath and apath.exists():
                    thumb_png, audio_wav = make_thumbnail_and_audio(str(apath), int(THUMB_SEC))
                    if thumb_png:
                        st.image(thumb_png, width="stretch")  # replaced use_container_width
                    else:
                        st.info("No thumbnail")
                    if audio_wav:
                        st.audio(io.BytesIO(audio_wav), format="audio/wav")
                    else:
                        st.error("Audio unreadable")
                else:
                    st.error("Audio not found")

                eff = str(row["FinalLabelEffective"]).lower()
                idx = 1 if eff == "present" else 0
                choice = st.radio("Label", ["absent", "present"], index=idx,
                                  key=f"val_{start+i}", horizontal=True)

                proposed = "" if choice == m else choice
                if proposed != u:
                    updates.append({"filename": row["filename"], "UserLabel": proposed})

# Pending changes table
st.subheader("Pending changes (unsaved)")
if not updates:
    st.write("No pending changes.")
else:
    upd_df = pd.DataFrame(updates).drop_duplicates(subset=["filename", "UserLabel"])
    view_cols = ["filename", "FinalLabel", "UserLabel", "FinalLabelEffective", "clip_prob"]
    merged = upd_df.merge(df[view_cols], on="filename", how="left", suffixes=("", "_current"))
    merged = merged.sort_values(["clip_prob"], ascending=[False])
    st.dataframe(
        merged[["filename", "FinalLabel", "UserLabel", "FinalLabelEffective", "clip_prob"]],
        width="stretch"  # replaced use_container_width
    )

# Save
left, right = st.columns([1, 3])
with left:
    if st.button("Save all updates"):
        try:
            df_master2 = load_filename_level()  # fresh
            changes = 0
            for rec in updates:
                mask = df_master2["filename"].astype(str) == str(rec["filename"])
                if mask.any():
                    old = str(df_master2.loc[mask, "UserLabel"].iloc[0]) if "UserLabel" in df_master2.columns else ""
                    if old != rec["UserLabel"]:
                        if "UserLabel" not in df_master2.columns:
                            df_master2["UserLabel"] = ""
                        df_master2.loc[mask, "UserLabel"] = rec["UserLabel"]
                        changes += int(mask.sum())
            out_path = Path(st.session_state.get("filename_level_path", RESULTS_DIR / "filename_level.csv"))
            save_filename_level(out_path, df_master2)
            st.success(f"Saved {changes} update(s) to {out_path.name}.")
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to save updates: {e}")
with right:
    st.caption("Only the **UserLabel** column is updated. The original **FinalLabel** is never overwritten.")

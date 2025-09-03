# Dashboard.py
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import pydeck as pdk
import os
import math
from pathlib import Path
from pyproj import Transformer
import matplotlib.pyplot as plt
import librosa
import librosa.display
import io
import soundfile as sf
import base64
from typing import Optional, Tuple
from config import RAW_AUDIO_DIR, EMBEDDINGS_DIR, RESULTS_DIR, TARGET_LENGTH, DEVICE

st.set_page_config(layout="wide", page_title="Dashboard")
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"  # Disable the file watcher

# Defaults for Settings-Dashboard bridge
AUDIO_DEFAULT_DIR = RAW_AUDIO_DIR / "processed" / "present"
if "audio_base_dir" not in st.session_state:
    st.session_state["audio_base_dir"] = str(AUDIO_DEFAULT_DIR)

if "filename_level_path" not in st.session_state:
    st.session_state["filename_level_path"] = str(RESULTS_DIR / "filename_level.csv")

# Header + Logo
LOGO_PATHS = [RESULTS_DIR / "assets" / "logo-ci.png", Path("logo-ci.png")]

def _logo_b64(paths):
    for p in paths:
        try:
            data = Path(p).read_bytes()
            return base64.b64encode(data).decode()
        except Exception:
            continue
    return ""

_logo = _logo_b64(LOGO_PATHS)
if _logo:
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between;">
          <h1 style="margin:0;">Pileated Gibbon Detection Dashboard</h1>
          <img src="data:image/png;base64,{_logo}" style="height:5em; margin-left:1rem;" alt="logo">
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("Pileated Gibbon Detection Dashboard")
    st.caption("Logo not found (looked in RESULTS_DIR/assets/logo-ci.png and ./logo-ci.png)")

PROCESSED_DIR = RAW_AUDIO_DIR / "processed"
PRESENT_DIR   = PROCESSED_DIR / "present"

# Data sources
FILENAME_LEVEL_CSV   = RESULTS_DIR / "filename_level.csv"
FILENAME_LEVEL_XLSX  = RESULTS_DIR / "merged_classification_results_cleaned.xlsx"
FILENAME_LEVEL_SHEET = "filename_level"

def ensure_userlabel(df_in: pd.DataFrame) -> pd.DataFrame:
    """Ensure an editable UserLabel column exists and uses empty strings (not NaN/'nan')."""
    df = df_in.copy()
    if "UserLabel" not in df.columns:
        df["UserLabel"] = ""
    else:
        df["UserLabel"] = df["UserLabel"].fillna("").replace({"nan": ""})
    return df

def load_filename_level():
    """Load the active filename-level file chosen in Settings, else fall back."""
    chosen = Path(st.session_state.get("filename_level_path", ""))
    if str(chosen) and chosen.exists():
        if chosen.suffix.lower() in (".xlsx", ".xls"):
            try:
                df = pd.read_excel(chosen, sheet_name="filename_level")
            except Exception:
                df = pd.read_excel(chosen)
        else:
            df = pd.read_csv(chosen)
        return ensure_userlabel(df)

    if FILENAME_LEVEL_CSV.exists():
        return ensure_userlabel(pd.read_csv(FILENAME_LEVEL_CSV))

    if not FILENAME_LEVEL_XLSX.exists():
        st.error(f"Could not find {FILENAME_LEVEL_CSV.name} or {FILENAME_LEVEL_XLSX.name}.")
        st.stop()
    try:
        return ensure_userlabel(pd.read_excel(FILENAME_LEVEL_XLSX, sheet_name=FILENAME_LEVEL_SHEET))
    except ValueError:
        try:
            return ensure_userlabel(pd.read_excel(FILENAME_LEVEL_XLSX, sheet_name="Filename_Level"))
        except Exception as e:
            st.error(f"Unable to open sheet '{FILENAME_LEVEL_SHEET}' in {FILENAME_LEVEL_XLSX.name}: {e}")
            st.stop()

def with_effective_labels(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Add FinalLabelEffective (UserLabel if set, else FinalLabel),
    is_present from FinalLabelEffective, and Changed flag.
    Robust to NaN/'nan' in UserLabel.
    """
    df = df_in.copy()
    if "FinalLabel" not in df.columns:
        st.error("Missing 'FinalLabel' in filename-level data.")
        st.stop()

    df = ensure_userlabel(df)

    u = (
        df["UserLabel"]
        .fillna("")
        .astype(str)
        .replace({"nan": ""})
        .str.strip()
        .str.lower()
    )
    m = (
        df["FinalLabel"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    eff = np.where(u != "", u, m)
    df["FinalLabelEffective"] = eff
    df["is_present"] = (eff == "present").astype(int)
    df["Changed"] = (u != "") & (u != m)
    return df

def save_filename_level(path: Path, df: pd.DataFrame) -> None:
    """Save back to the same file type (CSV/XLSX)."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            with pd.ExcelWriter(path, engine="openpyxl", mode="w") as writer:
                df.to_excel(writer, sheet_name="filename_level", index=False)
        except Exception:
            df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)

df_raw = load_filename_level()

# Normalise + Effective labels
required_cols = ["filename", "FinalLabel"]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"Missing required columns in filename-level data: {missing}")
    st.stop()

df = with_effective_labels(df_raw)

def to_stem_lower(x: str) -> str:
    return Path(str(x)).stem.lower()

df["filename_stem"] = df["filename"].astype(str).map(to_stem_lower)

# UTM to lat/lon
if "utm_x" in df.columns and "utm_y" in df.columns:
    try:
        df["utm_x"] = pd.to_numeric(df["utm_x"], errors="coerce")
        df["utm_y"] = pd.to_numeric(df["utm_y"], errors="coerce")
        valid_mask = df["utm_x"].notna() & df["utm_y"].notna() & (df["utm_x"] != 0) & (df["utm_y"] != 0)
        df["lat"] = np.nan
        df["lon"] = np.nan
        if valid_mask.any():
            transformer = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)
            xs = df.loc[valid_mask, "utm_x"].to_numpy()
            ys = df.loc[valid_mask, "utm_y"].to_numpy()
            lons, lats = transformer.transform(xs, ys)
            df.loc[valid_mask, "lon"] = lons
            df.loc[valid_mask, "lat"] = lats
    except Exception as e:
        st.warning(f"UTM to lat/lon transform failed: {e}")
else:
    st.warning("UTM coordinate columns (utm_x, utm_y) not found in the data.")

# Helpers
@st.cache_data(show_spinner=False)
def load_image_bytes(src: str) -> bytes:
    p = Path(src)
    if p.exists():
        return p.read_bytes()
    try:
        import urllib.request
        with urllib.request.urlopen(src, timeout=10) as r:
            return r.read()
    except Exception:
        return b""

# Robust parsers
def parse_dt_col(s: pd.Series) -> pd.Series:
    """
    Tolerant parser for date_time for DATE USE:
    - strips non-digits
    - tries %Y%m%d%H%M%S then falls back to %Y%m%d
    - RETURNS NORMALISED (midnight) timestamps for stable date grouping/merges
    """
    ss = s.astype(str).str.replace(r"\D", "", regex=True)
    dt14 = pd.to_datetime(ss.str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    mask = dt14.isna()
    if mask.any():
        dt8 = pd.to_datetime(ss.str.slice(0, 8), format="%Y%m%d", errors="coerce")
        dt14[mask] = dt8[mask]
    return dt14.dt.normalize()

def parse_dt_full(s: pd.Series) -> pd.Series:
    """
    Parser for date_time for TIME-OF-DAY USE:
    - strips non-digits
    - tries %Y%m%d%H%M%S, falls back to %Y%m%d (those will be midnight)
    - PRESERVES HH:MM:SS where available
    """
    ss = s.astype(str).str.replace(r"\D", "", regex=True)
    dt = pd.to_datetime(ss.str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    missing = dt.isna()
    if missing.any():
        dt8 = pd.to_datetime(ss.str.slice(0, 8), format="%Y%m%d", errors="coerce")
        dt[missing] = dt8[missing]
    return dt

# Prepare datetime column for filtering
df_dt = df.copy()
if "date_time" in df_dt.columns:
    df_dt["dt"] = parse_dt_col(df_dt["date_time"])
else:
    df_dt["dt"] = pd.NaT

no_dates = df_dt["dt"].dropna().empty
if not no_dates:
    min_dt, max_dt = df_dt["dt"].min(), df_dt["dt"].max()
else:
    today = pd.Timestamp.utcnow().normalize()
    min_dt = max_dt = today

# FILTER BAR
if "filters_version" not in st.session_state:
    st.session_state["filters_version"] = 0
kv = st.session_state["filters_version"]
date_key = f"date_range_{kv}"
rec_key  = f"recorder_{kv}"

fb1, fb2, fb3 = st.columns([1.4, 1.0, 0.6])

default_range = (min_dt.date(), max_dt.date())

with fb1:
    date_sel = st.date_input(
        "Date range",
        value=default_range,
        min_value=default_range[0],
        max_value=default_range[1],
        key=date_key,
        disabled=no_dates,
    )

# First filter by date (unless no valid dates)
if no_dates:
    df_range = df_dt.copy()
else:
    if isinstance(date_sel, (tuple, list)):
        d_start, d_end = date_sel[0], date_sel[-1]
    else:
        d_start = d_end = date_sel
    range_mask = df_dt["dt"].dt.date.between(d_start, d_end)
    df_range = df_dt.loc[range_mask].copy()

# Recorder options depend on the date range
rec_options = ["All recorders"] + sorted(
    list(pd.Series(df_range.get("recorder_id", pd.Series(dtype=object))).dropna().unique())
)

with fb2:
    rec_choice = st.selectbox("Recorder", rec_options, index=0, key=rec_key)

with fb3:
    st.markdown("<div style='height:1.95em'></div>", unsafe_allow_html=True)
    if st.button("Clear filters", use_container_width=True):
        st.session_state["filters_version"] += 1
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

# Apply BOTH filters
df_page = df_range.copy()
if rec_choice != "All recorders":
    df_page = df_page[df_page["recorder_id"] == rec_choice]

# Headline metrics
total_recordings = len(df_page)
total_detections = int(df_page["is_present"].sum())
detection_rate = (total_detections / total_recordings * 100.0) if total_recordings else 0.0

m1, m2, m3, m4 = st.columns([1, 1, 1, 1.2])
m1.metric("Total detections", f"{total_detections:,}")
m2.metric("Total recordings", f"{total_recordings:,}")
m3.metric("Detection rate", f"{detection_rate:.1f}%")

GIBBON_IMAGE = os.environ.get("GIBBON_IMAGE", str(RESULTS_DIR / "assets" / "pileated_gibbon.jpg"))
img_bytes = load_image_bytes(GIBBON_IMAGE)
if img_bytes:
    m4.image(img_bytes, use_container_width=True)
else:
    with m4:
        st.caption("Add an image at RESULTS_DIR/assets/pileated_gibbon.jpg or set GIBBON_IMAGE.")

# Location Stats
if "recorder_id" not in df.columns:
    st.error("Column 'recorder_id' is required for location stats and charts.")
    st.stop()

present_stats_p = df_page.groupby("recorder_id", dropna=False).agg(
    present_count=("is_present", "sum")
).reset_index()

def first_valid(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if not s.empty else np.nan

agg_map = {}
for col in ["lat", "lon"]:
    if col in df_page.columns:
        agg_map[col] = first_valid
for col in ["station_name", "Elevation", "location_situation"]:
    if col in df_page.columns:
        agg_map[col] = "first"

if agg_map:
    location_df_p = df_page.groupby("recorder_id", dropna=False).agg(agg_map).reset_index()
else:
    location_df_p = present_stats_p.copy()
    location_df_p["lat"] = np.nan
    location_df_p["lon"] = np.nan
    location_df_p["station_name"] = ""
    location_df_p["Elevation"] = np.nan
    location_df_p["location_situation"] = ""

location_stats_p = pd.merge(location_df_p, present_stats_p, on="recorder_id", how="left")
location_stats_p["present_count"] = location_stats_p["present_count"].fillna(0).astype(int)

st.header("Location Stats")

_loc = location_stats_p.copy()
_totals = df_page.groupby("recorder_id", dropna=False).size().reset_index(name="total_recordings")
loc_aug = _loc.merge(_totals, on="recorder_id", how="left")
loc_aug["total_recordings"] = loc_aug["total_recordings"].fillna(0).astype(int)
loc_aug["detection_rate"] = np.where(
    loc_aug["total_recordings"] > 0,
    loc_aug["present_count"] / loc_aug["total_recordings"],
    np.nan
)

display_df = loc_aug.sort_values("present_count", ascending=False)

cols_order = [c for c in [
    "recorder_id", "station_name", "Elevation", "location_situation",
    "present_count", "total_recordings", "detection_rate", "lat", "lon"
] if c in display_df.columns]

pretty = (display_df[cols_order]
          .rename(columns={
              "recorder_id": "Recorder",
              "station_name": "Station Name",
              "Elevation": "Elevation",
              "location_situation": "Location Situation",
              "present_count": "Present",
              "total_recordings": "Total Recordings",
              "detection_rate": "Detection Rate (%)",
              "lat": "Lat",
              "lon": "Lon"
          }))

# Bold, centred headers; percentage formatting
try:
    styled = (pretty.style
              .format({"Detection Rate (%)": "{:.1%}"})
              .set_properties(**{"text-align": "center"})
              .set_table_styles([
                  {"selector": "th", "props": [("text-align", "center"), ("font-weight", "bold")]}
              ]))
    if hasattr(styled, "hide"):
        styled = styled.hide(axis="index")
    elif hasattr(styled, "hide_index"):
        styled = styled.hide_index()
    try:
        st.write(styled)
    except Exception:
        st.markdown(styled.to_html(), unsafe_allow_html=True)
except Exception:
    tmp = pretty.copy()
    if "Detection Rate (%)" in tmp.columns:
        tmp["Detection Rate (%)"] = (tmp["Detection Rate (%)"] * 100).round(1).astype(str) + "%"
    st.dataframe(tmp, use_container_width=True)

# Map
plot_df = location_stats_p.dropna(subset=["lat", "lon"])
if not plot_df.empty:
    plot_df = plot_df.copy()
    plot_df["radius"] = np.maximum(plot_df["present_count"] * 40, 40)
    plot_df = plot_df.sort_values("radius", ascending=False)

    layer_scatter = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position=["lon", "lat"],
        get_color="[255, 0, 0, 160]",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
        stroked=True,
        get_line_color=[0, 0, 0, 180],
        line_width_min_pixels=1,
        radius_min_pixels=2,
    )
    layer_text = pdk.Layer(
        "TextLayer",
        data=plot_df,
        get_position=["lon", "lat"],
        get_text="present_count",
        get_color="[0, 0, 0, 255]",
        sizeScale=5,
        get_size=16,
        get_alignment_baseline="'bottom'",
    )
    view_state = pdk.ViewState(
        latitude=float(plot_df["lat"].mean()),
        longitude=float(plot_df["lon"].mean()),
        zoom=9,
        pitch=0,
    )
    deck = pdk.Deck(
        layers=[layer_scatter, layer_text],
        initial_view_state=view_state,
        tooltip={"text": "Recorder: {recorder_id}\nPresences: {present_count}"}
    )
    st.pydeck_chart(deck, height=800)
else:
    st.info("Map not shown (no valid lat/lon in the selected filters).")

# Detections Over Time
st.header("Detections Over Time")
if "date_time" in df_page.columns and not df_page.empty and not df_dt["dt"].dropna().empty:
    dfc = df_page.copy()
    dfc["date"] = parse_dt_col(dfc["date_time"])
    unique_dates = pd.DataFrame({"date": pd.to_datetime(sorted(dfc["date"].dropna().unique()))})
    unique_recorders = pd.DataFrame({"recorder_id": dfc["recorder_id"].dropna().unique()})

    if unique_dates.empty or unique_recorders.empty:
        st.write("No data for the selected filters.")
    else:
        all_combinations = unique_dates.merge(unique_recorders, how="cross")

        counts = (
            dfc[dfc["is_present"] == 1]
            .groupby(["date", "recorder_id"])
            .size()
            .reset_index(name="present_count")
        )

        counts["date"] = pd.to_datetime(counts["date"]).dt.normalize()
        df_time = pd.merge(all_combinations, counts, on=["date", "recorder_id"], how="left")
        df_time["present_count"] = df_time["present_count"].fillna(0)

        date_chart = (
            alt.Chart(df_time)
            .mark_bar()
            .encode(
                x=alt.X("date:T", title="Date", axis=alt.Axis(format="%d-%m-%y")),
                y=alt.Y("present_count:Q", title="Number of Present Detections", axis=alt.Axis(format="d", tickMinStep=1)),
                color=alt.Color("recorder_id:N", title="Recorder ID"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%d-%m-%y"),
                    alt.Tooltip("recorder_id:N", title="Recorder"),
                    alt.Tooltip("present_count:Q", title="Presences", format="d"),
                ],
            )
            .interactive()
        )
        st.altair_chart(date_chart, use_container_width=True)
else:
    st.write("No data for the selected filters.")

# Detections by Time of Day
st.header("Detections by Time of Day")
if "date_time" in df_page.columns and not df_page.empty:
    dft = df_page.copy()
    dft["dt"] = parse_dt_full(dft["date_time"])  # preserve real times here
    dft["time_of_day"] = dft["dt"].dt.time
    tod_counts = (
        dft[dft["is_present"] == 1]
        .groupby(["time_of_day", "recorder_id"])
        .size()
        .reset_index(name="present_count")
    )
    tod_counts["tod_ts"] = pd.to_datetime(tod_counts["time_of_day"].astype(str), format="%H:%M:%S", errors="coerce")

    if tod_counts.empty:
        st.write("No data for the selected filters.")
    else:
        tod_chart = (
            alt.Chart(tod_counts.dropna(subset=["tod_ts"]))
            .mark_bar()
            .encode(
                x=alt.X("tod_ts:T", title="Time of Day", axis=alt.Axis(format="%H:%M")),
                y=alt.Y("present_count:Q", title="Number of Present Detections", axis=alt.Axis(format="d", tickMinStep=1)),
                color=alt.Color("recorder_id:N", title="Recorder ID"),
                tooltip=[
                    alt.Tooltip("recorder_id:N", title="Recorder"),
                    alt.Tooltip("tod_ts:T", title="Time", format="%H:%M"),
                    alt.Tooltip("present_count:Q", title="Presences", format="d"),
                ],
            )
            .interactive()
        )
        st.altair_chart(tod_chart, use_container_width=True)
else:
    st.write("No data for the selected filters.")

# Gibbon detection examples
st.header("Gibbon detection examples")

# Controls
c1, c2, c3, c4 = st.columns(4)
with c1:
    NUM_PER_PAGE = st.number_input("Clips per page", min_value=6, max_value=60, value=12, step=6)
with c2:
    COLS_PER_ROW = st.slider("Columns per row", min_value=2, max_value=5, value=3)
with c3:
    THUMB_SEC = st.slider("Spectrogram preview (seconds)", min_value=5, max_value=60, value=60, help="Thumbnail only; audio = full file")
with c4:
    PAGE = st.number_input("Page", min_value=1, value=1, step=1)

# Only files currently considered present
df_present = df_page[df_page["FinalLabelEffective"].astype(str).str.lower() == "present"].copy()

if df_present.empty:
    st.info("No present clips for the selected filters.")
else:
    total_clips = len(df_present)
    start = (int(PAGE)-1) * int(NUM_PER_PAGE)
    end = start + int(NUM_PER_PAGE)
    page_df = df_present.iloc[start:end].reset_index(drop=True)

    st.caption(f"Showing {len(page_df)} of {total_clips} present clips (page {PAGE})")

    def _figure_to_png(fig) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    @st.cache_data(show_spinner=False)
    def make_thumbnail_and_audio(audio_path: str, preview_seconds: int) -> Tuple[Optional[bytes], Optional[bytes]]:
        try:
            y, sr = librosa.load(audio_path, sr=None)
            y_thumb = y[: int(preview_seconds * sr)]
            fig, ax = plt.subplots(figsize=(2.6, 1.4), dpi=120)
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
        base = Path(st.session_state.get("audio_base_dir", str(RAW_AUDIO_DIR / "processed" / "present")))
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

    def _stem(val: str) -> str:
        return Path(str(val)).stem

    page_df = page_df.assign(_stem=page_df["filename"].astype(str).map(_stem))

    # Collect proposed overrides to UserLabel
    updates = []
    n_rows = math.ceil(len(page_df) / int(COLS_PER_ROW))
    for r in range(n_rows):
        cols = st.columns(int(COLS_PER_ROW))
        for c in range(int(COLS_PER_ROW)):
            i = r * int(COLS_PER_ROW) + c
            if i >= len(page_df):
                break
            row = page_df.iloc[i]
            with cols[c]:
                # header + change badge
                u = str(row.get("UserLabel", "")).strip().lower()
                m = str(row.get("FinalLabel", "")).strip().lower()
                badge = " 🟠 changed" if (u != "" and u != m) else ""
                st.caption(f"**{row['filename']}**{badge}")

                apath = find_audio_path_by_stem(row["_stem"])
                if apath and apath.exists():
                    thumb_png, audio_wav = make_thumbnail_and_audio(str(apath), int(THUMB_SEC))
                    if thumb_png:
                        st.image(thumb_png, use_container_width=True)
                    else:
                        st.info("No thumbnail")
                    if audio_wav:
                        st.audio(io.BytesIO(audio_wav), format="audio/wav")
                    else:
                        st.error("Audio unreadable")
                else:
                    st.error("Audio not found")

                # Radio defaults to the current effective label
                eff = str(row["FinalLabelEffective"]).lower()
                idx = 1 if eff == "present" else 0
                new_choice = st.radio("Label", ["absent", "present"], index=idx, key=f"val_{start+i}", horizontal=True)

                # If user choice equals model FinalLabel -> clear override (empty)
                # Else store the chosen label in UserLabel
                proposed = "" if new_choice == m else new_choice
                if proposed != u:
                    updates.append({"filename": row["filename"], "UserLabel": proposed})

    # Save overrides to the active file (CSV/XLSX)
    if st.button("Save All Updates"):
        try:
            df_master = load_filename_level()  # reload fresh (ensures UserLabel present)
            changes = 0
            for rec in updates:
                mask = df_master["filename"].astype(str) == str(rec["filename"])
                if mask.any():
                    old = str(df_master.loc[mask, "UserLabel"].iloc[0]) if "UserLabel" in df_master.columns else ""
                    if old != rec["UserLabel"]:
                        if "UserLabel" not in df_master.columns:
                            df_master["UserLabel"] = ""
                        df_master.loc[mask, "UserLabel"] = rec["UserLabel"]
                        changes += int(mask.sum())
            out_path = Path(st.session_state.get("filename_level_path", RESULTS_DIR / "filename_level.csv"))
            save_filename_level(out_path, df_master)
            st.success(f"Saved {changes} update(s) to {out_path.name}.")
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to save updates: {e}")

# Full Data Table
st.header("Full Data Table")
if st.checkbox("Show full filename-level data"):
    st.dataframe(df_page)
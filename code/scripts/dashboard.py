from __future__ import annotations

import os
import io
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import pydeck as pdk
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
from pyproj import Transformer  

try:
    st.set_page_config(layout="wide", page_title="Dashboard")
except Exception:
    pass

os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s or "unnamed"


def _make_export_filename(project_root: Path, user: str | None = None) -> str:
    project_slug = _slugify(project_root.name)
    user_slug = _slugify(user or "reviewer")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    return f"pamalytics_{project_slug}_{user_slug}_{ts}_validated.csv"


def _apply_canonical_overrides(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    num = lambda s: pd.to_numeric(s, errors="coerce")

    if "file_id" in df.columns and "source_file" not in df.columns:
        df["source_file"] = df["file_id"].astype(str)
    if "file_path" in df.columns and "path" not in df.columns:
        df["path"] = df["file_path"].astype(str)

    if "start_s" not in df.columns and "detection_start_s" in df.columns:
        df["start_s"] = num(df["detection_start_s"])
    if "end_s" not in df.columns and "detection_end_s" in df.columns:
        df["end_s"] = num(df["detection_end_s"])

    if "presence_label" in df.columns:
        if "FinalLabel" not in df.columns:
            df["FinalLabel"] = df["presence_label"].astype(str)
        elif "label" not in df.columns:
            df["label"] = df["presence_label"].astype(str)

    if "species_name" in df.columns and "class" not in df.columns:
        df["class"] = df["species_name"].astype(str)

    if "detection_probability" in df.columns:
        if "class_prob" not in df.columns:
            df["class_prob"] = num(df["detection_probability"])
        elif "probability" not in df.columns and "score" not in df.columns:
            df["probability"] = num(df["detection_probability"])

    return df


def ensure_userlabel(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if "UserLabel" not in df.columns:
        df["UserLabel"] = ""
    else:
        df["UserLabel"] = df["UserLabel"].fillna("").replace({"nan": ""})
    return df


def with_effective_labels(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    if "FinalLabel" not in df.columns:
        if "label" in df.columns:
            df["FinalLabel"] = df["label"].astype(str)
        elif "presence_label" in df.columns:
            df["FinalLabel"] = df["presence_label"].astype(str)
        else:
            df["FinalLabel"] = "absent"

    df = ensure_userlabel(df)

    u = df["UserLabel"].fillna("").astype(str).replace({"nan": ""}).str.strip().str.lower()
    m = df["FinalLabel"].astype(str).str.strip().str.lower()
    eff = np.where(u != "", u, m)
    df["FinalLabelEffective"] = eff
    df["is_present"] = (eff == "present").astype(int)
    df["Changed"] = (u != "") & (u != m)
    return df


def parse_dt_col(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.replace(r"\D", "", regex=True)
    dt14 = pd.to_datetime(ss.str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    mask = dt14.isna()
    if mask.any():
        dt8 = pd.to_datetime(ss.str.slice(0, 8), format="%Y%m%d", errors="coerce")
        dt14[mask] = dt8[mask]
    return dt14.dt.normalize()


def parse_dt_full(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.replace(r"\D", "", regex=True)
    dt = pd.to_datetime(ss.str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    missing = dt.isna()
    if missing.any():
        dt8 = pd.to_datetime(ss.str.slice(0, 8), format="%Y%m%d", errors="coerce")
        dt[missing] = dt8[missing]
    return dt


def _ensure_latlon(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "lat" not in out.columns:
        out["lat"] = np.nan
    if "lon" not in out.columns:
        out["lon"] = np.nan

    has_vals = (pd.to_numeric(out["lat"], errors="coerce").notna().any() and
                pd.to_numeric(out["lon"], errors="coerce").notna().any())
    if has_vals:
        return out

    if "utm_x" in out.columns and "utm_y" in out.columns:
        try:
            out["utm_x"] = pd.to_numeric(out["utm_x"], errors="coerce")
            out["utm_y"] = pd.to_numeric(out["utm_y"], errors="coerce")
            valid = out["utm_x"].notna() & out["utm_y"].notna()
            if valid.any():
                transformer = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)
                xs = np.array(out.loc[valid, "utm_x"], dtype=float)
                ys = np.array(out.loc[valid, "utm_y"], dtype=float)
                lons, lats = transformer.transform(xs, ys)
                out.loc[valid, "lon"] = np.asarray(lons, dtype=float)
                out.loc[valid, "lat"] = np.asarray(lats, dtype=float)
        except Exception:
            return out
    return out


def _build_latlon_lookup(df_all: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    res: Dict[str, pd.DataFrame] = {}
    if df_all is None or df_all.empty:
        return res
    df_ll = _ensure_latlon(df_all)
    if {"basename", "lat", "lon"} <= set(df_ll.columns):
        by_base = (df_ll.dropna(subset=["lat", "lon"])
                        .groupby("basename", dropna=False)[["lat", "lon"]]
                        .mean()
                        .reset_index())
        if not by_base.empty:
            res["by_basename"] = by_base
    if {"recorder_id", "lat", "lon"} <= set(df_ll.columns):
        by_rec = (df_ll.dropna(subset=["lat", "lon"])
                       .groupby("recorder_id", dropna=False)[["lat", "lon"]]
                       .mean()
                       .reset_index())
        if not by_rec.empty:
            res["by_recorder"] = by_rec
    return res


def _attach_latlon_from_glob(df_page: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
    if df_page is None or df_page.empty:
        return df_page
    out = df_page.copy()
    if "lat" not in out.columns:
        out["lat"] = np.nan
    if "lon" not in out.columns:
        out["lon"] = np.nan

    need = out[["lat", "lon"]].dropna().empty
    if not need:
        return out

    lk = _build_latlon_lookup(df_all)
    if "by_basename" in lk and "basename" in out.columns:
        out = out.merge(lk["by_basename"], on="basename", how="left", suffixes=("", "_lk"))
        if "lat_lk" in out.columns and "lon_lk" in out.columns:
            out["lat"] = out["lat"].fillna(out["lat_lk"])
            out["lon"] = out["lon"].fillna(out["lon_lk"])
            out = out.drop(columns=[c for c in ["lat_lk", "lon_lk"] if c in out.columns])

    if out[["lat", "lon"]].dropna().empty and "by_recorder" in lk and "recorder_id" in out.columns:
        out = out.merge(lk["by_recorder"], on="recorder_id", how="left", suffixes=("", "_rk"))
        if "lat_rk" in out.columns and "lon_rk" in out.columns:
            out["lat"] = out["lat"].fillna(out["lat_rk"])
            out["lon"] = out["lon"].fillna(out["lon_rk"])
            out = out.drop(columns=[c for c in ["lat_rk", "lon_rk"] if c in out.columns])

    return out


def _extract_prob(row: pd.Series) -> float:
    for key in ("detection_probability", "class_prob", "probability", "score"):
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                continue
    return float("nan")


def _collect_boxes_and_probs(gdf: pd.DataFrame):
    mids, lows, highs, ps = [], [], [], []
    for _, row in gdf.iterrows():
        try:
            sx = float(row.get("start_s", np.nan))
            ex = float(row.get("end_s",   np.nan))
            if not (np.isfinite(sx) and np.isfinite(ex) and ex > sx):
                continue
            lf = float(row.get("low_freq",  np.nan))
            hf = float(row.get("high_freq", np.nan))
            p  = _extract_prob(row)
            mids.append(0.5 * (sx + ex))
            lows.append(lf)
            highs.append(hf)
            ps.append(float(np.clip(p, 0.0, 1.0)))
        except Exception:
            continue
    if not mids:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (np.asarray(mids, dtype=float),
            np.asarray(lows, dtype=float),
            np.asarray(highs, dtype=float),
            np.asarray(ps, dtype=float))


def _draw_prob_labels_inline(ax, gdf: pd.DataFrame, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    mids, lows, highs, ps = _collect_boxes_and_probs(gdf)
    if mids.size == 0:
        return
    keep = (mids >= xmin) & (mids <= xmax)
    mids, lows, highs, ps = mids[keep], lows[keep], highs[keep], ps[keep]
    if mids.size == 0:
        return
    order = np.argsort(-ps)
    if order.size > 10:
        order = order[:10]
    mids, lows, highs, ps = mids[order], lows[order], highs[order], ps[order]

    vspan = max(1.0, ymax - ymin)
    vpad = 0.02 * vspan
    fixed_high = ymin + 0.90 * vspan

    for i, (x, lf, hf, p) in enumerate(zip(mids, lows, highs, ps)):
        label = f"{p:.2f}"
        has_lf = np.isfinite(lf)
        has_hf = np.isfinite(hf)
        if has_lf and has_hf:
            y_raw = (hf + vpad) if (i % 2 == 0) else (lf - vpad)
            va = "bottom" if (i % 2 == 0) else "top"
        else:
            y_raw = fixed_high
            va = "center"
        y_clamped = float(np.clip(y_raw, ymin + vpad, ymax - vpad))
        ax.text(
            x,
            y_clamped,
            label,
            ha="center",
            va=va,
            fontsize=9,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc=(0, 0, 0, 0.55),
                ec=(1, 1, 1, 0.25),
                lw=0.5,
            ),
        )


def _num(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _estimate_low_edge_hz_for_group(gdf: pd.DataFrame) -> Optional[float]:
    vals: List[float] = []
    for _, row in gdf.iterrows():
        lf = _num(row.get("low_freq"))
        hf = _num(row.get("high_freq"))
        if np.isfinite(lf) and np.isfinite(hf) and hf > lf:
            vals.append(lf)
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return float(np.nanmedian(arr))


def _choose_te_for_group(low_edge_hz: Optional[float], sr: int) -> int:
    if not isinstance(sr, (int, float)) or not np.isfinite(sr):
        return 1
    if sr < 96_000:
        return 1
    if not (isinstance(low_edge_hz, (int, float)) and np.isfinite(low_edge_hz)):
        return 1
    if low_edge_hz <= 20_000:
        return 1
    return 10


def _apply_time_expansion_for_playback(y: np.ndarray, sr: int, te: int) -> Tuple[np.ndarray, int]:
    te = max(1, int(te))
    y_out = y.astype(np.float32, copy=False)
    if y_out.size == 0:
        return y_out, int(sr)

    peak = float(np.max(np.abs(y_out)))
    if peak > 0:
        y_out = (y_out / peak * 0.98).astype(np.float32, copy=False)

    if te == 1:
        return y_out, int(sr)

    psr = max(1, int(sr // te))
    return y_out, psr


def show_detection_examples(df_page: pd.DataFrame, df_all: pd.DataFrame):
    st.header("Detection examples")

    c1, c2, _, c4 = st.columns(4)
    with c1:
        NUM_PER_PAGE = st.number_input("Spectrograms per page", min_value=4, max_value=60, value=12, step=4)
    with c2:
        COLS_PER_ROW = st.slider("Columns per row", min_value=2, max_value=5, value=3)
    with c4:
        PAGE = st.number_input("Page", min_value=1, value=1, step=1)

    df_det = df_page[df_page["FinalLabelEffective"].astype(str).str.lower() == "present"].copy()
    if "basename" not in df_det.columns:
        df_det["basename"] = df_det.get("source_file", "").astype(str).map(lambda p: Path(p).name)
    if "filename_stem" not in df_det.columns:
        df_det["filename_stem"] = df_det["basename"].astype(str).map(lambda s: Path(s).stem.lower())
    if "class" not in df_det.columns and "species_name" in df_det.columns:
        df_det["class"] = df_det["species_name"]
    disp_species = df_det.get("class", "").astype(str).str.strip()
    norm_species = disp_species.str.lower().replace({"": np.nan, "nan": np.nan})
    df_det["species_display"] = disp_species.where(norm_species.notna(), "[absent]")

    if "start_s" not in df_det.columns and "detection_start_s" in df_det.columns:
        df_det["start_s"] = pd.to_numeric(df_det["detection_start_s"], errors="coerce")
    if "end_s" not in df_det.columns and "detection_end_s" in df_det.columns:
        df_det["end_s"] = pd.to_numeric(df_det["detection_end_s"], errors="coerce")

    if df_det.empty:
        st.info("No present detections with audio available for the selected filters.")
        return

    df_det = df_det.sort_values(["basename", "species_display", "start_s"])
    grouped = df_det.groupby(["basename", "species_display"], dropna=False)
    keys: List[Tuple[str, str]] = list(grouped.indices.keys())

    per_group_max: Dict[Tuple[str, str], float] = {}
    try:
        tmp = df_det.assign(_p=df_det.apply(_extract_prob, axis=1))
        per_group_max = tmp.groupby(["basename", "species_display"])["_p"].max(numeric_only=True).to_dict()
    except Exception:
        pass

    def _sort_key(k: Tuple[str, str]):
        mp = per_group_max.get(k, -1)
        mpv = mp if (isinstance(mp, (int, float)) and np.isfinite(mp)) else -1
        return (-mpv, k[0], k[1])

    keys = sorted(keys, key=_sort_key)

    total_cards = len(keys)
    start_idx = (int(PAGE) - 1) * int(NUM_PER_PAGE)
    end_idx = min(total_cards, start_idx + int(NUM_PER_PAGE))
    page_keys = keys[start_idx:end_idx]
    st.caption(f"Showing {len(page_keys)} of {total_cards} Spectrograms (page {PAGE})")

    def _resolve_audio_path(row_or_df) -> Optional[Path]:
        if isinstance(row_or_df, pd.Series):
            rows = [row_or_df]
        else:
            rows = [row_or_df.iloc[0]] if len(row_or_df) else []
        for r_ in rows:
            for col_ in ("path", "file_path"):
                p_ = r_.get(col_)
                if isinstance(p_, str) and p_.strip() and Path(p_).exists():
                    return Path(p_)
        def _by_stem(df_present: pd.DataFrame, stem: str) -> Optional[Path]:
            for col_ in ("path", "file_path"):
                if col_ not in df_present.columns:
                    continue
                rows2 = df_present[df_present["filename_stem"] == stem]
                if rows2.empty:
                    continue
                for p_ in rows2[col_]:
                    if isinstance(p_, str) and p_.strip() and Path(p_).exists():
                        return Path(p_)
                q = rows2[col_].dropna().astype(str).head(1)
                if not q.empty and Path(q.iloc[0]).exists():
                    return Path(q.iloc[0])
            return None
        if isinstance(row_or_df, pd.Series):
            stem = Path(str(row_or_df.get("basename", row_or_df.get("source_file", "")))).stem.lower()
        else:
            s = row_or_df.iloc[0]
            stem = Path(str(s.get("basename", s.get("source_file", "")))).stem.lower()
        return _by_stem(df_all, stem)

    n_rows = math.ceil(len(page_keys) / int(COLS_PER_ROW))
    for r in range(n_rows):
        cols = st.columns(int(COLS_PER_ROW))
        for c in range(int(COLS_PER_ROW)):
            gi = r * int(COLS_PER_ROW) + c
            if gi >= len(page_keys):
                break

            base, species_line = page_keys[gi]
            gdf = grouped.get_group((base, species_line)).copy()

            n_det = int(len(gdf))
            try:
                ps = gdf.apply(_extract_prob, axis=1).to_numpy()
                ps = ps[np.isfinite(ps)]
                max_cp = float(np.max(ps)) if ps.size else None
            except Exception:
                max_cp = None

            apath = _resolve_audio_path(gdf)

            with cols[c]:
                title_html = (
                    f"<div style='margin-bottom:2px'><strong>{base}</strong>"
                    f"<br>{species_line}"
                    f"<br>Detections: {n_det}"
                )
                if max_cp is not None and np.isfinite(max_cp):
                    title_html += f"<br>Max probability: {max_cp:.2f}"
                title_html += "</div>"
                st.markdown(title_html, unsafe_allow_html=True)

                if not (apath and apath.exists()):
                    st.error("Audio not found")
                    continue

                try:
                    y, sr = librosa.load(str(apath), sr=None, mono=True)
                except Exception as e:
                    st.error(f"Audio read error: {e}")
                    continue

                boxes: List[Dict[str, float]] = []
                for _, row in gdf.iterrows():
                    b = {
                        "start_s": _num(row.get("start_s", row.get("detection_start_s"))),
                        "end_s": _num(row.get("end_s", row.get("detection_end_s"))),
                        "low_freq": _num(row.get("low_freq")),
                        "high_freq": _num(row.get("high_freq")),
                        "prob": _num(_extract_prob(row)),
                    }
                    if (np.isfinite(b["start_s"]) and np.isfinite(b["end_s"]) and b["end_s"] > b["start_s"]):
                        boxes.append(b)
                if boxes:
                    boxes = sorted(
                        boxes,
                        key=lambda b: (b["prob"] if np.isfinite(b["prob"]) else -1.0),
                        reverse=True,
                    )[:10]

                highs = [b["high_freq"] for b in boxes if np.isfinite(b["high_freq"])]
                lows = [b["low_freq"] for b in boxes if np.isfinite(b["low_freq"])]
                if highs and lows and max(highs) > min(lows):
                    fmin, fmax = min(lows), max(highs)
                else:
                    fmin, fmax = 0.0, 0.5 * sr
                span = max(1.0, (fmax - fmin))
                pad = max(4_000.0, 0.30 * span)
                nyq = 0.5 * sr * 0.98
                ymin = max(0.0, fmin - pad)
                ymax = min(nyq, fmax + pad)

                try:
                    n_fft = 4096 if sr <= 48_000 else 8192
                    hop = n_fft // 8
                    D = librosa.stft(y=y, n_fft=n_fft, hop_length=hop)
                    S = np.abs(D) ** 2
                    S_dB = librosa.power_to_db(S, ref=np.max, top_db=90)
                    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
                    freqs_hz = np.linspace(0.0, sr * 0.5, S.shape[0])
                    dur = max(1e-6, len(y) / sr)
                    tpad = dur * 0.01
                    xmin, xmax = 0 - tpad, dur + tpad
                except Exception as e:
                    st.error(f"Spectrogram setup error: {e}")
                    continue

                try:
                    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=220, constrained_layout=False)
                    extent = [times.min(), times.max(), freqs_hz.min(), freqs_hz.max()]
                    ax.imshow(
                        S_dB,
                        origin="lower",
                        aspect="auto",
                        interpolation="nearest",
                        extent=extent,
                        vmin=S_dB.max() - 90,
                        vmax=S_dB.max(),
                    )
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Frequency (kHz)")
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda ytick, pos: f"{ytick / 1000:.0f}"))
                    for b in boxes:
                        x0, x1 = b["start_s"], b["end_s"]
                        ax.add_patch(
                            Rectangle(
                                (x0, ymin),
                                x1 - x0,
                                ymax - ymin,
                                facecolor=(1, 1, 1, 0.06),
                                edgecolor=(1, 1, 1, 0.12),
                                lw=0.6,
                            )
                        )
                    _draw_prob_labels_inline(ax, gdf, xmin, xmax, ymin, ymax)
                    st.pyplot(fig, use_container_width=True, clear_figure=True)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Spectrogram error: {e}")

                # Playback – full clip, TE applied if needed
                try:
                    y_seg = y

                    low_edge = _estimate_low_edge_hz_for_group(gdf)
                    te = _choose_te_for_group(low_edge, sr)
                    y_play, psr = _apply_time_expansion_for_playback(y_seg, sr, te)

                    abuf = io.BytesIO()
                    sf.write(abuf, y_play, psr, format="WAV")
                    abuf.seek(0)
                    st.audio(abuf, format="audio/wav")
                except Exception as e:
                    st.error(f"Playback error: {e}")

    st.caption("")


def _augment_grouping_fields_class(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if "basename" not in df.columns:
        if "path" in df.columns and df["path"].notna().any():
            df["basename"] = df["path"].astype(str).apply(lambda p: Path(p).name if p else "")
        else:
            df["basename"] = df.get("source_file", "").astype(str).apply(lambda p: Path(p).name if p else "")
    df["filename_stem"] = df["basename"].astype(str).apply(lambda n: Path(n).stem.lower())

    if "FinalLabel" not in df.columns and "label" in df.columns:
        df["FinalLabel"] = df["label"].astype(str)
    df = with_effective_labels(df)

    if "class" not in df.columns and "species_name" in df.columns:
        df["class"] = df["species_name"].astype(str)
    if "class" not in df.columns:
        df["class"] = "Unknown"

    if "lat" not in df.columns:
        df["lat"] = np.nan
    if "lon" not in df.columns:
        df["lon"] = np.nan

    return df


def _load_csv_safe(p: Path) -> Optional[pd.DataFrame]:
    try:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            try:
                df.columns = df.columns.str.strip()
            except Exception:
                pass
            return df
    except Exception:
        return None
    return None


def _dataset_choice(sources: Dict[str, str]) -> Tuple[pd.DataFrame, str, Dict[str, pd.DataFrame], Dict[str, Path]]:
    proj_root = Path(sources.get("project") or sources.get("project_root") or ".")
    data_dir = proj_root / "data_normalised"
    data_dir.mkdir(parents=True, exist_ok=True)

    p_original = data_dir / "detections_normalised.csv"
    p_valid_pub = data_dir / "detections_validated.csv"

    choices: Dict[str, pd.DataFrame] = {}
    path_map: Dict[str, Path] = {}

    df_orig = _load_csv_safe(p_original)
    if df_orig is not None:
        choices["Original"] = df_orig
        path_map["Original"] = p_original

    df_val_pub = _load_csv_safe(p_valid_pub)
    if df_val_pub is not None:
        choices["Validated (published)"] = df_val_pub
        path_map["Validated (published)"] = p_valid_pub

    if not choices:
        return pd.DataFrame(), "None", {}, {}

    default_label = "Validated (published)" if "Validated (published)" in choices else "Original"

    active = st.session_state.get("active_dataset_label")
    if isinstance(active, str) and active in choices:
        default_label = active

    return choices[default_label].copy(), default_label, choices, path_map


def _has_data(df: pd.DataFrame, col: str) -> bool:
    if col not in df.columns:
        return False
    s = df[col].astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return s.notna().any()


def render_dashboard(df: Optional[pd.DataFrame], sources: Dict[str, str], page: str = "Dashboard", use_internal_nav: Optional[bool] = None):
    st.title("Detection Dashboard")

    running_in_studio = df is not None and isinstance(df, pd.DataFrame)
    if use_internal_nav is None:
        use_internal_nav = not running_in_studio

    if use_internal_nav:
        with st.sidebar:
            st.header("Pages")
            pages = ["Dashboard"]
            _ = st.radio("Navigate", pages, index=0, key="dashboard_nav_radio")

    df_default, ds_label, ds_choices, ds_paths = _dataset_choice(sources)
    if ds_label == "None" or df_default.empty:
        st.error("No dataset found in this project. Ingest data first.")
        return

    ds_labels = list(ds_choices.keys())
    ds_index = ds_labels.index(ds_label) if ds_label in ds_labels else 0

    df_dt_probe = df_default.copy()
    if "date_time" not in df_dt_probe.columns and "recording_dt" in df_dt_probe.columns:
        df_dt_probe["date_time"] = df_dt_probe["recording_dt"].astype(str)
    if "date_time" in df_dt_probe.columns:
        df_dt_probe["dt"] = parse_dt_col(df_dt_probe["date_time"])
    else:
        df_dt_probe["dt"] = pd.NaT
    no_dates = df_dt_probe["dt"].dropna().empty
    if not no_dates:
        min_dt, max_dt = df_dt_probe["dt"].min(), df_dt_probe["dt"].max()
    else:
        today = pd.Timestamp.utcnow().normalize()
        min_dt = max_dt = today

    group_candidates: List[str] = []
    if _has_data(df_default, "species_name"):
        group_candidates.append("species_name")
    if _has_data(df_default, "recorder_id"):
        group_candidates.append("recorder_id")
    if not group_candidates:
        group_candidates = ["species_name"]
    default_group = (
        "species_name"
        if (
            "species_name" in group_candidates
            and df_default["species_name"].astype(str).replace({"": np.nan, "nan": np.nan}).nunique() > 1
        )
        else ("recorder_id" if "recorder_id" in group_candidates else group_candidates[0])
    )

    c0, c1, c2, c3 = st.columns([1.3, 1.2, 1.0, 0.7])
    with c0:
        dataset_label = st.selectbox("Dataset", ds_labels, index=ds_index, key="dataset_selector")
    with c1:
        default_range = (min_dt.date(), max_dt.date())
        date_sel = st.date_input(
            "Date range",
            value=default_range,
            min_value=default_range[0],
            max_value=default_range[1],
            disabled=no_dates,
            key=f"date_range_{dataset_label}",
        )
    with c2:
        group_key = st.selectbox(
            "Group by",
            options=group_candidates,
            index=group_candidates.index(default_group),
            key=f"group_key_{dataset_label}",
        )
    with c3:
        st.markdown("<div style='height:1.95em'></div>", unsafe_allow_html=True)
        if st.button("Clear filters", use_container_width=True):
            for k in list(st.session_state.keys()):
                if str(k).startswith("date_range_") or str(k).startswith("group_key_"):
                    st.session_state.pop(k, None)
            if hasattr(st, "rerun"):
                st.rerun()

    if dataset_label != ds_label:
        df_default = ds_choices[dataset_label].copy()
        st.session_state["active_dataset_label"] = dataset_label
        st.session_state["active_dataset_path"] = str(ds_paths.get(dataset_label, ""))
        st.session_state["pa_df_det"] = df_default.copy()
    st.session_state.setdefault("active_dataset_label", dataset_label)
    st.session_state.setdefault("active_dataset_path", str(ds_paths.get(dataset_label, "")))
    st.session_state["pa_df_det"] = df_default.copy()

    df_raw = _apply_canonical_overrides(df_default)
    df_all = _augment_grouping_fields_class(df_raw)
    df_all = _ensure_latlon(df_all)

    df_dt = df_all.copy()
    if "date_time" not in df_dt.columns and "recording_dt" in df_dt.columns:
        df_dt["date_time"] = df_dt["recording_dt"].astype(str)
    if "date_time" in df_dt.columns:
        df_dt["dt"] = parse_dt_col(df_dt["date_time"])
    else:
        df_dt["dt"] = pd.NaT

    if not no_dates:
        if isinstance(date_sel, (tuple, list)):
            d_start, d_end = date_sel[0], date_sel[-1]
        else:
            d_start = d_end = date_sel
        mask = df_dt["dt"].dt.date.between(d_start, d_end)
        df_page = df_dt.loc[mask].copy()
    else:
        df_page = df_dt.copy()

    if "class" not in df_page.columns and "species_name" in df_page.columns:
        df_page["class"] = df_page["species_name"].astype(str)

    total_dets = int(len(df_page))
    present_dets = int((df_page["FinalLabelEffective"].astype(str).str.lower() == "present").sum())
    det_rate_pct = (100.0 * present_dets / total_dets) if total_dets else 0.0

    m1, m2, m3 = st.columns([1, 1, 1])
    m1.metric("Present detections", f"{present_dets:,}")
    m2.metric("Total detections", f"{total_dets:,}")
    m3.metric("Detection rate", f"{det_rate_pct:.1f}%")

    grp = (
        df_page.assign(_present=(df_page["FinalLabelEffective"].str.lower() == "present"))
        .groupby(group_key, dropna=False)["_present"]
        .agg(present_detections="sum", total_detections="count")
        .reset_index()
    )
    grp["detection_rate"] = grp["present_detections"] / grp["total_detections"]
    grp = grp.sort_values("present_detections", ascending=False)

    pretty = grp.rename(
        columns={
            group_key: ("Species" if group_key == "species_name" else "Recorder"),
            "present_detections": "Present Detections",
            "total_detections": "Total Detections",
            "detection_rate": "Detection Rate (%)",
        }
    )
    try:
        styled = pretty.style.format({"Detection Rate (%)": "{:.1%}"}).set_properties(**{"text-align": "center"})
        if hasattr(styled, "hide_index"):
            styled = styled.hide_index()
        st.write(styled)
    except Exception:
        tmp = pretty.copy()
        if "Detection Rate (%)" in tmp.columns:
            tmp["Detection Rate (%)"] = (tmp["Detection Rate (%)"] * 100).round(1).astype(str) + "%"
        st.dataframe(tmp, use_container_width=True)

    df_page = _ensure_latlon(df_page)
    need_latlon = df_page[["lat", "lon"]].dropna().empty
    if need_latlon:
        df_page = _attach_latlon_from_glob(df_page, df_all)

    present_by_group = (
        df_page.assign(_present=(df_page["FinalLabelEffective"].str.lower() == "present"))
        .groupby([group_key, "basename"], dropna=False)["_present"]
        .max()
        .reset_index()
        .groupby(group_key, dropna=False)["_present"]
        .sum()
        .reset_index(name="present_files")
    )

    if "lat" in df_page.columns and "lon" in df_page.columns:
        latlon_source = df_page.dropna(subset=["lat", "lon"])
        if not latlon_source.empty:
            latlon_by_group = (
                latlon_source.groupby(group_key, dropna=False)[["lat", "lon"]]
                .mean()
                .reset_index()
            )
        else:
            latlon_by_group = pd.DataFrame(columns=[group_key, "lat", "lon"])
    else:
        latlon_by_group = pd.DataFrame(columns=[group_key, "lat", "lon"])

    present_by_group[group_key] = present_by_group[group_key].astype(str)
    latlon_by_group[group_key] = latlon_by_group[group_key].astype(str)
    location_stats_p = present_by_group.merge(latlon_by_group, on=group_key, how="left")

    plot_df = location_stats_p.dropna(subset=["lat", "lon"])
    if not plot_df.empty:
        plot_df = plot_df.copy()
        plot_df["radius"] = np.maximum(plot_df["present_files"] * 40, 40)
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
            get_text="present_files",
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
            tooltip={
                "text": f"{'Species' if group_key=='species_name' else 'Recorder'}: {{{group_key}}}\nPresent files: {{present_files}}"
            },
        )
        st.pydeck_chart(deck, height=800)

    # Detections over time
    if "date_time" in df_page.columns and not df_page.empty and not df_dt["dt"].dropna().empty:
        dfc = df_page.copy()
        dfc["date"] = parse_dt_col(dfc["date_time"])

        unique_dates = pd.DataFrame({"date": pd.to_datetime(sorted(dfc["date"].dropna().unique()))})
        unique_group = pd.DataFrame({group_key: dfc[group_key].dropna().unique()})

        if not unique_dates.empty and not unique_group.empty:
            all_combinations = unique_dates.merge(unique_group, how="cross")

            counts = (
                dfc.assign(_present=(dfc["FinalLabelEffective"].str.lower() == "present"))
                .groupby(["date", group_key, "basename"], dropna=False)["_present"]
                .max()
                .reset_index()
                .groupby(["date", group_key], dropna=False)["_present"]
                .sum()
                .reset_index(name="present_files")
            )

            df_time = all_combinations.merge(counts, on=["date", group_key], how="left").fillna({"present_files": 0})

            # Only show the chart if there is at least one non-zero value
            if (df_time["present_files"] > 0).any():
                st.header(f"Detections Over Time (by {('Species' if group_key=='species_name' else 'Recorder')})")
                date_chart = (
                    alt.Chart(df_time)
                    .mark_bar()
                    .encode(
                        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%d-%m-%y")),
                        y=alt.Y("present_files:Q", title="Present Files", axis=alt.Axis(format="d", tickMinStep=1)),
                        color=alt.Color(f"{group_key}:N", title=("Species" if group_key == "species_name" else "Recorder")),
                        tooltip=[
                            alt.Tooltip("date:T", title="Date", format="%d-%m-%y"),
                            alt.Tooltip(f"{group_key}:N", title=("Species" if group_key == "species_name" else "Recorder")),
                            alt.Tooltip("present_files:Q", title="Present Files", format="d"),
                        ],
                    )
                    .interactive()
                )
                st.altair_chart(date_chart, use_container_width=True)

    # Detections by time of day
    if "date_time" in df_page.columns and not df_page.empty:
        dft = df_page.copy()
        dft["dt"] = parse_dt_full(dft["date_time"])
        dft["time_of_day"] = dft["dt"].dt.time

        tod = (
            dft.assign(_present=(dft["FinalLabelEffective"].str.lower() == "present"))
            .groupby([group_key, "basename", "time_of_day"], dropna=False)["_present"]
            .max()
            .reset_index()
            .groupby([group_key, "time_of_day"], dropna=False)["_present"]
            .sum()
            .reset_index(name="present_files")
        )
        tod["tod_ts"] = pd.to_datetime(tod["time_of_day"].astype(str), format="%H:%M:%S", errors="coerce")

        if not tod.empty:
            tod_nonzero = tod.dropna(subset=["tod_ts"])
            tod_nonzero = tod_nonzero[tod_nonzero["present_files"] > 0]

            if not tod_nonzero.empty:
                st.header(f"Detections by Time of Day (by {('Species' if group_key=='species_name' else 'Recorder')})")
                tod_chart = (
                    alt.Chart(tod_nonzero)
                    .mark_bar()
                    .encode(
                        x=alt.X("tod_ts:T", title="Time of Day", axis=alt.Axis(format="%H:%M")),
                        y=alt.Y("present_files:Q", title="Present Files", axis=alt.Axis(format="d", tickMinStep=1)),
                        color=alt.Color(f"{group_key}:N", title=("Species" if group_key == "species_name" else "Recorder")),
                        tooltip=[
                            alt.Tooltip(f"{group_key}:N", title=("Species" if group_key == "species_name" else "Recorder")),
                            alt.Tooltip("tod_ts:T", title="Time", format="%H:%M"),
                            alt.Tooltip("present_files:Q", title="Present Files", format="d"),
                        ],
                    )
                    .interactive()
                )
                st.altair_chart(tod_chart, use_container_width=True)

    show_detection_examples(df_page, df_all)

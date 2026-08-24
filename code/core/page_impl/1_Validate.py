from __future__ import annotations

import math
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle

# Page config
try:
    st.set_page_config(layout="wide", page_title="Validate")
except Exception:
    pass


st.markdown(
    """
    <style>
    div[data-testid="stFormSubmitButton"] button {
        background-color: #374151 !important;
        border-color: #374151 !important;
        color: white !important;
        font-weight: 700 !important;
        min-height: 3.0rem !important;
        border-radius: 0.65rem !important;
        margin-top: 0.35rem !important;
        margin-bottom: 0.15rem !important;
    }
    .pam-card-header {
        min-height: 6.15rem;
        height: 6.15rem;
        margin-bottom: 0.35rem;
        overflow: hidden;
    }
    .pam-card-title {
        line-height: 1.38;
    }
    .pam-status-panel {
        min-height: 6.15rem;
        height: 6.15rem;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: flex-end;
        gap: 0.35rem;
        padding-top: 0.05rem;
    }
    .pam-status-label {
        color: #6b7280;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.05rem;
    }
    .pam-pill-row {
        display: flex;
        gap: 0.35rem;
        flex-wrap: wrap;
        justify-content: flex-end;
        align-items: flex-start;
    }
    .pam-pill {
        padding: 0.24rem 0.64rem;
        border-radius: 999px;
        color: white;
        font-size: 0.78rem;
        font-weight: 700;
        line-height: 1.1;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.18);
        white-space: nowrap;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #d1d5db !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #1f2937 !important;
        border-color: #1f2937 !important;
        color: white !important;
    }
    div[data-testid="stAudio"] {
        height: 2.75rem !important;
        min-height: 2.75rem !important;
        max-height: 2.75rem !important;
        display: flex !important;
        align-items: center !important;
        margin-top: 0 !important;
        margin-bottom: 0.35rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        overflow: hidden !important;
    }
    div[data-testid="stAudio"] > div {
        height: 2.75rem !important;
        min-height: 2.75rem !important;
        max-height: 2.75rem !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stAudio"] audio {
        width: 100% !important;
        height: 2.35rem !important;
        min-height: 2.35rem !important;
        max-height: 2.35rem !important;
        margin: 0 !important;
    }
    div[data-testid="stForm"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] button {
        margin-top: 0 !important;
    }
    div[data-testid="stImage"] {
        width: 100% !important;
        margin-top: 0.15rem !important;
        margin-bottom: 0.55rem !important;
    }
    div[data-testid="stImage"] img {
        width: 100% !important;
        height: auto !important;
    }
    div[data-testid="stExpander"] {
        margin-top: 0.35rem !important;
        margin-bottom: 0.50rem !important;
    }
    div[data-testid="stExpander"] details > summary {
        min-height: 2.65rem !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stButton"] button {
        min-height: 2.35rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Generic utilities

def _num(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def _best_prob_from_row(row: pd.Series) -> float:
    for c in ("detection_probability", "probability", "prob", "score", "class_prob", "det_prob"):
        if c in row and pd.notna(row[c]):
            try:
                v = float(row[c])
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return np.nan


def _now_iso() -> str:
    try:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    except Exception:
        return ""


def _user_name() -> str:
    return str(
        st.session_state.get("user_name")
        or st.session_state.get("auth_user")
        or st.session_state.get("user_id")
        or st.session_state.get("username")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or ""
    )


def _make_export_filename(proj_root: Path, user_name: str) -> str:
    try:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    except Exception:
        ts = "export"
    safe_user = "".join(ch for ch in str(user_name or "reviewer") if ch.isalnum() or ch in ("-", "_")).strip("_-")
    safe_user = safe_user or "reviewer"
    proj = proj_root.name or "project"
    return f"{proj}_validated_{safe_user}_{ts}.csv"


def _make_export_xlsx_filename(csv_filename: str) -> str:
    base = str(csv_filename or "validated_export.csv")
    if base.lower().endswith(".csv"):
        return base[:-4] + ".xlsx"
    return base + ".xlsx"


def _safe_widget_key(prefix: str, *parts: object) -> str:
    s = prefix + "|" + "|".join(str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"




_ADD_SPECIES_OPTION = "* Add species..."

def _clean_added_species(value: object) -> str:
    s = str(value or "").strip()
    s = " ".join(s.split())
    if s.lower() in ("nan", "none", "<na>", "[absent]", _ADD_SPECIES_OPTION.lower()):
        return ""
    return s


def _force_string_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            try:
                df[c] = df[c].astype("string")
                df[c] = df[c].fillna("")
            except Exception:
                try:
                    df[c] = df[c].astype(str).replace({"nan": "", "None": ""})
                except Exception:
                    pass
    return df


def _bool_from_any(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            v = float(x)
            return np.isfinite(v) and v != 0.0
        except Exception:
            return False
    s = str(x).strip().lower()
    return s in ("1", "1.0", "true", "yes", "y")


def _clean_group_labels(s: pd.Series, fallback: str) -> pd.Series:
    s = s.astype(str).replace({"nan": "", "None": "", "<NA>": "", "none": ""}).fillna("")
    s = s.str.strip()
    s = s.mask(s.eq(""), fallback)
    return s


def _clean_index_labels(idx: pd.Index, fallback: str = "[unknown]") -> pd.Index:
    s = pd.Series(idx.astype(str), index=range(len(idx)))
    s = s.replace({"nan": "", "None": "", "<NA>": "", "none": ""}).fillna("").str.strip()
    s = s.mask(s.eq(""), fallback)
    return pd.Index(s.tolist())


def _fmt_ms(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x * 1000:.0f} ms"


def _fmt_khz(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x / 1000:.1f} kHz"


# Dataset loading

def _load_csv_safe(p: Path) -> Optional[pd.DataFrame]:
    """Read a validation CSV once per session/file version.

    The cache is deliberately session-scoped rather than ``st.cache_data`` so a
    large detections table is not serialised into a second global cache copy.
    A changed size or mtime invalidates the entry automatically.
    """
    try:
        if not p.exists():
            return None
        stat = p.stat()
        signature = (str(p), int(stat.st_size), int(stat.st_mtime_ns))
        cache = st.session_state.setdefault("_validate_csv_session_cache", {})
        entry = cache.get(str(p))
        if isinstance(entry, dict) and entry.get("signature") == signature:
            return entry.get("df")

        df = pd.read_csv(p, low_memory=False)
        try:
            df.columns = df.columns.str.strip()
        except Exception:
            pass
        cache[str(p)] = {"signature": signature, "df": df}
        if len(cache) > 2:
            for key in list(cache.keys()):
                if key != str(p) and len(cache) > 2:
                    cache.pop(key, None)
        return df
    except Exception:
        return None


def _dataset_choice_validate(sources: dict) -> Tuple[pd.DataFrame, str, Dict[str, pd.DataFrame], Dict[str, Path]]:
    proj_root = Path(sources.get("project") or sources.get("project_root") or ".")
    data_dir = proj_root / "data_normalised"
    data_dir.mkdir(parents=True, exist_ok=True)

    p_original = data_dir / "detections_normalised.csv"
    p_valid = data_dir / "detections_validated.csv"

    choices: Dict[str, pd.DataFrame] = {}
    path_map: Dict[str, Path] = {}

    df_orig = _load_csv_safe(p_original)
    if df_orig is not None:
        choices["Original"] = df_orig
        path_map["Original"] = p_original

    df_val = _load_csv_safe(p_valid)
    if df_val is not None:
        choices["Validated (published)"] = df_val
        path_map["Validated (published)"] = p_valid

    if not choices:
        return pd.DataFrame(), "None", {}, {}

    default_label = "Validated (published)" if "Validated (published)" in choices else "Original"

    active = st.session_state.get("active_dataset_label")
    if isinstance(active, str) and active in choices:
        default_label = active

    return choices[default_label].copy(), default_label, choices, path_map


# Canonical validation prep

def _ensure_validation_ready(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    if "species_name" not in df.columns:
        df["species_name"] = df.get("class", "")
    if "presence_label" not in df.columns:
        if "FinalLabelEffective" in df.columns:
            df["presence_label"] = df["FinalLabelEffective"]
        elif "FinalLabel" in df.columns:
            df["presence_label"] = df["FinalLabel"]
        elif "label" in df.columns:
            df["presence_label"] = df["label"]
        else:
            df["presence_label"] = "present"

    if "path" not in df.columns and "file_path" in df.columns:
        df["path"] = df["file_path"]

    if "basename" not in df.columns:
        src = df.get("file_id", df.get("source_file", ""))
        df["basename"] = src.astype(str).map(lambda p: Path(p).name)

    if "filename_stem" not in df.columns:
        df["filename_stem"] = df["basename"].astype(str).map(lambda s: Path(s).stem.lower())

    if "start_s" not in df.columns and "detection_start_s" in df.columns:
        df["start_s"] = pd.to_numeric(df["detection_start_s"], errors="coerce")
    if "end_s" not in df.columns and "detection_end_s" in df.columns:
        df["end_s"] = pd.to_numeric(df["detection_end_s"], errors="coerce")

    if "detection_probability" not in df.columns:
        df["detection_probability"] = df.apply(_best_prob_from_row, axis=1)

    if "species_name_original" not in df.columns:
        df["species_name_original"] = df["species_name"]
    if "presence_label_original" not in df.columns:
        df["presence_label_original"] = df["presence_label"]

    for c, default in [
        ("validation_state", ""), ("validation_label", ""), ("validation_species", ""),
        ("validated_by", ""), ("validated_at", ""), ("validation_method", ""),
        ("validation_notes", ""),
        ("user_changed", ""), ("user_changed_by", ""), ("user_changed_at", ""),
        ("uncertain_flag", "")
    ]:
        if c not in df.columns:
            df[c] = default

    df = _force_string_cols(df, [
        "species_name", "presence_label",
        "species_name_original", "presence_label_original",
        "validation_state", "validation_label", "validation_species",
        "validated_by", "validated_at", "validation_method",
        "validation_notes",
        "user_changed", "user_changed_by", "user_changed_at",
        "uncertain_flag",
        "path", "file_path", "basename", "filename_stem",
    ])

    pleff = df["presence_label"].astype(str).str.strip().str.lower()
    df["FinalLabelEffective"] = np.where(pleff == "present", "present", "absent")

    sp = df["species_name"].astype(str)
    df["species_display"] = np.where(
        (df["FinalLabelEffective"] != "present") | (sp.str.strip() == ""),
        "[absent]",
        sp
    )

    sp0 = df["species_name_original"].astype(str)
    pl0 = df["presence_label_original"].astype(str).str.strip().str.lower()
    df["species_display_original"] = np.where(
        (pl0 != "present") | (sp0.str.strip() == ""),
        "[absent]",
        sp0
    )
    df["species_display_original"] = _clean_group_labels(df["species_display_original"], "[unknown species]")

    return df


def _apply_card_widget_state(
    det: pd.DataFrame,
    base: str,
    species_orig: str,
    selected_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    out = det.copy()

    out = _force_string_cols(out, [
        "species_name", "presence_label", "uncertain_flag",
        "species_name_original", "presence_label_original",
        "basename", "species_display_original",
    ])

    mask_card = (
        out["basename"].astype(str).eq(base)
        & out["species_display_original"].astype(str).eq(species_orig)
    )

    if selected_indices is not None:
        selected_idx_set = set(int(i) for i in selected_indices)
        mask_card = mask_card & out.index.to_series().isin(selected_idx_set)

    card_rows = out.loc[mask_card].copy()
    if card_rows.empty:
        return out

    card_rows = card_rows.sort_values("start_s")
    card_rows["__orig_index"] = card_rows.index
    rgdf = card_rows.reset_index(drop=True)

    for ridx, row in rgdf.iterrows():
        idx = int(row["__orig_index"])

        sp_key = f"sp_{base}_{species_orig}_{ridx}"
        unc_key = f"unc_{base}_{species_orig}_{ridx}"
        note_key = f"note_{base}_{species_orig}_{ridx}"

        current_presence = str(row.get("presence_label", "") or "").strip().lower()
        current_species = str(row.get("species_name", "") or "")

        choice = st.session_state.get(sp_key, None)
        if choice is None:
            choice = "[absent]" if (current_presence != "present" or current_species.strip() == "") else current_species

        if choice == _ADD_SPECIES_OPTION:
            choice = _clean_added_species(st.session_state.get(f"{sp_key}_new", ""))
            if choice:
                st.session_state[sp_key] = choice

        if choice == "[absent]" or not str(choice).strip():
            out.at[idx, "species_name"] = ""
            out.at[idx, "presence_label"] = "absent"
        else:
            out.at[idx, "species_name"] = str(choice).strip()
            out.at[idx, "presence_label"] = "present"

        current_unc = st.session_state.get(unc_key, _bool_from_any(row.get("uncertain_flag", "")))
        out.at[idx, "uncertain_flag"] = "1" if bool(current_unc) else ""

        current_note = st.session_state.get(note_key, row.get("validation_notes", ""))
        out.at[idx, "validation_notes"] = str(current_note or "").strip()

    return out


def _apply_card_submitted_values(
    det: pd.DataFrame,
    base: str,
    species_orig: str,
    submitted_values: Dict[int, Dict[str, object]],
    selected_indices: Optional[List[int]] = None,
) -> pd.DataFrame:
    out = det.copy()

    out = _force_string_cols(out, [
        "species_name", "presence_label", "uncertain_flag",
        "species_name_original", "presence_label_original",
        "basename", "species_display_original",
    ])

    mask_card = (
        out["basename"].astype(str).eq(base)
        & out["species_display_original"].astype(str).eq(species_orig)
    )

    if selected_indices is not None:
        selected_idx_set = set(int(i) for i in selected_indices)
        mask_card = mask_card & out.index.to_series().isin(selected_idx_set)

    card_rows = out.loc[mask_card].copy()
    if card_rows.empty:
        return out

    card_rows = card_rows.sort_values("start_s")
    card_rows["__orig_index"] = card_rows.index
    rgdf = card_rows.reset_index(drop=True)

    for ridx, row in rgdf.iterrows():
        idx = int(row["__orig_index"])
        values = submitted_values.get(int(ridx), {}) if submitted_values else {}

        current_presence = str(row.get("presence_label", "") or "").strip().lower()
        current_species = str(row.get("species_name", "") or "")
        default_choice = "[absent]" if (current_presence != "present" or current_species.strip() == "") else current_species

        choice = values.get("species_value", default_choice)
        if choice == _ADD_SPECIES_OPTION:
            choice = _clean_added_species(values.get("new_species_value", ""))

        if choice == "[absent]" or not str(choice).strip():
            out.at[idx, "species_name"] = ""
            out.at[idx, "presence_label"] = "absent"
        else:
            out.at[idx, "species_name"] = str(choice).strip()
            out.at[idx, "presence_label"] = "present"

        current_unc = values.get("uncertain_value", _bool_from_any(row.get("uncertain_flag", "")))
        out.at[idx, "uncertain_flag"] = "1" if bool(current_unc) else ""

        current_note = values.get("note_value", row.get("validation_notes", ""))
        out.at[idx, "validation_notes"] = str(current_note or "").strip()

    return out


# Audio path + TE helpers

def _is_abs_like(p: str) -> bool:
    p = (p or "").strip()
    if not p:
        return False
    if len(p) >= 2 and p[1] == ":":
        return True
    if p.startswith("\\\\") or p.startswith("//"):
        return True
    try:
        return Path(p).is_absolute()
    except Exception:
        return False


def _resolve_audio_candidate(proj_root: Path, p: str) -> Optional[Path]:
    p = (p or "").strip()
    if not p:
        return None

    cand = Path(p) if _is_abs_like(p) else (proj_root / p)

    try:
        cand = cand.expanduser()
    except Exception:
        pass

    try:
        cand = cand.resolve()
    except Exception:
        try:
            cand = Path(os.path.normpath(str(cand)))
        except Exception:
            pass

    return cand if cand.exists() else None


def _resolve_audio_path(proj_root: Path, row_or_df, df_all: pd.DataFrame) -> Optional[Path]:
    if isinstance(row_or_df, pd.Series):
        rows = [row_or_df]
    else:
        rows = [row_or_df.iloc[0]] if len(row_or_df) else []

    for r in rows:
        for col in ("file_path", "path", "file_path_rel", "file_path_abs", "file_path_original", "original_path"):
            p = r.get(col)
            if isinstance(p, str) and p.strip():
                cand = _resolve_audio_candidate(proj_root, p)
                if cand is not None:
                    return cand

    cand_cols = [c for c in ("file_path", "path", "file_path_rel", "file_path_abs", "file_path_original", "original_path") if c in df_all.columns]
    if not cand_cols:
        return None

    if isinstance(row_or_df, pd.Series):
        stem = Path(str(row_or_df.get("basename", row_or_df.get("source_file", "")))).stem.lower()
    else:
        s = row_or_df.iloc[0]
        stem = Path(str(s.get("basename", s.get("source_file", "")))).stem.lower()

    rows2 = df_all[df_all["filename_stem"] == stem]
    if rows2.empty:
        return None

    for col in cand_cols:
        for q in rows2[col]:
            if isinstance(q, str) and q.strip():
                cand = _resolve_audio_candidate(proj_root, q)
                if cand is not None:
                    return cand

    for col in cand_cols:
        q = rows2[col].dropna().astype(str).head(1)
        if not q.empty:
            cand = _resolve_audio_candidate(proj_root, str(q.iloc[0]))
            if cand is not None:
                return cand

    return None



def _rows_for_resolved_audio(proj_root: Path, df: pd.DataFrame, apath: Optional[Path]) -> pd.DataFrame:
    if apath is None or df is None or df.empty:
        return df
    try:
        target = apath.resolve()
    except Exception:
        target = Path(os.path.normpath(str(apath)))
    path_cols = [c for c in ("file_path", "path", "file_path_rel", "file_path_abs", "file_path_original", "original_path") if c in df.columns]
    if not path_cols:
        return df
    keep = []
    for _, row in df.iterrows():
        matched = False
        for col in path_cols:
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                cand = _resolve_audio_candidate(proj_root, val)
                if cand is None:
                    continue
                try:
                    if cand.resolve() == target:
                        matched = True
                        break
                except Exception:
                    if os.path.normcase(os.path.normpath(str(cand))) == os.path.normcase(os.path.normpath(str(target))):
                        matched = True
                        break
        keep.append(matched)
    if any(keep):
        return df.loc[keep].copy()
    return df

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


def _largest_valid_fft_at_or_below(limit: int) -> Optional[int]:
    allowed_ffts = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    valid = [v for v in allowed_ffts if v <= int(limit)]
    return max(valid) if valid else None


def _card_metric_fft(gdf: pd.DataFrame, y: np.ndarray, sr: int, requested_n_fft: int) -> Optional[int]:
    if y.size == 0 or sr <= 0 or gdf.empty:
        return None

    seg_lengths: List[int] = []
    for _, row in gdf.iterrows():
        start_s = _num(row.get("start_s", row.get("detection_start_s")))
        end_s = _num(row.get("end_s", row.get("detection_end_s")))
        if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
            continue

        s0 = max(0, int(round(start_s * sr)))
        s1 = min(len(y), int(round(end_s * sr)))
        seg_len = int(s1 - s0)
        if seg_len >= 32:
            seg_lengths.append(seg_len)

    if not seg_lengths:
        return None

    shortest_seg = min(seg_lengths)
    return _largest_valid_fft_at_or_below(min(int(requested_n_fft), int(shortest_seg)))


def _group_max_prob(gdf: pd.DataFrame) -> float:
    ps = pd.to_numeric(gdf.get("detection_probability"), errors="coerce")
    return float(ps.max()) if ps.notna().any() else -np.inf


def _tmp_audio_path(proj_root: Path, base: str, species_line: str, te: int, sr: int, n: int) -> Path:
    ws = proj_root / "workspace" / "tmp_audio"
    ws.mkdir(parents=True, exist_ok=True)
    key = f"{base}|{species_line}|te={te}|sr={sr}|n={n}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return ws / f"play_{h}.wav"


def _get_validate_n_fft(sr: int) -> int:
    if bool(st.session_state.get("validate_use_fft_override", False)):
        return int(st.session_state.get("validate_fft_size", 4096))
    return 8192 if sr > 48_000 else 4096

def _match_frame_count(x: np.ndarray, n_frames: int) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size == n_frames:
        return arr
    if arr.size == 0:
        return np.full(n_frames, np.nan, dtype=float)
    if arr.size > n_frames:
        return arr[:n_frames]
    pad = np.full(n_frames - arr.size, arr[-1], dtype=float)
    return np.concatenate([arr, pad])


def _audio_file_signature(apath: Path) -> Tuple[int, int]:
    try:
        stat = apath.stat()
        return int(stat.st_size), int(stat.st_mtime_ns)
    except Exception:
        return 0, 0


@st.cache_data(show_spinner=False, max_entries=256)
def _audio_info_cached(apath_str: str, file_size: int, file_mtime_ns: int) -> Tuple[int, float]:
    apath = Path(apath_str)
    try:
        info = sf.info(str(apath))
        sr = int(info.samplerate or 0)
        frames = int(info.frames or 0)
        dur = float(frames / sr) if sr > 0 else 0.0
        return sr, max(0.0, dur)
    except Exception:
        try:
            y_tmp, sr_tmp = librosa.load(str(apath), sr=None, mono=True, duration=0.01)
            sr = int(sr_tmp or 0)
            return sr, float(max(0.0, len(y_tmp) / sr)) if sr > 0 else 0.0
        except Exception:
            return 0, 0.0


def _audio_info(apath: Path) -> Tuple[int, float]:
    file_size, file_mtime_ns = _audio_file_signature(apath)
    return _audio_info_cached(str(apath), file_size, file_mtime_ns)


@st.cache_data(show_spinner=False, max_entries=128)
def _load_audio_window_cached(
    apath_str: str,
    start_s: float,
    end_s: float,
    fallback_sr: int,
    file_size: int,
    file_mtime_ns: int,
) -> Tuple[np.ndarray, int, float, float]:
    apath = Path(apath_str)
    start_s = max(0.0, float(start_s)) if np.isfinite(_num(start_s)) else 0.0
    end_s = max(start_s, float(end_s)) if np.isfinite(_num(end_s)) else start_s

    try:
        info = sf.info(str(apath))
        sr = int(info.samplerate or fallback_sr or 0)
        total_frames = int(info.frames or 0)
        if sr <= 0 or total_frames <= 0:
            raise ValueError("invalid audio metadata")

        start_frame = int(max(0, min(total_frames, round(start_s * sr))))
        end_frame = int(max(start_frame + 1, min(total_frames, round(end_s * sr))))
        frames = int(max(1, end_frame - start_frame))
        y, _ = sf.read(
            str(apath),
            start=start_frame,
            frames=frames,
            dtype="float32",
            always_2d=False,
        )
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:
            y = y.mean(axis=1).astype(np.float32, copy=False)
        actual_start = float(start_frame / sr)
        actual_end = float((start_frame + len(y)) / sr)
        return y, sr, actual_start, actual_end
    except Exception:
        duration = max(0.0, float(end_s - start_s)) if end_s > start_s else None
        try:
            y, sr = librosa.load(str(apath), sr=None, mono=True, offset=float(start_s), duration=duration)
            return y.astype(np.float32, copy=False), int(sr), float(start_s), float(start_s + (len(y) / sr if sr else 0.0))
        except Exception:
            return np.array([], dtype=np.float32), int(fallback_sr or 1), float(start_s), float(start_s)


def _load_audio_window(apath: Path, start_s: float, end_s: float, fallback_sr: int = 0) -> Tuple[np.ndarray, int, float, float]:
    file_size, file_mtime_ns = _audio_file_signature(apath)
    return _load_audio_window_cached(
        str(apath),
        round(float(start_s), 3) if np.isfinite(_num(start_s)) else 0.0,
        round(float(end_s), 3) if np.isfinite(_num(end_s)) else 0.0,
        int(fallback_sr or 0),
        file_size,
        file_mtime_ns,
    )


def _default_detection_window(
    boxes: List[Dict[str, float]],
    duration_s: float,
    default_single_window_s: float,
    padding_s: float = 2.0,
) -> Tuple[float, float, Tuple[object, ...]]:
    dur = max(0.0, float(duration_s)) if np.isfinite(_num(duration_s)) else 0.0
    valid_boxes = [
        b for b in boxes
        if np.isfinite(_num(b.get("start_s")))
        and np.isfinite(_num(b.get("end_s")))
        and float(b.get("end_s")) > float(b.get("start_s"))
    ]

    if dur <= 0.0 or not valid_boxes:
        return 0.0, max(1e-6, dur), ("full", round(float(dur), 3))

    if len(valid_boxes) == 1 and bool(st.session_state.get("validate_auto_zoom_single_detection", True)):
        b = valid_boxes[0]
        width = min(dur, max(1.0, float(default_single_window_s)))
        centre = 0.5 * (float(b["start_s"]) + float(b["end_s"]))
        start = centre - width * 0.5
        end = centre + width * 0.5
        mode = "single"
    else:
        start = min(float(b["start_s"]) for b in valid_boxes) - max(0.0, float(padding_s))
        end = max(float(b["end_s"]) for b in valid_boxes) + max(0.0, float(padding_s))
        mode = "detections"

    if start < 0.0:
        end = min(dur, end - start)
        start = 0.0
    if end > dur:
        shift = end - dur
        start = max(0.0, start - shift)
        end = dur
    if end <= start:
        start, end = 0.0, dur

    signature = (
        mode,
        tuple((round(float(b["start_s"]), 3), round(float(b["end_s"]), 3)) for b in valid_boxes),
        round(float(default_single_window_s), 3),
        round(float(padding_s), 3),
        round(float(dur), 3),
    )
    return float(start), float(end), signature


def _offset_detection_times(gdf: pd.DataFrame, offset_s: float) -> pd.DataFrame:
    out = gdf.copy()
    for col in ("start_s", "end_s", "detection_start_s", "detection_end_s"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") - float(offset_s)
    return out

@st.cache_data(show_spinner=False, max_entries=128)
def _compute_static_spectrogram_data(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
) -> Dict[str, np.ndarray]:
    """Compute only what the static validation card actually displays."""
    out = {
        "S_dB": np.zeros((2, 2), dtype=float),
        "times": np.zeros(2, dtype=float),
        "freqs_hz": np.zeros(2, dtype=float),
    }
    if y.size == 0 or sr <= 0:
        return out
    D = librosa.stft(y=y, n_fft=int(n_fft), hop_length=int(hop_length))
    S_power = np.abs(D) ** 2
    if S_power.size == 0:
        return out
    out["S_dB"] = librosa.power_to_db(S_power, ref=np.max, top_db=90)
    out["times"] = librosa.frames_to_time(
        np.arange(S_power.shape[1]), sr=sr, hop_length=hop_length
    )
    out["freqs_hz"] = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return out


@st.cache_data(show_spinner=False, max_entries=128)
def _compute_spectrogram_data(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
) -> Dict[str, np.ndarray]:
    out = {
        "S_power": np.zeros((2, 2), dtype=float),
        "S_dB": np.zeros((2, 2), dtype=float),
        "times": np.zeros(2, dtype=float),
        "freqs_hz": np.zeros(2, dtype=float),
        "frame_peak_freq_hz": np.zeros(2, dtype=float),
        "frame_centroid_hz": np.zeros(2, dtype=float),
        "frame_bandwidth_hz": np.zeros(2, dtype=float),
        "frame_rolloff_hz": np.zeros(2, dtype=float),
        "frame_flatness": np.zeros(2, dtype=float),
        "frame_rms": np.zeros(2, dtype=float),
        "frame_zcr": np.zeros(2, dtype=float),
    }

    if y.size == 0 or sr <= 0:
        return out

    D = librosa.stft(y=y, n_fft=int(n_fft), hop_length=int(hop_length))
    S_power = np.abs(D) ** 2
    if S_power.size == 0:
        return out

    S_mag = np.sqrt(S_power)
    S_dB = librosa.power_to_db(S_power, ref=np.max, top_db=90)
    times = librosa.frames_to_time(np.arange(S_power.shape[1]), sr=sr, hop_length=hop_length)
    freqs_hz = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_frames = S_power.shape[1]

    frame_peak_idx = np.argmax(S_power, axis=0)
    frame_peak_freq_hz = freqs_hz[frame_peak_idx]

    denom = S_power.sum(axis=0)
    frame_centroid_hz = np.full(n_frames, np.nan, dtype=float)
    valid = denom > 0
    if np.any(valid):
        frame_centroid_hz[valid] = (freqs_hz[:, None] * S_power)[:, valid].sum(axis=0) / denom[valid]

    try:
        frame_bandwidth_hz = librosa.feature.spectral_bandwidth(S=S_mag, sr=sr)[0]
    except Exception:
        frame_bandwidth_hz = np.full(n_frames, np.nan, dtype=float)

    try:
        frame_rolloff_hz = librosa.feature.spectral_rolloff(S=S_mag, sr=sr, roll_percent=0.85)[0]
    except Exception:
        frame_rolloff_hz = np.full(n_frames, np.nan, dtype=float)

    try:
        frame_flatness = librosa.feature.spectral_flatness(S=S_mag)[0]
    except Exception:
        frame_flatness = np.full(n_frames, np.nan, dtype=float)

    try:
        frame_rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length, center=True)[0]
    except Exception:
        frame_rms = np.full(n_frames, np.nan, dtype=float)

    try:
        frame_zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length, center=True)[0]
    except Exception:
        frame_zcr = np.full(n_frames, np.nan, dtype=float)

    out["S_power"] = S_power
    out["S_dB"] = S_dB
    out["times"] = times
    out["freqs_hz"] = freqs_hz
    out["frame_peak_freq_hz"] = _match_frame_count(frame_peak_freq_hz, n_frames)
    out["frame_centroid_hz"] = _match_frame_count(frame_centroid_hz, n_frames)
    out["frame_bandwidth_hz"] = _match_frame_count(frame_bandwidth_hz, n_frames)
    out["frame_rolloff_hz"] = _match_frame_count(frame_rolloff_hz, n_frames)
    out["frame_flatness"] = _match_frame_count(frame_flatness, n_frames)
    out["frame_rms"] = _match_frame_count(frame_rms, n_frames)
    out["frame_zcr"] = _match_frame_count(frame_zcr, n_frames)
    return out

def _plotly_spectrogram_figure(
    S_dB: np.ndarray,
    times: np.ndarray,
    freqs_hz: np.ndarray,
    frame_peak_freq_hz: np.ndarray,
    frame_centroid_hz: np.ndarray,
    frame_bandwidth_hz: np.ndarray,
    frame_rolloff_hz: np.ndarray,
    frame_flatness: np.ndarray,
    frame_rms: np.ndarray,
    frame_zcr: np.ndarray,
    boxes: List[Dict[str, float]],
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> go.Figure:
    zmax = float(np.nanmax(S_dB)) if np.size(S_dB) else 0.0
    zmin = zmax - 90.0

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=S_dB,
            x=times,
            y=freqs_hz,
            colorscale="Viridis",
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="dB"),
            hovertemplate=(
                "<b>Cursor</b><br>"
                "Time: %{x:.3f} s<br>"
                "Frequency: %{y:.0f} Hz<br>"
                "Level: %{z:.1f} dB"
                "<extra></extra>"
            ),
        )
    )

    frame_customdata = np.column_stack([
        frame_peak_freq_hz,
        frame_centroid_hz,
        frame_bandwidth_hz,
        frame_rolloff_hz,
        frame_flatness,
        frame_rms,
        frame_zcr,
    ])

    hover_y = np.full(len(times), ymin + 0.5 * (ymax - ymin), dtype=float)

    fig.add_trace(
        go.Scatter(
            x=times,
            y=hover_y,
            mode="markers",
            marker=dict(size=16, opacity=0),
            customdata=frame_customdata,
            hovertemplate=(
                "<b>Frame summary</b><br>"
                "Time: %{x:.3f} s<br>"
                "Frame peak frequency: %{customdata[0]:.0f} Hz<br>"
                "Frame centroid: %{customdata[1]:.0f} Hz<br>"
                "Frame bandwidth: %{customdata[2]:.0f} Hz<br>"
                "Frame rolloff (85%): %{customdata[3]:.0f} Hz<br>"
                "Frame flatness: %{customdata[4]:.4f}<br>"
                "Frame RMS: %{customdata[5]:.5f}<br>"
                "Frame ZCR: %{customdata[6]:.5f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    for b in boxes:
        x0 = float(b["start_s"])
        x1 = float(b["end_s"])
        low_f = _num(b.get("low_freq"))
        high_f = _num(b.get("high_freq"))
        prob = b.get("prob", np.nan)

        y0 = low_f if np.isfinite(low_f) else ymin
        y1 = high_f if np.isfinite(high_f) and high_f > y0 else ymax

        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line=dict(width=1, color="rgba(255,255,255,0.22)"),
            fillcolor="rgba(255,255,255,0.08)",
        )

        if np.isfinite(prob):
            fig.add_annotation(
                x=(x0 + x1) * 0.5,
                y=ymin + 0.88 * (ymax - ymin),
                text=f"{prob:.2f}",
                showarrow=False,
                bgcolor="rgba(0,0,0,0.55)",
                bordercolor="rgba(255,255,255,0.25)",
                font=dict(size=11, color="white"),
            )

    tick_step = 1000 if (ymax - ymin) <= 15000 else 5000
    tick_vals = np.arange(max(0, int(ymin // 1000) * 1000), int(ymax) + 1, tick_step)

    fig.update_xaxes(title_text="Time (s)", range=[xmin, xmax], fixedrange=True)
    fig.update_yaxes(
        title_text="Frequency (kHz)",
        range=[ymin, ymax],
        tickvals=tick_vals.tolist(),
        ticktext=[f"{v/1000:.0f}" for v in tick_vals],
        fixedrange=True,
    )

    fig.update_layout(
        height=560,
        autosize=False,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="closest",
    )

    return fig


def _render_interactive_validate_dialog(
    proj_root: Path,
    df_all: pd.DataFrame,
    grouped,
    base: str,
    species_orig: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n_fft: int,
    hop_length: int,
):
    gdf_int = grouped.get_group((base, species_orig)).copy()
    apath_int = _resolve_audio_path(proj_root, gdf_int, df_all)

    if not (apath_int and apath_int.exists()):
        st.error("Audio not found for the selected card.")
        return

    boxes_int: List[Dict[str, float]] = []
    for _, row in gdf_int.iterrows():
        b = {
            "start_s": _num(row.get("start_s", row.get("detection_start_s"))),
            "end_s": _num(row.get("end_s", row.get("detection_end_s"))),
            "low_freq": _num(row.get("low_freq")),
            "high_freq": _num(row.get("high_freq")),
            "prob": _num(row.get("detection_probability")),
        }
        if np.isfinite(b["start_s"]) and np.isfinite(b["end_s"]) and b["end_s"] > b["start_s"]:
            boxes_int.append(b)

    if boxes_int:
        boxes_int = sorted(
            boxes_int,
            key=lambda b: (b["prob"] if np.isfinite(b["prob"]) else -1.0),
            reverse=True,
        )[:10]

    try:
        sr_int, dur_int = _audio_info(apath_int)
        if sr_int <= 0 or dur_int <= 0:
            raise ValueError("Audio metadata could not be read")

        xmin_int = max(0.0, float(xmin))
        xmax_int = min(float(dur_int), float(xmax))
        if xmax_int <= xmin_int:
            xmin_int, xmax_int, _ = _default_detection_window(
                boxes=boxes_int,
                duration_s=dur_int,
                default_single_window_s=float(
                    st.session_state.get("validate_auto_zoom_window_s", 5.0)
                ),
                padding_s=2.0,
            )

        y_int, sr_int, actual_start_int, actual_end_int = _load_audio_window(
            apath_int,
            xmin_int,
            xmax_int,
            fallback_sr=sr_int,
        )
        if y_int.size == 0 or sr_int <= 0:
            raise ValueError("Audio window could not be read")
        xmin_int = float(actual_start_int)
        xmax_int = float(actual_end_int)
    except Exception as e:
        st.error(f"Audio read error: {e}")
        return

    n_fft_int = max(2, int(n_fft))
    hop_int = max(1, int(hop_length))

    nyq_int = 0.5 * sr_int * 0.98
    ymin_int = max(0.0, float(ymin))
    ymax_int = min(float(ymax), nyq_int)
    if ymax_int <= ymin_int:
        ymin_int = 0.0
        ymax_int = nyq_int

    spec_int = _compute_spectrogram_data(
        y=y_int,
        sr=sr_int,
        n_fft=n_fft_int,
        hop_length=hop_int,
    )

    S_dB_int = spec_int["S_dB"]
    times_int = spec_int["times"] + float(xmin_int)
    freqs_hz_int = spec_int["freqs_hz"]
    frame_peak_freq_hz_int = spec_int["frame_peak_freq_hz"]
    frame_centroid_hz_int = spec_int["frame_centroid_hz"]
    frame_bandwidth_hz_int = spec_int["frame_bandwidth_hz"]
    frame_rolloff_hz_int = spec_int["frame_rolloff_hz"]
    frame_flatness_int = spec_int["frame_flatness"]
    frame_rms_int = spec_int["frame_rms"]
    frame_zcr_int = spec_int["frame_zcr"]

    fig_int = _plotly_spectrogram_figure(
        S_dB=S_dB_int,
        times=times_int,
        freqs_hz=freqs_hz_int,
        frame_peak_freq_hz=frame_peak_freq_hz_int,
        frame_centroid_hz=frame_centroid_hz_int,
        frame_bandwidth_hz=frame_bandwidth_hz_int,
        frame_rolloff_hz=frame_rolloff_hz_int,
        frame_flatness=frame_flatness_int,
        frame_rms=frame_rms_int,
        frame_zcr=frame_zcr_int,
        boxes=boxes_int,
        xmin=xmin_int,
        xmax=xmax_int,
        ymin=ymin_int,
        ymax=ymax_int,
    )

    st.caption(f"{base} • {species_orig}")
    st.plotly_chart(
        fig_int,
        config={"displayModeBar": True},
    )

def _acoustic_metrics_for_detection(
    y: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    low_freq: Optional[float],
    high_freq: Optional[float],
    n_fft: int,
    hop_length: int,
) -> Dict[str, float]:
    out = {
        "duration_s": np.nan,
        "peak_freq_hz": np.nan,
        "centroid_hz": np.nan,
        "effective_n_fft": np.nan,
    }

    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        return out

    out["duration_s"] = float(end_s - start_s)

    s0 = max(0, int(round(start_s * sr)))
    s1 = min(len(y), int(round(end_s * sr)))
    if s1 <= s0:
        return out

    y_seg = y[s0:s1]
    seg_len = int(y_seg.size)
    if seg_len < 32:
        return out

    try:
        local_n_fft = int(max(32, n_fft))
        if local_n_fft > seg_len:
            return out

        out["effective_n_fft"] = float(local_n_fft)
        local_hop = max(1, min(int(hop_length), local_n_fft // 8))

        S = np.abs(librosa.stft(y=y_seg, n_fft=local_n_fft, hop_length=local_hop)) ** 2
        if S.size == 0:
            return out

        mean_spectrum = S.mean(axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=local_n_fft)

        band_mask = np.ones_like(freqs, dtype=bool)
        lf = _num(low_freq)
        hf = _num(high_freq)
        if np.isfinite(lf) and np.isfinite(hf) and hf > lf:
            band_mask = (freqs >= float(lf)) & (freqs <= float(hf))

        freqs_band = freqs[band_mask]
        spec_band = mean_spectrum[band_mask]

        if spec_band.size and np.any(spec_band > 0):
            out["peak_freq_hz"] = float(freqs_band[np.argmax(spec_band)])

        denom = float(np.sum(spec_band))
        if spec_band.size and denom > 0:
            out["centroid_hz"] = float(np.sum(freqs_band * spec_band) / denom)
    except Exception:
        pass

    return out


def _compute_detection_acoustic_summary(
    y: np.ndarray,
    sr: int,
    gdf: pd.DataFrame,
    n_fft: int,
    hop_length: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if y.size == 0 or gdf.empty:
        return pd.DataFrame()

    gdf2 = gdf.copy().sort_values("start_s").reset_index(drop=True)

    for ridx, row in gdf2.iterrows():
        start_s = _num(row.get("start_s", row.get("detection_start_s")))
        end_s = _num(row.get("end_s", row.get("detection_end_s")))
        low_freq = _num(row.get("low_freq"))
        high_freq = _num(row.get("high_freq"))
        prob = _num(row.get("detection_probability"))

        metrics = _acoustic_metrics_for_detection(
            y=y,
            sr=sr,
            start_s=start_s,
            end_s=end_s,
            low_freq=low_freq,
            high_freq=high_freq,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        rows.append({
            "Detection": int(ridx + 1),
            "Start": f"{start_s:.2f}s" if np.isfinite(start_s) else "—",
            "Duration": _fmt_ms(metrics["duration_s"]),
            "Peak energy freq": _fmt_khz(metrics["peak_freq_hz"]),
            "Centroid": _fmt_khz(metrics["centroid_hz"]),
            "Prob": f"{prob:.2f}" if np.isfinite(prob) else "—",
        })

    return pd.DataFrame(rows)


# Strategy persistence helpers

def _strategy_store_path(proj_root: Path, user_name: str) -> Path:
    safe_user = "".join(ch for ch in str(user_name or "default_user") if ch.isalnum() or ch in ("-", "_")).strip("_-")
    safe_user = safe_user or "default_user"
    d = proj_root / "workspace" / "user_settings"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"validate_strategy_{safe_user}.json"


def _strategy_state_payload() -> Dict[str, object]:
    keys = [
        "validate_strategy_goal",
        "validate_strategy_balance",
        "validate_strategy_target_mode",
        "validate_strategy_target_value",
        "validate_strategy_bins",
        "validate_strategy_seed",
        "validate_strategy_available",
        "validate_strategy_selected",
        "validate_strategy_strata",
        "validate_strategy_undersized",
        "validate_strategy_metrics_source",
        "validate_strategy_dont_auto_show",
        "validate_strategy_prompt_seen",
    ]
    return {k: st.session_state.get(k) for k in keys}


def _save_strategy_state(proj_root: Path) -> None:
    try:
        p = _strategy_store_path(proj_root, _user_name())
        with open(p, "w", encoding="utf-8") as f:
            json.dump(_strategy_state_payload(), f, indent=2)
    except Exception:
        pass


def _load_strategy_state(proj_root: Path) -> None:
    loaded_flag = "_validate_strategy_loaded_once"
    current_user = _user_name()
    current_key = f"{str(proj_root.resolve())}|{current_user}"

    if st.session_state.get(loaded_flag) == current_key:
        return

    try:
        p = _strategy_store_path(proj_root, current_user)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)

            allowed_keys = {
                "validate_strategy_goal",
                "validate_strategy_balance",
                "validate_strategy_target_mode",
                "validate_strategy_target_value",
                "validate_strategy_bins",
                "validate_strategy_seed",
                "validate_strategy_available",
                "validate_strategy_selected",
                "validate_strategy_strata",
                "validate_strategy_undersized",
                "validate_strategy_metrics_source",
                "validate_strategy_dont_auto_show",
                "validate_strategy_prompt_seen",
            }
            for k, v in payload.items():
                if k in allowed_keys:
                    st.session_state[k] = v
    except Exception:
        pass

    st.session_state[loaded_flag] = current_key


def _sync_validate_page_input_from_page():
    if st.session_state.get("validate_page_sync_pending", False):
        st.session_state["validate_page_input"] = int(st.session_state.get("validate_page", 1))
        st.session_state["validate_page_sync_pending"] = False


def _on_validate_page_input_change():
    st.session_state["validate_page"] = int(st.session_state.get("validate_page_input", 1))


def _reset_validate_page_for_sort_change():
    st.session_state["validate_page"] = 1
    st.session_state["validate_page_input"] = 1
    st.session_state["validate_page_sync_pending"] = False


def _clear_validate_time_window_state():
    for k in list(st.session_state.keys()):
        if str(k).startswith((
            "validate_time_window_state_",
            "validate_time_xmin_input_",
            "validate_time_xmax_input_",
        )):
            del st.session_state[k]


def _mark_validate_time_window_override(state_key: str):
    state = dict(st.session_state.get(state_key, {}))
    state["user_override"] = True
    st.session_state[state_key] = state


def _go_to_previous_validate_page():
    current_page = int(st.session_state.get("validate_page", 1))
    st.session_state["validate_page"] = max(1, current_page - 1)
    st.session_state["validate_page_sync_pending"] = True


def _go_to_next_validate_page():
    current_page = int(st.session_state.get("validate_page", 1))
    total_pages = int(st.session_state.get("_validate_total_pages", 1))
    st.session_state["validate_page"] = min(total_pages, current_page + 1)
    st.session_state["validate_page_sync_pending"] = True


# Strategy helpers

def _strategy_balance_options(df: pd.DataFrame, goal: Optional[str] = None) -> Dict[str, str]:
    if goal in ("find_likely_mistakes", "review_strongest"):
        opts = {"all": "All clips"}
        if "species_display_original" in df.columns:
            opts["species"] = "Species"
        if "site" in df.columns:
            opts["site"] = "Site"
        if "recorder_id" in df.columns:
            opts["recorder"] = "Recorder"
        return opts

    if goal == "equal_allocation":
        return {"all": "Confidence bands only"}

    opts = {"all": "All clips"}
    if "species_display_original" in df.columns:
        opts["species"] = "Species"
        opts["species_confidence"] = "Species + confidence"
    if "site" in df.columns:
        opts["site"] = "Site"
        opts["site_confidence"] = "Site + confidence"
    if "recorder_id" in df.columns:
        opts["recorder"] = "Recorder"
        opts["recorder_confidence"] = "Recorder + confidence"
    return opts


def _strategy_group_series(df: pd.DataFrame, balance: str) -> pd.Series:
    if balance.startswith("species"):
        raw = df.get("species_display_original", pd.Series([""] * len(df), index=df.index))
        return _clean_group_labels(raw, "[unknown species]")
    if balance.startswith("site"):
        raw = df.get("site", pd.Series([""] * len(df), index=df.index))
        return _clean_group_labels(raw, "[unknown site]")
    if balance.startswith("recorder"):
        raw = df.get("recorder_id", pd.Series([""] * len(df), index=df.index))
        return _clean_group_labels(raw, "[unknown recorder]")
    return pd.Series(["all"] * len(df), index=df.index)


def _strategy_parent_label(balance: str) -> str:
    if balance.startswith("species"):
        return "Species"
    if balance.startswith("site"):
        return "Site"
    if balance.startswith("recorder"):
        return "Recorder"
    return "Group"


def _strategy_goal_label(goal: str) -> str:
    return {
        "representative_sample": "Representative sample",
        "find_likely_mistakes": "Find likely mistakes",
        "review_strongest": "Review strongest detections",
        "custom_stratified": "Custom stratified plan",
        "equal_allocation": "Equal allocation",
    }.get(goal, "Representative sample")


def _strategy_balance_label(balance: str, df: pd.DataFrame, goal: Optional[str] = None) -> str:
    return _strategy_balance_options(df, goal).get(balance, "All clips")


def _strategy_target_summary(value: int, mode: str) -> str:
    if mode == "per_group_percent":
        return f"{int(value)}% per group"
    if mode == "per_group_clips":
        return f"{int(value)} clips per group"
    return f"{int(value)} clips"


def _strategy_defaults_for_goal(goal: str, df_len: int) -> Tuple[str, int]:
    df_len = max(1, int(df_len))
    if goal == "custom_stratified":
        return "per_group_percent", 10
    if goal == "equal_allocation":
        return "total_clips", min(200, df_len)
    if goal == "find_likely_mistakes":
        return "total_clips", min(100, df_len)
    if goal == "review_strongest":
        return "total_clips", min(100, df_len)
    return "total_clips", min(200, df_len)


def _target_value_for_widget(
    goal: str,
    target_mode: str,
    stored_value: int,
    df_len: int,
) -> int:
    default_mode, default_value = _strategy_defaults_for_goal(goal, df_len)

    if goal != "custom_stratified":
        return int(max(1, min(default_value, max(1, df_len))))

    if target_mode == "per_group_percent":
        if 1 <= int(stored_value) <= 100:
            return int(stored_value)
        return 10

    if int(stored_value) >= 1:
        return int(min(int(stored_value), max(1, df_len)))

    if default_mode == "per_group_percent":
        return 10
    return int(max(1, min(default_value, max(1, df_len))))


def _strategy_presets(df_len: int) -> Dict[str, Dict[str, object]]:
    default_total = min(200, max(1, int(df_len)))
    review_total = min(100, max(1, int(df_len)))
    return {
        "Representative sample": {
            "goal": "representative_sample",
            "balance": "species_confidence",
            "target_mode": "total_clips",
            "target_value": default_total,
            "bins": 5,
            "seed": 42,
            "description": "Balanced sampling across species and confidence bands."
        },
        "Likely mistakes": {
            "goal": "find_likely_mistakes",
            "balance": "species",
            "target_mode": "total_clips",
            "target_value": review_total,
            "bins": 5,
            "seed": 42,
            "description": "Lowest-confidence detections, balanced across species."
        },
        "Strongest detections": {
            "goal": "review_strongest",
            "balance": "all",
            "target_mode": "total_clips",
            "target_value": review_total,
            "bins": 5,
            "seed": 42,
            "description": "Highest-confidence detections, regardless of group."
        },
        "Equal allocation": {
            "goal": "equal_allocation",
            "balance": "all",
            "target_mode": "total_clips",
            "target_value": default_total,
            "bins": 5,
            "seed": 42,
            "description": "Even spread across confidence bands."
        },
        "Custom": {
            "goal": "custom_stratified",
            "balance": "species_confidence",
            "target_mode": "per_group_percent",
            "target_value": 10,
            "bins": 5,
            "seed": 42,
            "description": "Manual control over the stratified sampling settings."
        },
    }


def _apply_strategy_preset_if_requested(df_len: int, selected_preset: str) -> None:
    presets = _strategy_presets(df_len)
    preset = presets.get(selected_preset)
    if not preset:
        return

    last_applied = st.session_state.get("_validate_strategy_last_preset_applied")
    if last_applied == selected_preset:
        return

    st.session_state["validate_strategy_goal"] = str(preset["goal"])
    st.session_state["validate_strategy_balance"] = str(preset["balance"])
    st.session_state["validate_strategy_target_mode"] = str(preset["target_mode"])
    st.session_state["validate_strategy_target_value"] = int(preset["target_value"])
    st.session_state["validate_strategy_bins"] = int(preset["bins"])
    st.session_state["validate_strategy_seed"] = int(preset["seed"])
    st.session_state["_validate_strategy_last_preset_applied"] = selected_preset


def _effective_strategy_settings(df_len: int, df: Optional[pd.DataFrame] = None) -> Tuple[str, str, str, int, int, int]:
    goal = str(st.session_state.get("validate_strategy_goal", "representative_sample"))
    allowed_balance = _strategy_balance_options(df, goal) if df is not None else {"all": "All clips"}

    balance = str(st.session_state.get("validate_strategy_balance", "all"))
    if balance not in allowed_balance:
        balance = next(iter(allowed_balance.keys()))

    if goal == "equal_allocation":
        balance = "all"

    target_mode = str(st.session_state.get("validate_strategy_target_mode", "total_clips"))
    target_value = int(st.session_state.get("validate_strategy_target_value", 1))
    bins = int(st.session_state.get("validate_strategy_bins", 5))
    seed = int(st.session_state.get("validate_strategy_seed", 42))

    default_mode, default_value = _strategy_defaults_for_goal(goal, df_len)

    if goal == "custom_stratified":
        if target_mode not in ("total_clips", "per_group_clips", "per_group_percent"):
            target_mode = default_mode
    else:
        target_mode = "total_clips"

    if target_mode == "per_group_percent":
        if not (1 <= target_value <= 100):
            target_value = default_value if default_mode == "per_group_percent" else 10
        target_value = int(max(1, min(target_value, 100)))
    else:
        if target_value <= 0:
            target_value = default_value
        target_value = int(max(1, min(target_value, max(1, df_len))))

    bins = int(max(2, min(bins, 20)))
    return goal, balance, target_mode, target_value, bins, seed


def _strategy_summary(df: pd.DataFrame) -> str:
    goal, balance, target_mode, target_value, bins, _ = _effective_strategy_settings(len(df), df)
    goal_text = _strategy_goal_label(goal)
    balance_text = _strategy_balance_label(balance, df, goal)
    target_text = _strategy_target_summary(target_value, target_mode)
    if goal == "equal_allocation":
        return f"{goal_text} across confidence bands • {target_text} • {bins} bands"
    if "confidence" in balance:
        return f"{goal_text} across {balance_text} • {target_text} • {bins} bands"
    return f"{goal_text} across {balance_text} • {target_text}"




def _validation_card_group_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    if "basename" in df.columns and "species_display_original" in df.columns:
        return int(df[["basename", "species_display_original"]].astype(str).drop_duplicates().shape[0])
    if "detection_id" in df.columns:
        return int(df["detection_id"].astype(str).nunique())
    return int(len(df))


def _strategy_export_summary_df(df: pd.DataFrame, proj_root: Path, user_name: str) -> pd.DataFrame:
    goal, balance, target_mode, target_value, bins, seed = _effective_strategy_settings(len(df), df)
    selected_df, meta = _compute_strategy_plan(df.copy(), goal, balance, target_mode, target_value, bins, seed)

    stored_available = st.session_state.get("validate_strategy_available")
    stored_selected = st.session_state.get("validate_strategy_selected")
    stored_strata = st.session_state.get("validate_strategy_strata")
    stored_undersized = st.session_state.get("validate_strategy_undersized")
    metrics_source = st.session_state.get("validate_strategy_metrics_source")

    preview_metrics = {
        "available": int(len(df)),
        "selected": int(len(selected_df)),
        "strata": int(len(meta)),
        "undersized": _strategy_shortfall_count(
            df_scope=df,
            df_selected=selected_df,
            goal=goal,
            balance=balance,
            target_mode=target_mode,
            target_value=target_value,
        ),
    }

    use_stored_metrics = metrics_source == "wizard_preview_metrics"

    available_count = int(stored_available) if use_stored_metrics and stored_available not in (None, "") else int(preview_metrics["available"])
    selected_count = int(stored_selected) if use_stored_metrics and stored_selected not in (None, "") else int(preview_metrics["selected"])
    group_count = int(stored_strata) if use_stored_metrics and stored_strata not in (None, "") else int(preview_metrics["strata"])
    below_target = int(stored_undersized) if use_stored_metrics and stored_undersized not in (None, "") else int(preview_metrics["undersized"])

    rows = [
        ("Project", proj_root.name),
        ("Reviewer", str(user_name or "")),
        ("Selected strategy", _strategy_goal_label(goal)),
        ("Balance across", _strategy_balance_label(balance, df, goal)),
        ("Selection target", _strategy_target_summary(target_value, target_mode)),
    ]

    if goal == "equal_allocation" or "confidence" in str(balance):
        rows.append(("Confidence bands", int(bins)))

    rows.extend([
        ("Random seed", int(seed)),
        ("Available detections", available_count),
        ("Selected for review", selected_count),
        ("Groups / strata", group_count),
        ("Groups below requested target", below_target),
    ])

    rows.extend([
        ("Summary", _strategy_summary(df)),
        ("Exported at", _now_iso()),
    ])
    return pd.DataFrame(rows, columns=["field", "value"])


def _validated_workbook_bytes(export_df: pd.DataFrame, strategy_df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="validated_detections")
        strategy_df.to_excel(writer, index=False, sheet_name="validation_strategy")
        try:
            for ws in writer.book.worksheets:
                ws.freeze_panes = "A2"
                for col_cells in ws.columns:
                    max_len = 0
                    for cell in col_cells[:200]:
                        value = "" if cell.value is None else str(cell.value)
                        max_len = max(max_len, len(value))
                    ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 10), 48)
        except Exception:
            pass
    bio.seek(0)
    return bio.getvalue()


def _strategy_review_summary_text(
    df: pd.DataFrame,
    goal: str,
    balance: str,
    target_mode: str,
    target_value: int,
    bins: int,
) -> str:
    balance_text = _strategy_balance_label(balance, df, goal).lower()
    target_text = _strategy_target_summary(target_value, target_mode)

    if goal == "find_likely_mistakes":
        if balance == "all":
            return f"Review the lowest-confidence clips only. Target {target_text} from the filtered pool."
        return f"Review the lowest-confidence clips within each {balance_text} group. Target {target_text}."

    if goal == "review_strongest":
        if balance == "all":
            return f"Review the highest-confidence clips only. Target {target_text} from the filtered pool."
        return f"Review the highest-confidence clips within each {balance_text} group. Target {target_text}."

    if goal == "equal_allocation":
        return f"Review {target_text} with equal allocation across the {bins} confidence bands only. If one band cannot fill its share, the remainder is topped up from the other bands."

    if goal == "custom_stratified":
        if "confidence" in balance:
            return f"Review a custom stratified sample across {balance_text}. Target {target_text} using {bins} confidence bands. Sparse bands will be topped up from neighbouring bands of the same parent group where possible."
        return f"Review a custom stratified sample across {balance_text}. Target {target_text}."

    if "confidence" in balance:
        return f"Review a representative random sample across {balance_text}. Target {target_text} using {bins} confidence bands. Sparse bands will be topped up from neighbouring bands of the same parent group where possible."

    return f"Review a representative random sample across {balance_text}. Target {target_text}."


def _strategy_shortfall_count(
    df_scope: pd.DataFrame,
    df_selected: pd.DataFrame,
    goal: str,
    balance: str,
    target_mode: str,
    target_value: int,
) -> int:
    if df_scope.empty:
        return 0

    if balance == "all":
        desired_total = _desired_total_from_settings(df_scope, goal, target_mode, target_value)
        if desired_total < 0:
            return 0
        return int(len(df_selected) < desired_total)

    parent_scope = _strategy_group_series(df_scope, balance).astype(str)
    parent_selected = _strategy_group_series(df_selected, balance).astype(str)

    available = parent_scope.value_counts(dropna=False).sort_index()
    selected = parent_selected.value_counts(dropna=False).reindex(available.index).fillna(0).astype(int)

    if target_mode == "per_group_clips":
        requested = pd.Series(int(target_value), index=available.index, dtype=int)
    elif target_mode == "per_group_percent":
        pct = max(0.0, min(float(target_value), 100.0))
        requested = np.ceil(available * (pct / 100.0)).astype(int)
        requested = pd.Series(requested, index=available.index, dtype=int)
    else:
        requested = (
            _parent_target_counts(available, target_mode, target_value)
            .reindex(available.index)
            .fillna(0)
            .astype(int)
        )

    shortfall = selected < requested
    return int(shortfall.sum())


def _confidence_band_edges(n_bins: int) -> np.ndarray:
    n_bins = max(1, int(n_bins))
    return np.linspace(0.0, 1.0, n_bins + 1)


def _confidence_band_labels(n_bins: int) -> List[str]:
    edges = _confidence_band_edges(n_bins)
    return [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges) - 1)]


def _make_probability_bins(df: pd.DataFrame, n_bins: int) -> pd.Series:
    probs = pd.to_numeric(df.get("detection_probability"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    n_bins = max(1, int(n_bins))
    edges = _confidence_band_edges(n_bins)
    b = pd.cut(probs, bins=edges, labels=False, include_lowest=True)
    return b.fillna(0).astype(int)


def _priority_series(df: pd.DataFrame, goal: str, seed: int) -> pd.Series:
    probs = pd.to_numeric(df.get("detection_probability"), errors="coerce").fillna(0.0)
    if goal == "find_likely_mistakes":
        return probs
    if goal == "review_strongest":
        return -probs
    rng = np.random.default_rng(int(seed))
    return pd.Series(rng.random(len(df)), index=df.index)


def _build_strategy_strata(df_in: pd.DataFrame, balance: str, n_bins: int) -> pd.DataFrame:
    df = df_in.copy()
    df["__strategy_parent"] = "all"
    df["__strategy_bin"] = 0
    df["__strategy_stratum"] = "all"

    if balance == "all":
        return df

    parent = _strategy_group_series(df, balance)
    df["__strategy_parent"] = parent.astype(str)

    if "confidence" in balance:
        df["__strategy_bin"] = _make_probability_bins(df, n_bins)
        df["__strategy_stratum"] = (
            df["__strategy_parent"].astype(str)
            + "||bin="
            + df["__strategy_bin"].astype(int).astype(str)
        )
    else:
        df["__strategy_stratum"] = df["__strategy_parent"].astype(str)

    return df


def _allocate_even_targets(meta: pd.DataFrame, total: int) -> pd.Series:
    if meta.empty or total <= 0:
        return pd.Series(dtype=int)

    k = len(meta)
    base = total // k
    remainder = total % k

    order = (
        meta.assign(__stratum_key=meta.index.astype(str))
        .sort_values(["available", "__stratum_key"], ascending=[False, True])
        .index.tolist()
    )
    targets = pd.Series(base, index=meta.index, dtype=int)
    for idx in order[:remainder]:
        targets.loc[idx] += 1
    return targets


def _allocate_even_with_caps(available: pd.Series, total: int) -> pd.Series:
    available = available.astype(int)
    total = int(max(0, min(total, int(available.sum()))))
    out = pd.Series(0, index=available.index, dtype=int)
    if total <= 0 or available.empty:
        return out

    base = total // len(available)
    out[:] = np.minimum(available.values, base)

    remaining = total - int(out.sum())
    if remaining <= 0:
        return out

    order = available.sort_values(ascending=False).index.tolist()
    while remaining > 0:
        moved = False
        for idx in order:
            if remaining <= 0:
                break
            if out.loc[idx] < available.loc[idx]:
                out.loc[idx] += 1
                remaining -= 1
                moved = True
        if not moved:
            break

    return out


def _allocate_weighted_bin_targets(available: pd.Series, total: int, weights: np.ndarray) -> pd.Series:
    available = available.astype(int)
    total = int(max(0, min(total, int(available.sum()))))
    out = pd.Series(0, index=available.index, dtype=int)
    if total <= 0 or available.empty:
        return out

    w = np.asarray(weights, dtype=float)
    if len(w) < len(available):
        w = np.pad(w, (0, len(available) - len(w)), constant_values=1.0)
    if len(w) > len(available):
        w = w[:len(available)]

    active = np.where(available.values > 0, w, 0.0)
    if np.sum(active) <= 0:
        active = np.where(available.values > 0, 1.0, 0.0)

    raw = total * (active / np.sum(active))
    base = np.floor(raw).astype(int)
    base = np.minimum(base, available.values)
    out.iloc[:] = base

    remaining = total - int(out.sum())
    if remaining <= 0:
        return out

    while remaining > 0:
        capacity_mask = available.values > out.values
        if not capacity_mask.any():
            break

        residual = raw - out.values
        candidate_idx = np.where(capacity_mask)[0]
        order = sorted(
            candidate_idx.tolist(),
            key=lambda i: (residual[i], active[i], -i),
            reverse=True,
        )

        moved = False
        for i in order:
            if remaining <= 0:
                break
            if available.iloc[i] > out.iloc[i]:
                out.iloc[i] += 1
                remaining -= 1
                moved = True
        if not moved:
            break

    return out


def _parent_target_counts(parent_available: pd.Series, target_mode: str, target_value: int) -> pd.Series:
    parent_available = parent_available.astype(int)

    if target_mode == "per_group_clips":
        return np.minimum(parent_available, int(max(0, target_value))).astype(int)

    if target_mode == "per_group_percent":
        pct = max(0.0, min(float(target_value), 100.0))
        vals = np.ceil(parent_available * (pct / 100.0)).astype(int)
        return np.minimum(vals, parent_available).astype(int)

    total = int(max(1, min(int(target_value), int(parent_available.sum()))))
    meta = pd.DataFrame({"available": parent_available}, index=parent_available.index)
    return _allocate_even_targets(meta, total).reindex(parent_available.index).fillna(0).astype(int)


def _apply_local_refill(meta: pd.DataFrame, shortfalls: Dict[str, int]) -> Tuple[pd.DataFrame, int]:
    leftover = 0
    if not shortfalls:
        return meta, leftover

    for stratum, short in shortfalls.items():
        if short <= 0 or stratum not in meta.index:
            continue

        parent = meta.at[stratum, "parent"]
        bin_id = int(meta.at[stratum, "bin"])
        sib = meta[
            (meta["parent"] == parent)
            & (meta.index != stratum)
            & (meta["remaining"] > 0)
        ].copy()

        if sib.empty:
            leftover += int(short)
            continue

        sib["distance"] = (sib["bin"] - bin_id).abs()
        sib = (
            sib.assign(__stratum_key=sib.index.astype(str))
            .sort_values(["distance", "remaining", "__stratum_key"], ascending=[True, False, True])
        )

        need = int(short)
        for sib_stratum, _ in sib.iterrows():
            if need <= 0:
                break
            take = int(min(need, int(meta.at[sib_stratum, "remaining"])))
            if take <= 0:
                continue
            meta.at[sib_stratum, "selected"] += take
            meta.at[sib_stratum, "remaining"] -= take
            need -= take

        if need > 0:
            leftover += int(need)

    return meta, leftover


def _apply_global_refill(meta: pd.DataFrame, leftover: int) -> pd.DataFrame:
    if leftover <= 0:
        return meta

    pool = meta[meta["remaining"] > 0].copy()
    if pool.empty:
        return meta

    pool = (
        pool.assign(__stratum_key=pool.index.astype(str))
        .sort_values(["remaining", "__stratum_key"], ascending=[False, True])
    )

    need = int(leftover)
    for stratum, _ in pool.iterrows():
        if need <= 0:
            break
        take = int(min(need, int(meta.at[stratum, "remaining"])))
        if take <= 0:
            continue
        meta.at[stratum, "selected"] += take
        meta.at[stratum, "remaining"] -= take
        need -= take

    return meta


def _desired_total_from_settings(
    df: pd.DataFrame,
    goal: str,
    target_mode: str,
    target_value: int,
) -> int:
    if df.empty:
        return 0

    if target_mode == "total_clips":
        return int(max(0, min(int(target_value), len(df))))

    if target_mode == "per_group_percent" and goal == "custom_stratified":
        return -1

    if target_mode == "per_group_clips" and goal == "custom_stratified":
        return -1

    return int(max(0, min(int(target_value), len(df))))


def _enforce_final_selection_total(
    df_source: pd.DataFrame,
    df_selected: pd.DataFrame,
    goal: str,
    desired_total: int,
    seed: int,
) -> pd.DataFrame:
    desired_total = int(max(0, min(desired_total, len(df_source))))
    if desired_total <= 0:
        return df_source.head(0).copy()

    if len(df_selected) >= desired_total:
        if goal == "find_likely_mistakes":
            return (
                df_selected
                .sort_values(["detection_probability", "basename", "start_s"], ascending=[True, True, True])
                .head(desired_total)
                .copy()
            )
        if goal == "review_strongest":
            return (
                df_selected
                .sort_values(["detection_probability", "basename", "start_s"], ascending=[False, True, True])
                .head(desired_total)
                .copy()
            )

        pr = _priority_series(df_selected, goal, seed)
        return (
            df_selected
            .assign(__tmp_priority=pr)
            .sort_values(["__tmp_priority", "basename", "start_s"], ascending=True)
            .head(desired_total)
            .drop(columns="__tmp_priority", errors="ignore")
            .copy()
        )

    if "detection_id" in df_source.columns and "detection_id" in df_selected.columns:
        selected_ids = set(df_selected["detection_id"].astype(str))
        pool = df_source[~df_source["detection_id"].astype(str).isin(selected_ids)].copy()
    else:
        pool = df_source.drop(index=df_selected.index, errors="ignore").copy()

    shortfall = desired_total - len(df_selected)
    if shortfall <= 0 or pool.empty:
        return df_selected.copy()

    if goal == "find_likely_mistakes":
        refill = (
            pool
            .sort_values(["detection_probability", "basename", "start_s"], ascending=[True, True, True])
            .head(shortfall)
            .copy()
        )
        out = pd.concat([df_selected, refill], axis=0)
        return (
            out
            .sort_values(["detection_probability", "basename", "start_s"], ascending=[True, True, True])
            .head(desired_total)
            .copy()
        )

    if goal == "review_strongest":
        refill = (
            pool
            .sort_values(["detection_probability", "basename", "start_s"], ascending=[False, True, True])
            .head(shortfall)
            .copy()
        )
        out = pd.concat([df_selected, refill], axis=0)
        return (
            out
            .sort_values(["detection_probability", "basename", "start_s"], ascending=[False, True, True])
            .head(desired_total)
            .copy()
        )

    pr_pool = _priority_series(pool, goal, seed)
    refill = (
        pool
        .assign(__tmp_priority=pr_pool)
        .sort_values(["__tmp_priority", "basename", "start_s"], ascending=True)
        .head(shortfall)
        .drop(columns="__tmp_priority", errors="ignore")
        .copy()
    )

    out = pd.concat([df_selected, refill], axis=0)
    pr_out = _priority_series(out, goal, seed)
    return (
        out
        .assign(__tmp_priority=pr_out)
        .sort_values(["__tmp_priority", "basename", "start_s"], ascending=True)
        .head(desired_total)
        .drop(columns="__tmp_priority", errors="ignore")
        .copy()
    )


def _compute_strategy_plan(
    df_in: pd.DataFrame,
    goal: str,
    balance: str,
    target_mode: str,
    target_value: int,
    n_bins: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df_in.copy()
    if df.empty:
        return df, pd.DataFrame()

    desired_total = _desired_total_from_settings(df, goal, target_mode, target_value)

    if goal == "equal_allocation":
        work = df.copy()
        work["__strategy_bin"] = _make_probability_bins(work, n_bins)

        band_available = work.groupby("__strategy_bin", dropna=False).size().reindex(range(n_bins), fill_value=0)
        band_targets = _allocate_even_with_caps(band_available, int(max(1, min(int(target_value), len(work)))))

        meta = pd.DataFrame({
            "available": band_available.astype(int),
            "target": band_targets.astype(int),
        })
        meta["selected"] = np.minimum(meta["available"], meta["target"]).astype(int)
        meta["remaining"] = (meta["available"] - meta["selected"]).astype(int)
        meta["parent"] = "all"
        meta["bin"] = meta.index.astype(int)
        meta["stratum"] = meta.index.astype(str)

        shortfalls: Dict[str, int] = {}
        for band_id, row in meta.iterrows():
            short = int(row["target"] - row["selected"])
            if short > 0:
                shortfalls[str(band_id)] = short

        if shortfalls:
            pool = meta[meta["remaining"] > 0].copy()
            if not pool.empty:
                need = sum(shortfalls.values())
                order = pool.sort_values(["remaining"], ascending=[False]).index.tolist()
                for idx in order:
                    if need <= 0:
                        break
                    take = int(min(need, int(meta.at[idx, "remaining"])))
                    if take <= 0:
                        continue
                    meta.at[idx, "selected"] += take
                    meta.at[idx, "remaining"] -= take
                    need -= take

        chosen_parts: List[pd.DataFrame] = []
        for band_id, g in work.groupby("__strategy_bin", dropna=False):
            take_n = int(meta.at[int(band_id), "selected"]) if int(band_id) in meta.index else 0
            if take_n <= 0:
                continue
            pr = _priority_series(g, goal, seed)
            g2 = (
                g.assign(__strategy_priority=pr)
                .sort_values(["__strategy_priority", "basename", "start_s"], ascending=True)
                .head(take_n)
                .drop(columns="__strategy_priority", errors="ignore")
                .copy()
            )
            chosen_parts.append(g2)

        out = pd.concat(chosen_parts, axis=0) if chosen_parts else work.head(0).copy()
        out = _enforce_final_selection_total(
            work.drop(columns="__strategy_bin", errors="ignore"),
            out.drop(columns="__strategy_bin", errors="ignore"),
            goal,
            int(max(1, min(int(target_value), len(work)))),
            seed,
        )
        return out, meta

    if goal in ("find_likely_mistakes", "review_strongest"):
        if balance == "all":
            total = int(max(1, min(int(target_value), len(df))))
            pr = _priority_series(df, goal, seed)
            out = (
                df.assign(__strategy_priority=pr)
                .sort_values(["__strategy_priority", "basename", "start_s"], ascending=True)
                .head(total)
                .drop(columns="__strategy_priority", errors="ignore")
                .copy()
            )
            meta = pd.DataFrame({
                "available": [len(df)],
                "selected": [len(out)],
                "target": [total],
                "remaining": [max(0, len(df) - len(out))],
                "parent": ["all"],
                "bin": [0],
                "stratum": ["all"],
            }, index=["all"])
            return out, meta

        parent = _strategy_group_series(df, balance)
        df["__strategy_parent"] = parent.astype(str)
        meta = (
            df.groupby("__strategy_parent", dropna=False)
            .agg(available=("__strategy_parent", "size"))
        ).copy()
        meta["parent"] = meta.index.astype(str)
        meta["bin"] = 0
        meta["stratum"] = meta.index.astype(str)

        parent_targets = _parent_target_counts(meta["available"], "total_clips", target_value)
        meta["target"] = parent_targets.reindex(meta.index).fillna(0).astype(int)
        meta["selected"] = np.minimum(meta["target"], meta["available"]).astype(int)
        meta["remaining"] = (meta["available"] - meta["selected"]).astype(int)

        chosen_parts: List[pd.DataFrame] = []
        for parent_name, g in df.groupby("__strategy_parent", dropna=False):
            take_n = int(meta.at[parent_name, "selected"]) if parent_name in meta.index else 0
            if take_n <= 0:
                continue
            pr = _priority_series(g, goal, seed)
            g2 = (
                g.assign(__strategy_priority=pr)
                .sort_values(["__strategy_priority", "basename", "start_s"], ascending=True)
                .head(take_n)
                .drop(columns="__strategy_priority", errors="ignore")
                .copy()
            )
            chosen_parts.append(g2)

        out = pd.concat(chosen_parts, axis=0) if chosen_parts else df.head(0).copy()
        if desired_total >= 0:
            out = _enforce_final_selection_total(df.drop(columns="__strategy_parent", errors="ignore"), out, goal, desired_total, seed)

        if goal == "find_likely_mistakes":
            out = out.sort_values(["detection_probability", "basename", "start_s"], ascending=[True, True, True])
        else:
            out = out.sort_values(["detection_probability", "basename", "start_s"], ascending=[False, True, True])

        return out, meta

    df = _build_strategy_strata(df, balance, n_bins)
    df["__strategy_priority"] = _priority_series(df, goal, seed)

    meta = (
        df.groupby("__strategy_stratum", dropna=False)
        .agg(
            available=("__strategy_stratum", "size"),
            parent=("__strategy_parent", "first"),
            bin=("__strategy_bin", "first"),
        )
    ).copy()
    meta["available"] = meta["available"].astype(int)
    meta["bin"] = pd.to_numeric(meta["bin"], errors="coerce").fillna(0).astype(int)
    meta["stratum"] = meta.index.astype(str)

    if balance == "all":
        total = int(max(1, min(int(target_value), len(df))))
        meta["target"] = _allocate_even_with_caps(meta["available"], total).reindex(meta.index).fillna(0).astype(int)
    else:
        parent_available = meta.groupby("parent", dropna=False)["available"].sum()
        parent_targets = _parent_target_counts(parent_available, target_mode, target_value)

        if "confidence" in balance:
            meta["target"] = 0
            for parent_name, parent_target in parent_targets.items():
                parent_rows = meta[meta["parent"] == parent_name].sort_values("bin")
                if parent_rows.empty:
                    continue
                weights = np.ones(n_bins, dtype=float)
                bin_targets = _allocate_weighted_bin_targets(parent_rows["available"], int(parent_target), weights)
                meta.loc[parent_rows.index, "target"] = (
                    bin_targets.reindex(parent_rows.index).fillna(0).astype(int)
                )
        else:
            meta["target"] = 0
            for parent_name, parent_target in parent_targets.items():
                parent_rows = meta[meta["parent"] == parent_name]
                if parent_rows.empty:
                    continue
                if len(parent_rows) == 1:
                    meta.loc[parent_rows.index, "target"] = int(min(parent_target, int(parent_rows["available"].iloc[0])))
                else:
                    alloc = _allocate_even_targets(parent_rows[["available"]], int(parent_target))
                    meta.loc[parent_rows.index, "target"] = alloc.reindex(parent_rows.index).fillna(0).astype(int)

    meta["selected"] = np.minimum(meta["target"], meta["available"]).astype(int)
    meta["remaining"] = (meta["available"] - meta["selected"]).astype(int)

    shortfalls: Dict[str, int] = {}
    for stratum, row in meta.iterrows():
        short = int(row["target"] - row["selected"])
        if short > 0:
            shortfalls[stratum] = short

    if "confidence" in balance:
        meta, leftover = _apply_local_refill(meta, shortfalls)
    else:
        leftover = sum(shortfalls.values())

    meta = _apply_global_refill(meta, leftover)

    chosen_parts: List[pd.DataFrame] = []
    for stratum, g in df.groupby("__strategy_stratum", dropna=False):
        take_n = int(meta.at[stratum, "selected"]) if stratum in meta.index else 0
        if take_n <= 0:
            continue
        g2 = (
            g.sort_values(["__strategy_priority", "basename", "start_s"], ascending=True)
            .head(take_n)
            .copy()
        )
        chosen_parts.append(g2)

    out = pd.concat(chosen_parts, axis=0) if chosen_parts else df.head(0).copy()

    if desired_total >= 0:
        out_clean = out.drop(columns=["__strategy_parent", "__strategy_bin", "__strategy_stratum", "__strategy_priority"], errors="ignore")
        df_clean = df.drop(columns=["__strategy_parent", "__strategy_bin", "__strategy_stratum", "__strategy_priority"], errors="ignore")
        out = _enforce_final_selection_total(df_clean, out_clean, goal, desired_total, seed)
    else:
        out = out.drop(columns=["__strategy_parent", "__strategy_bin", "__strategy_stratum", "__strategy_priority"], errors="ignore")

    if goal == "find_likely_mistakes":
        out = out.sort_values(["detection_probability", "basename", "start_s"], ascending=[True, True, True])
    elif goal == "review_strongest":
        out = out.sort_values(["detection_probability", "basename", "start_s"], ascending=[False, True, True])
    else:
        pr_out = _priority_series(out, goal, seed)
        out = (
            out.assign(__tmp_priority=pr_out)
            .sort_values(["__tmp_priority", "basename", "start_s"], ascending=True)
            .drop(columns="__tmp_priority", errors="ignore")
        )

    return out, meta


def _select_by_strategy(df_in: pd.DataFrame, df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    goal, balance, target_mode, target_value, bins, seed = _effective_strategy_settings(len(df_in), df_all)
    return _compute_strategy_plan(df_in, goal, balance, target_mode, target_value, bins, seed)


def _finalise_preview_table(preview: pd.DataFrame, fallback_label: str = "[unknown]", drop_zero_rows: bool = True) -> pd.DataFrame:
    if preview.empty:
        return preview

    out = preview.copy()
    out.index = _clean_index_labels(out.index, fallback=fallback_label)

    if drop_zero_rows and not out.empty:
        keep_mask = pd.Series(False, index=out.index)

        for c in out.columns:
            if pd.api.types.is_numeric_dtype(out[c]):
                keep_mask = keep_mask | (pd.to_numeric(out[c], errors="coerce").fillna(0) > 0)
            else:
                vals = out[c].astype(str).fillna("").str.strip()
                keep_mask = keep_mask | vals.ne("") | vals.str.contains("/", regex=False)

        if keep_mask.any():
            out = out.loc[keep_mask.values]

    return out

def _strategy_preview_matrix(
    df_in: pd.DataFrame,
    goal: str,
    balance: str,
    target_mode: str,
    target_value: int,
    n_bins: int,
    seed: int,
    max_rows: int = 12,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    selected_df, meta = _compute_strategy_plan(df_in, goal, balance, target_mode, target_value, n_bins, seed)

    metrics = {
        "available": int(len(df_in)),
        "selected": int(len(selected_df)),
        "strata": int(len(meta)),
        "undersized": _strategy_shortfall_count(
            df_scope=df_in,
            df_selected=selected_df,
            goal=goal,
            balance=balance,
            target_mode=target_mode,
            target_value=target_value,
        ),
    }

    if df_in.empty:
        return pd.DataFrame(), metrics

    if goal == "equal_allocation":
        work = df_in.copy()
        sel_work = selected_df.copy()

        work["__bin"] = _make_probability_bins(work, n_bins)
        sel_work["__bin"] = _make_probability_bins(sel_work, n_bins)

        species_col = "species_display_original" if "species_display_original" in work.columns else None

        labels = _confidence_band_labels(n_bins)
        band_cols = list(range(max(1, int(n_bins))))

        avail_all = work.groupby("__bin", dropna=False).size().reindex(band_cols, fill_value=0)
        sel_all = sel_work.groupby("__bin", dropna=False).size().reindex(band_cols, fill_value=0)

        rows = []
        all_row = {}
        for i, lab in enumerate(labels):
            all_row[lab] = f"{int(sel_all.get(i, 0))}/{int(avail_all.get(i, 0))}"
        all_row["Total"] = f"{int(len(selected_df))}/{int(len(df_in))}"
        rows.append(("All clips", all_row))

        if species_col is not None:
            work_sp = work.copy()
            sel_sp = sel_work.copy()

            work_sp["__species"] = _clean_group_labels(work_sp[species_col], "[unknown species]")
            sel_sp["__species"] = _clean_group_labels(sel_sp[species_col], "[unknown species]")

            avail_sp = (
                work_sp.groupby(["__species", "__bin"], dropna=False)
                .size()
                .unstack(fill_value=0)
                .reindex(columns=band_cols, fill_value=0)
            )
            sel_sp_tab = (
                sel_sp.groupby(["__species", "__bin"], dropna=False)
                .size()
                .unstack(fill_value=0)
                .reindex(index=avail_sp.index, columns=band_cols, fill_value=0)
            )

            row_available = avail_sp.sum(axis=1)
            keep = row_available > 0
            avail_sp = avail_sp.loc[keep]
            sel_sp_tab = sel_sp_tab.loc[keep]

            row_meta = pd.DataFrame({
                "selected": sel_sp_tab.sum(axis=1),
                "available": avail_sp.sum(axis=1),
                "label": avail_sp.index.astype(str),
            }, index=avail_sp.index)

            species_order = row_meta.sort_values(
                ["selected", "available", "label"],
                ascending=[False, False, True]
            ).index.tolist()

            for sp in species_order:
                row = {}
                for i, lab in enumerate(labels):
                    row[lab] = f"{int(sel_sp_tab.loc[sp, i])}/{int(avail_sp.loc[sp, i])}"
                row["Total"] = f"{int(sel_sp_tab.loc[sp].sum())}/{int(avail_sp.loc[sp].sum())}"
                rows.append((sp, row))

        preview = pd.DataFrame(
            [r[1] for r in rows],
            index=[r[0] for r in rows],
        )
        preview.index.name = "Selection"
        return _finalise_preview_table(preview, fallback_label="[unknown selection]"), metrics

    if balance == "all" and goal in ("find_likely_mistakes", "review_strongest"):
        work = df_in.copy()
        sel_work = selected_df.copy()
        work["__bin"] = _make_probability_bins(work, n_bins)
        sel_work["__bin"] = _make_probability_bins(sel_work, n_bins)

        avail = work.groupby("__bin", dropna=False).size()
        sel = sel_work.groupby("__bin", dropna=False).size()

        labels = _confidence_band_labels(n_bins)
        preview = pd.DataFrame(index=["All clips"])
        for i, lab in enumerate(labels):
            preview[lab] = [f"{int(sel.get(i, 0))}/{int(avail.get(i, 0))}"]
        preview["Total"] = [f"{len(selected_df)}/{len(df_in)}"]
        preview.index.name = "Selection"
        return _finalise_preview_table(preview, fallback_label="All clips"), metrics

    if goal in ("find_likely_mistakes", "review_strongest") and balance != "all":
        work = df_in.copy()
        sel_work = selected_df.copy()

        work["__parent"] = _strategy_group_series(work, balance).astype(str)
        sel_work["__parent"] = _strategy_group_series(sel_work, balance).astype(str)
        work["__bin"] = _make_probability_bins(work, n_bins)
        sel_work["__bin"] = _make_probability_bins(sel_work, n_bins)

        avail = (
            work.groupby(["__parent", "__bin"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )
        sel = (
            sel_work.groupby(["__parent", "__bin"], dropna=False)
            .size()
            .unstack(fill_value=0)
        )

        band_cols = list(range(max(1, int(n_bins))))
        avail = avail.reindex(columns=band_cols, fill_value=0)
        sel = sel.reindex(index=avail.index, columns=band_cols, fill_value=0)

        row_available = avail.sum(axis=1)
        keep = row_available > 0
        avail = avail.loc[keep]
        sel = sel.loc[keep]

        row_meta = pd.DataFrame({
            "selected": sel.sum(axis=1),
            "available": avail.sum(axis=1),
            "label": avail.index.astype(str),
        }, index=avail.index)

        row_order = row_meta.sort_values(
            ["selected", "available", "label"],
            ascending=[False, False, True]
        ).index.tolist()

        avail = avail.reindex(row_order)
        sel = sel.reindex(row_order)

        labels = _confidence_band_labels(n_bins)
        preview = pd.DataFrame(index=avail.index)
        for i, lab in enumerate(labels):
            preview[lab] = [
                f"{int(sel.loc[parent_name, i])}/{int(avail.loc[parent_name, i])}"
                for parent_name in avail.index
            ]
        sel_total = sel.sum(axis=1)
        avail_total = avail.sum(axis=1)
        preview["Total"] = [
            f"{int(sel_total.loc[parent_name])}/{int(avail_total.loc[parent_name])}"
            for parent_name in avail.index
        ]
        preview.index.name = _strategy_parent_label(balance)
        return _finalise_preview_table(preview, fallback_label=f"[unknown {_strategy_parent_label(balance).lower()}]"), metrics

    if balance == "all":
        preview = pd.DataFrame({
            "available": [len(df_in)],
            "selected": [len(selected_df)],
            "selected %": [round(100.0 * len(selected_df) / max(1, len(df_in)), 1)],
        }, index=["All clips"])
        preview.index.name = "Selection"
        return _finalise_preview_table(preview, fallback_label="All clips"), metrics

    if "confidence" not in balance:
        parent = _strategy_group_series(df_in, balance).astype(str)
        avail = parent.value_counts(dropna=False)
        sel_parent = _strategy_group_series(selected_df, balance).astype(str)
        sel = sel_parent.value_counts(dropna=False)

        preview = pd.DataFrame({
            "available": avail,
            "selected": sel.reindex(avail.index).fillna(0).astype(int),
        })
        preview["selected %"] = np.where(
            preview["available"] > 0,
            (100.0 * preview["selected"] / preview["available"]).round(1),
            0.0,
        )
        preview = preview.loc[preview["available"] > 0]
        preview = preview.sort_values(["selected", "available"], ascending=[False, False])
        preview.index.name = _strategy_parent_label(balance)
        return _finalise_preview_table(preview, fallback_label=f"[unknown {_strategy_parent_label(balance).lower()}]"), metrics

    work = _build_strategy_strata(df_in.copy(), balance, n_bins)
    work["__parent"] = work["__strategy_parent"].astype(str)
    work["__bin"] = work["__strategy_bin"].astype(int)

    sel_work = _build_strategy_strata(selected_df.copy(), balance, n_bins)
    sel_work["__parent"] = sel_work["__strategy_parent"].astype(str)
    sel_work["__bin"] = sel_work["__strategy_bin"].astype(int)

    avail = (
        work.groupby(["__parent", "__bin"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )
    sel = (
        sel_work.groupby(["__parent", "__bin"], dropna=False)
        .size()
        .unstack(fill_value=0)
    )

    band_cols = list(range(max(1, int(n_bins))))
    avail = avail.reindex(columns=band_cols, fill_value=0)
    sel = sel.reindex(index=avail.index, columns=band_cols, fill_value=0)

    row_available = avail.sum(axis=1)
    keep = row_available > 0
    avail = avail.loc[keep]
    sel = sel.loc[keep]

    row_meta = pd.DataFrame({
        "selected": sel.sum(axis=1),
        "available": avail.sum(axis=1),
        "label": avail.index.astype(str),
    }, index=avail.index)

    row_order = row_meta.sort_values(
        ["selected", "available", "label"],
        ascending=[False, False, True]
    ).index.tolist()

    avail = avail.reindex(row_order)
    sel = sel.reindex(row_order)

    labels = _confidence_band_labels(n_bins)
    preview = pd.DataFrame(index=avail.index)
    for i, lab in enumerate(labels):
        preview[lab] = [
            f"{int(sel.loc[parent_name, i])}/{int(avail.loc[parent_name, i])}"
            for parent_name in avail.index
        ]

    sel_total = sel.sum(axis=1)
    avail_total = avail.sum(axis=1)
    preview["Total"] = [
        f"{int(sel_total.loc[parent_name])}/{int(avail_total.loc[parent_name])}"
        for parent_name in avail.index
    ]
    preview.index.name = _strategy_parent_label(balance)
    return _finalise_preview_table(preview, fallback_label=f"[unknown {_strategy_parent_label(balance).lower()}]"), metrics


def _preview_display_df(preview_df: pd.DataFrame) -> pd.DataFrame:
    if preview_df.empty:
        return preview_df
    out = preview_df.reset_index()
    if out.columns[0] == "index":
        out = out.rename(columns={"index": preview_df.index.name or "Group"})
    if out.columns.size > 0:
        first_col = out.columns[0]
        out[first_col] = (
            out[first_col]
            .astype(str)
            .replace({"nan": "", "None": "", "<NA>": ""})
            .fillna("")
            .str.strip()
            .replace({"": "[unknown]"})
        )
    return out


def _render_strategy_summary_bar(df: pd.DataFrame):
    summary = _strategy_summary(df)
    goal, balance, target_mode, target_value, _, _ = _effective_strategy_settings(len(df), df)
    goal_text = _strategy_goal_label(goal)
    balance_text = _strategy_balance_label(balance, df, goal)

    chips = [
        f"Strategy: {goal_text}",
        f"Across: {balance_text}",
        f"Target: {_strategy_target_summary(target_value, target_mode)}",
    ]
    if goal == "equal_allocation" or "confidence" in balance:
        chips.append(f"Bands: {int(st.session_state.get('validate_strategy_bins', 5))}")

    left, right = st.columns([4.5, 1.2])
    with left:
        st.markdown(
            f"""
            <div style="border:1px solid #e5e7eb; border-radius:1rem; padding:0.85rem 1rem; background:white;">
              <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#6b7280; margin-bottom:0.35rem;">
                Validation strategy
              </div>
              <div style="font-size:1rem; font-weight:600; color:#111827; margin-bottom:0.5rem;">
                {summary}
              </div>
              <div style="display:flex; gap:0.4rem; flex-wrap:wrap;">
                {''.join([f"<span style='padding:0.18rem 0.55rem; border-radius:999px; background:#f3f4f6; color:#374151; font-size:0.78rem;'>{chip}</span>" for chip in chips])}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    def _open_validate_strategy_modal():
        st.session_state["validate_strategy_modal_open"] = True

    with right:
        st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
        st.button(
            "Change strategy",
            key="open_validate_strategy_modal",
            width="stretch",
            on_click=_open_validate_strategy_modal,
        )


def _compact_preview_caption(goal: str, balance: str) -> str:
    if goal in ("find_likely_mistakes", "review_strongest"):
        if balance == "all":
            return "Preview by confidence band for the selected review set."
        return "Preview by group. Each row shows selected versus available."
    if goal == "equal_allocation":
        return "Preview by confidence band. Each cell shows selected versus available."
    if "confidence" in balance:
        return "Preview by group and confidence band. Each cell shows selected versus available."
    return "Preview by group. Values show selected clips out of the total available."


def _preview_height_for_rows(n_rows: int) -> int:
    base = 44
    per_row = 32
    return max(180, min(420, base + per_row * int(max(1, n_rows))))


def _commit_card(
    proj_root: Path,
    df_all: pd.DataFrame,
    base: str,
    species_orig: str,
    selected_indices: Optional[List[int]] = None,
    submitted_values: Optional[Dict[int, Dict[str, object]]] = None,
) -> Tuple[pd.DataFrame, int, int]:
    det = df_all.copy()

    det = _force_string_cols(det, [
        "species_name", "presence_label",
        "species_name_original", "presence_label_original",
        "validation_state", "validation_label", "validation_species",
        "validated_by", "validated_at", "validation_method",
        "validation_notes",
        "user_changed", "user_changed_by", "user_changed_at",
        "uncertain_flag",
    ])

    if submitted_values is not None:
        det = _apply_card_submitted_values(
            det,
            base,
            species_orig,
            submitted_values,
            selected_indices=selected_indices,
        )
    else:
        det = _apply_card_widget_state(det, base, species_orig, selected_indices=selected_indices)

    mask_card = (
        det["basename"].astype(str).eq(base)
        & det["species_display_original"].astype(str).eq(species_orig)
    )

    if selected_indices is not None:
        selected_idx_set = set(int(i) for i in selected_indices)
        mask_card = mask_card & det.index.to_series().isin(selected_idx_set)

    card_rows_updated = det.loc[mask_card].copy()
    if card_rows_updated.empty:
        return det, 0, 0

    card_rows_updated = card_rows_updated.sort_values("start_s")

    user_id = st.session_state.get("user_id") or st.session_state.get("username") or _user_name()
    now_iso = _now_iso()

    cur_sp = card_rows_updated["species_name"].astype(str)
    cur_pl = card_rows_updated["presence_label"].astype(str).str.lower()
    orig_sp = card_rows_updated["species_name_original"].astype(str)
    orig_pl = card_rows_updated["presence_label_original"].astype(str).str.lower()
    species_changed_mask = cur_sp != orig_sp
    species_presence_changed_mask = species_changed_mask | (cur_pl != orig_pl)
    changed_mask = species_presence_changed_mask

    for i, changed_here, species_changed_here in zip(card_rows_updated.index, changed_mask, species_changed_mask):
        current_sp = str(det.at[i, "species_name"] or "")
        current_pl = str(det.at[i, "presence_label"] or "").strip().lower()
        original_sp = str(det.at[i, "species_name_original"] or "")
        original_pl = str(det.at[i, "presence_label_original"] or "").strip().lower()

        if changed_here:
            det.at[i, "user_changed"] = user_id or "1"
            det.at[i, "user_changed_by"] = user_id
            det.at[i, "user_changed_at"] = now_iso

        det.at[i, "validation_state"] = "incorrect" if species_changed_here else "correct"

        det.at[i, "validated_by"] = user_id
        det.at[i, "validated_at"] = now_iso

        det.at[i, "validation_label"] = "present" if current_pl == "present" else "absent"
        if det.at[i, "validation_label"] == "present":
            det.at[i, "validation_species"] = current_sp.strip()
        else:
            det.at[i, "validation_species"] = ""

        if original_sp.strip() in ("", "<NA>", "nan"):
            det.at[i, "species_name_original"] = current_sp
        if original_pl.strip() in ("", "<NA>", "nan"):
            det.at[i, "presence_label_original"] = current_pl

    return det, int(changed_mask.sum()), int(len(card_rows_updated))


def _init_filter_state():
    defaults = {
        "validate_num_per_page": 10,
        "validate_cols_per_row": 2,
        "validate_page": 1,
        "validate_page_input": 1,
        "validate_page_sync_pending": False,
        "validate_show_label": "present",
        "validate_min_prob": 0.0,
        "validate_conf_sort": "Strategy/default order",
        "validate_lock_freq": False,
        "validate_fmin_khz": 15.0,
        "validate_fmax_khz": 90.0,
        "validate_use_te_override": False,
        "validate_te_override": 10,
        "validate_use_fft_override": False,
        "validate_fft_size": 4096,
        "validate_auto_zoom_single_detection": True,
        "validate_auto_zoom_window_s": 5.0,
        "validate_interactive_card": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _init_strategy_state():
    defaults = {
        "validate_strategy_goal": "representative_sample",
        "validate_strategy_balance": "species_confidence",
        "validate_strategy_target_mode": "total_clips",
        "validate_strategy_target_value": 200,
        "validate_strategy_bins": 5,
        "validate_strategy_seed": 42,
        "validate_strategy_modal_open": False,
        "validate_strategy_dont_auto_show": False,
        "validate_strategy_prompt_seen": False,
        "validate_strategy_preset_label": "Representative sample",
        "_validate_strategy_last_preset_applied": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _card_change_counts(gdf: pd.DataFrame) -> Tuple[int, int]:
    if gdf.empty:
        return 0, 0
    cur_sp = gdf.get("species_name", "").astype(str)
    cur_pl = gdf.get("presence_label", "").astype(str).str.lower()
    orig_sp = gdf.get("species_name_original", cur_sp).astype(str)
    orig_pl = gdf.get("presence_label_original", cur_pl).astype(str).str.lower()
    changed = (cur_sp != orig_sp) | (cur_pl != orig_pl)
    return int(changed.sum()), int(len(gdf))


def _card_uncertain_count(gdf: pd.DataFrame) -> int:
    if gdf.empty or "uncertain_flag" not in gdf.columns:
        return 0
    return int(gdf["uncertain_flag"].map(_bool_from_any).sum())


def _card_classifier_label_and_colour(changed: int, total: int, reviewed: bool) -> Tuple[str, str]:
    if total == 0:
        return "Classifier: not assessed", "#777777"
    if not reviewed:
        return "Classifier: not assessed", "#777777"
    if changed == 0:
        return "Classifier: all unchanged", "#2e7d32"
    if changed == total:
        return "Classifier: all changed", "#c62828"
    return "Classifier: mixed", "#ef6c00"


def _render_pills(gdf: pd.DataFrame):
    changed, total = _card_change_counts(gdf)
    uncertain_n = _card_uncertain_count(gdf)
    val_state = gdf.get("validation_state", pd.Series([""] * len(gdf))).astype(str).str.lower()
    reviewed = bool(total) and val_state.replace({"nan": "", "<na>": ""}).ne("").all()

    review_colour = "#2e7d32" if reviewed else "#6b7280"
    review_text = "Reviewed" if reviewed else "Not reviewed"

    cls_label, cls_colour = _card_classifier_label_and_colour(changed, total, reviewed)

    pills_html = (
        "<div class='pam-status-panel'>"
        "<div class='pam-status-label'>Card status</div>"
        "<div class='pam-pill-row'>"
        f"<span class='pam-pill' style='background-color:{review_colour};'>{review_text}</span>"
        f"<span class='pam-pill' style='background-color:{cls_colour};'>{cls_label}</span>"
    )

    if uncertain_n > 0:
        tooltip = f"{uncertain_n} uncertain detection" + ("s" if uncertain_n != 1 else "")
        pills_html += (
            f"<span title='{tooltip}' class='pam-pill' style='background-color:#b7791f;'>! {uncertain_n}</span>"
        )

    pills_html += "</div></div>"
    st.markdown(pills_html, unsafe_allow_html=True)


def render_validation(detections: Optional[pd.DataFrame], sources: dict) -> None:
    _init_filter_state()
    _init_strategy_state()
    st.header("Validation")

    proj_root = Path(sources.get("project") or sources.get("project_root") or ".")
    _load_strategy_state(proj_root)

    df_default, ds_label, ds_choices, ds_paths = _dataset_choice_validate(sources)
    if ds_label == "None" or df_default.empty:
        st.warning("Validation cannot start because the analysis dataset is not initialised. Ingest data first.")
        return

    ds_labels = list(ds_choices.keys())

    forced = st.session_state.pop("_force_validate_dataset", None)
    if forced in ds_labels:
        st.session_state["validate_dataset_selector"] = forced

    if st.session_state.get("validate_dataset_selector") not in ds_labels:
        st.session_state["validate_dataset_selector"] = ds_label

    ds_col, _ = st.columns([1.4, 3])
    with ds_col:
        dataset_label = st.selectbox("Dataset", ds_labels, key="validate_dataset_selector")

    if dataset_label != ds_label:
        df_default = ds_choices[dataset_label].copy()

    st.session_state["active_dataset_label"] = dataset_label
    st.session_state["active_dataset_path"] = str(ds_paths.get(dataset_label, ""))
    st.session_state["pa_df_det"] = df_default.copy()

    df_all = _ensure_validation_ready(df_default)

    if not st.session_state.get("validate_strategy_prompt_seen", False):
        st.session_state["validate_strategy_prompt_seen"] = True
        _save_strategy_state(proj_root)
        if not st.session_state.get("validate_strategy_dont_auto_show", False):
            st.session_state["validate_strategy_modal_open"] = True

    if hasattr(st, "dialog"):
        @st.dialog("Validation strategy", width="large")
        def _strategy_dialog():
            st.caption("Choose how clips should be selected for this review session.")

            presets = _strategy_presets(len(df_all))
            preset_labels = list(presets.keys())

            current_goal_for_preset = str(st.session_state.get("validate_strategy_goal", "representative_sample"))
            current_preset = str(st.session_state.get("validate_strategy_preset_label", "Representative sample"))
            if current_preset not in preset_labels:
                current_preset = "Representative sample"

            preset_label = st.radio(
                "Review preset",
                options=preset_labels,
                index=preset_labels.index(current_preset) if current_preset in preset_labels else 0,
                horizontal=True,
            )
            st.session_state["validate_strategy_preset_label"] = preset_label
            _apply_strategy_preset_if_requested(len(df_all), preset_label)

            st.markdown(
                f"""
                <div style="border:1px solid #e5e7eb; border-radius:0.9rem; padding:0.8rem 0.95rem; background:#f9fafb; margin-bottom:0.8rem;">
                  <div style="font-size:0.92rem; color:#111827; font-weight:600; margin-bottom:0.2rem;">{preset_label}</div>
                  <div style="font-size:0.84rem; color:#4b5563;">{presets[preset_label]['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            selected_goal = str(st.session_state.get("validate_strategy_goal", current_goal_for_preset))

            default_mode, default_value = _strategy_defaults_for_goal(selected_goal, len(df_all))
            balance_label_map = _strategy_balance_options(df_all, selected_goal)
            balance_inv = {v: k for k, v in balance_label_map.items()}

            current_balance = str(st.session_state.get("validate_strategy_balance", next(iter(balance_label_map.keys()))))
            if current_balance not in balance_label_map:
                current_balance = next(iter(balance_label_map.keys()))
            current_balance_label = balance_label_map[current_balance]

            primary_left, primary_right = st.columns([1.2, 1.0])

            with primary_left:
                balance_label = st.selectbox(
                    "Balance across",
                    options=list(balance_label_map.values()),
                    index=list(balance_label_map.values()).index(current_balance_label),
                )
                selected_balance = balance_inv[balance_label]

                if selected_goal == "custom_stratified":
                    mode_map = {
                        "Total clips": "total_clips",
                        "Clips per group": "per_group_clips",
                        "% per group": "per_group_percent",
                    }
                    current_mode = str(st.session_state.get("validate_strategy_target_mode", default_mode))
                    if current_mode not in mode_map.values():
                        current_mode = default_mode
                    mode_label = st.selectbox(
                        "How many clips",
                        options=list(mode_map.keys()),
                        index=list(mode_map.values()).index(current_mode),
                    )
                    target_mode = mode_map[mode_label]
                else:
                    target_mode = "total_clips"

                stored_target_value = int(st.session_state.get("validate_strategy_target_value", default_value))
                target_value_default = _target_value_for_widget(
                    selected_goal,
                    target_mode,
                    stored_target_value,
                    len(df_all),
                )

                label = {
                    "total_clips": "Total clips to review",
                    "per_group_clips": "Clips per group",
                    "per_group_percent": "% per group",
                }[target_mode]

                target_value = st.number_input(
                    label,
                    min_value=1,
                    max_value=100 if target_mode == "per_group_percent" else max(1, len(df_all)),
                    value=int(target_value_default),
                    step=1,
                )

            with primary_right:
                review_summary = _strategy_review_summary_text(
                    df_all,
                    selected_goal,
                    selected_balance,
                    target_mode,
                    int(target_value),
                    int(st.session_state.get("validate_strategy_bins", 5)),
                )

                st.markdown(
                    f"""
                    <div style="border:1px solid #e5e7eb; border-radius:0.9rem; padding:0.85rem 0.95rem; background:white;">
                    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:#6b7280; margin-bottom:0.28rem;">
                        Strategy overview
                    </div>
                    <div style="font-size:0.98rem; font-weight:600; color:#111827; margin-bottom:0.35rem;">
                        {_strategy_goal_label(selected_goal)} across {_strategy_balance_label(selected_balance, df_all, selected_goal)}
                    </div>
                    <div style="font-size:0.84rem; color:#4b5563; line-height:1.45;">
                        {review_summary}
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            needs_bands = (selected_goal == "equal_allocation") or ("confidence" in selected_balance) or (preset_label == "Custom")
            with st.expander("Advanced options", expanded=False):
                adv1, adv2 = st.columns(2)
                with adv1:
                    bins_value = st.number_input(
                        "Confidence bands to use",
                        min_value=2,
                        max_value=20,
                        value=int(st.session_state.get("validate_strategy_bins", 5)),
                        step=1,
                        disabled=not needs_bands,
                    )
                with adv2:
                    seed_value = st.number_input(
                        "Random seed",
                        min_value=0,
                        max_value=100000,
                        value=int(st.session_state.get("validate_strategy_seed", 42)),
                        step=1,
                    )
            if not needs_bands:
                bins_value = int(st.session_state.get("validate_strategy_bins", 5))
            if "seed_value" not in locals():
                seed_value = int(st.session_state.get("validate_strategy_seed", 42))

            preview_df, preview_metrics = _strategy_preview_matrix(
                df_all,
                selected_goal,
                selected_balance,
                target_mode,
                int(target_value),
                int(bins_value),
                int(seed_value),
                max_rows=8,
            )

            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Available", int(preview_metrics.get("available", 0)))
            with metric_cols[1]:
                st.metric("Selected", int(preview_metrics.get("selected", 0)))
            with metric_cols[2]:
                st.metric("Groups / strata", int(preview_metrics.get("strata", 0)))
            with metric_cols[3]:
                st.metric("Groups below target", int(preview_metrics.get("undersized", 0)))

            st.caption(_compact_preview_caption(selected_goal, selected_balance))
            if not preview_df.empty:
                st.dataframe(
                    _preview_display_df(preview_df),
                    width="stretch",
                    height=min(520, 44 + 32 * max(1, len(preview_df))),
                )
            else:
                st.write("No preview available for the current strategy.")

            dont_show = st.checkbox(
                "Don’t show this automatically again for me",
                value=bool(st.session_state.get("validate_strategy_dont_auto_show", False)),
            )

            b1, b2 = st.columns(2)
            with b1:
                if st.button("Skip for now", width="stretch"):
                    st.session_state["validate_strategy_modal_open"] = False
                    st.session_state["validate_strategy_dont_auto_show"] = bool(dont_show)
                    st.session_state["validate_strategy_prompt_seen"] = True
                    _save_strategy_state(proj_root)
                    if hasattr(st, "rerun"):
                        st.rerun()
                    elif hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()

            with b2:
                if st.button("Start review", width="stretch", type="primary"):
                    st.session_state["validate_strategy_goal"] = selected_goal
                    st.session_state["validate_strategy_balance"] = selected_balance
                    st.session_state["validate_strategy_target_mode"] = target_mode
                    st.session_state["validate_strategy_target_value"] = int(target_value)
                    st.session_state["validate_strategy_bins"] = int(bins_value)
                    st.session_state["validate_strategy_seed"] = int(seed_value)
                    st.session_state["validate_strategy_available"] = int(preview_metrics.get("available", 0))
                    st.session_state["validate_strategy_selected"] = int(preview_metrics.get("selected", 0))
                    st.session_state["validate_strategy_strata"] = int(preview_metrics.get("strata", 0))
                    st.session_state["validate_strategy_undersized"] = int(preview_metrics.get("undersized", 0))
                    st.session_state["validate_strategy_metrics_source"] = "wizard_preview_metrics"
                    st.session_state["validate_strategy_modal_open"] = False
                    st.session_state["validate_strategy_dont_auto_show"] = bool(dont_show)
                    st.session_state["validate_strategy_prompt_seen"] = True
                    _save_strategy_state(proj_root)
                    if hasattr(st, "rerun"):
                        st.rerun()
                    elif hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()

        @st.dialog("Interactive spectrogram", width="large")
        def _interactive_spectrogram_dialog():
            selected_card = st.session_state.get("validate_interactive_card")
            if not selected_card:
                st.write("No card selected.")
                return

            sel_base = str(selected_card["base"])
            sel_species_orig = str(selected_card["species_orig"])
            try:
                _render_interactive_validate_dialog(
                    proj_root=proj_root,
                    df_all=df_all,
                    grouped=grouped,
                    base=sel_base,
                    species_orig=sel_species_orig,
                    xmin=float(selected_card["xmin"]),
                    xmax=float(selected_card["xmax"]),
                    ymin=float(selected_card["ymin"]),
                    ymax=float(selected_card["ymax"]),
                    n_fft=int(selected_card["n_fft"]),
                    hop_length=int(selected_card["hop_length"]),
                )
            except Exception as e:
                st.error(f"Interactive spectrogram error: {e}")

    _render_strategy_summary_bar(df_all)

    if hasattr(st, "dialog") and st.session_state.get("validate_strategy_modal_open", False):
        _strategy_dialog()

    top1, top2, top3 = st.columns([1, 1, 1])
    with top1:
        NUM_PER_PAGE = st.number_input(
            "Spectrograms per page",
            min_value=1,
            max_value=40,
            step=1,
            key="validate_num_per_page",
        )
    with top2:
        COLS_PER_ROW = st.slider(
            "Columns per row",
            min_value=1,
            max_value=5,
            key="validate_cols_per_row",
        )

    _sync_validate_page_input_from_page()

    with top3:
        st.number_input(
            "Page",
            min_value=1,
            step=1,
            key="validate_page_input",
            on_change=_on_validate_page_input_change,
        )

    with st.expander("Advanced filters", expanded=False):
        r1c1, r1c2, r1c3 = st.columns([1, 1, 1])
        with r1c1:
            show_label = st.selectbox(
                "Show clips labelled",
                ["present", "absent", "uncertain", "all", "user changed only"],
                key="validate_show_label",
            )
        with r1c2:
            min_prob = st.slider(
                "Min detection probability",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key="validate_min_prob",
            )
        with r1c3:
            conf_sort = st.selectbox(
                "Sort spectrograms",
                ["Strategy/default order", "Confidence high to low", "Confidence low to high"],
                key="validate_conf_sort",
                on_change=_reset_validate_page_for_sort_change,
                help="Sorts only the detections selected by the current filters and sampling strategy.",
            )

        frow1, frow2, frow3, frow4, frow5 = st.columns([0.9, 0.7, 0.9, 0.9, 0.9])
        with frow1:
            lock_freq = st.checkbox("Lock frequency (kHz)", key="validate_lock_freq")
        with frow2:
            fmin_khz = st.number_input(
                "Min",
                min_value=0.0,
                max_value=200.0,
                step=1.0,
                disabled=not lock_freq,
                key="validate_fmin_khz",
            )
        with frow3:
            fmax_khz = st.number_input(
                "Max",
                min_value=1.0,
                max_value=250.0,
                step=1.0,
                disabled=not lock_freq,
                key="validate_fmax_khz",
            )
        with frow4:
            use_te_override = st.checkbox("Set Time Expansion Factor", key="validate_use_te_override")
        with frow5:
            te_override = st.number_input(
                "TE factor",
                min_value=1,
                max_value=32,
                step=1,
                key="validate_te_override",
                disabled=not use_te_override,
            )

        zrow1, zrow2 = st.columns([1.2, 0.8])
        with zrow1:
            auto_zoom_single_detection = st.checkbox(
                "Auto-fit single-detection cards",
                key="validate_auto_zoom_single_detection",
                help="When a card contains one detection, the spectrogram opens around that detection. When it contains multiple detections, it opens around the smallest window covering those detections.",                
                on_change=_clear_validate_time_window_state,
            )
        with zrow2:
            auto_zoom_window_s = st.number_input(
                "Single-detection window (s)",
                min_value=1.0,
                max_value=120.0,
                step=0.5,
                format="%.1f",
                disabled=not auto_zoom_single_detection,
                key="validate_auto_zoom_window_s",
                on_change=_clear_validate_time_window_state,
            )

        fft_col1, fft_col2 = st.columns([1.0, 1.2])
        with fft_col1:
            use_fft_override = st.checkbox("Set FFT size", key="validate_use_fft_override")
        with fft_col2:
            st.selectbox(
                "FFT size",
                options=[1024, 2048, 4096, 8192, 16384],
                key="validate_fft_size",
                disabled=not use_fft_override,
            )

        group_candidates = []
        label_map: Dict[str, str] = {}
        for label, col in [
            ("Species", "species_display_original"),
            ("Recorder ID", "recorder_id"),
            ("Site", "site"),
            ("Detector ID", "detector_id"),
        ]:
            if col in df_all.columns:
                group_candidates.append(label)
                label_map[label] = col

        if group_candidates:
            st.markdown("---")
            st.markdown("**Group filter (optional)**")
            group_options = ["[none]"] + group_candidates
            group_label = st.selectbox("Filter by group", group_options, key="validate_group_label")

            if group_label != "[none]":
                group_col = label_map[group_label]
                all_vals = df_all[group_col].dropna().astype(str).sort_values().unique()
                st.multiselect("Only show these values", options=list(all_vals), key="validate_group_values")
                st.session_state["validate_group_col"] = group_col
            else:
                st.session_state["validate_group_col"] = ""
        else:
            st.session_state["validate_group_col"] = ""

    group_col = st.session_state.get("validate_group_col", "")
    group_values = st.session_state.get("validate_group_values", [])

    orig_sp_all = df_all.get("species_name_original", df_all.get("species_name", "")).astype(str)
    orig_pl_all = df_all.get("presence_label_original", df_all.get("presence_label", "")).astype(str).str.lower()
    cur_sp_all = df_all.get("species_name", "").astype(str)
    cur_pl_all = df_all.get("presence_label", "").astype(str).str.lower()
    val_state_all = df_all.get("validation_state", pd.Series([""] * len(df_all))).astype(str).str.lower()
    df_all["reviewed_flag"] = val_state_all.replace({"nan": "", "<na>": ""}).ne("")
    df_all["uncertain_flag_bool"] = df_all.get(
        "uncertain_flag",
        pd.Series([""] * len(df_all), index=df_all.index)
    ).map(_bool_from_any)
    df_all["changed_flag"] = (orig_sp_all != cur_sp_all) | (orig_pl_all != cur_pl_all)

    df_candidates = df_all.copy()

    if group_col and group_values:
        df_candidates = df_candidates[df_candidates[group_col].astype(str).isin(group_values)]

    if show_label in ("present", "absent"):
        orig_pl_view = df_candidates.get("presence_label_original", df_candidates.get("presence_label", "")).astype(str).str.lower()
        if show_label == "present":
            df_candidates = df_candidates[orig_pl_view.eq("present")]
        else:
            df_candidates = df_candidates[orig_pl_view.ne("present")]
    elif show_label == "uncertain":
        df_candidates = df_candidates[df_candidates["uncertain_flag_bool"].astype(bool)]
    elif show_label == "user changed only":
        df_candidates = df_candidates[df_candidates["changed_flag"].astype(bool)]

    df_candidates["detection_probability"] = pd.to_numeric(df_candidates["detection_probability"], errors="coerce").fillna(0.0)
    df_candidates = df_candidates[df_candidates["detection_probability"] >= float(min_prob)]
    if df_candidates.empty:
        st.info("No clips match the current filters.")
        st.session_state["pa_df_det"] = df_all.copy()
        return

    strategy_scope_n = len(df_candidates)
    goal, balance, target_mode, target_value, bins, seed = _effective_strategy_settings(len(df_candidates), df_all)

    strategy_preview_df, strategy_preview_metrics = _strategy_preview_matrix(
        df_candidates, goal, balance, target_mode, target_value, bins, seed, max_rows=8
    )

    df_view, strategy_meta = _select_by_strategy(df_candidates, df_all)
    sampled_n = len(df_view)

    total_in_scope = len(df_view)
    reviewed_mask = df_view["reviewed_flag"].astype(bool)
    changed_mask = df_view["changed_flag"].astype(bool) & reviewed_mask
    uncertain_mask = df_view["uncertain_flag_bool"].astype(bool)

    val_state_local = df_view.get("validation_state", pd.Series([""] * len(df_view))).astype(str).str.lower()
    correct_mask = reviewed_mask & val_state_local.eq("correct")

    n_reviewed = int(reviewed_mask.sum())
    n_changed = int(changed_mask.sum())
    n_correct = int(correct_mask.sum())
    n_uncertain = int(uncertain_mask.sum())
    n_sparse = _strategy_shortfall_count(
        df_scope=df_candidates,
        df_selected=df_view,
        goal=goal,
        balance=balance,
        target_mode=target_mode,
        target_value=target_value,
    )
    pct_reviewed = (100.0 * n_reviewed / total_in_scope) if total_in_scope else 0.0
    pct_correct = (100.0 * n_correct / n_reviewed) if n_reviewed else 0.0
    pct_changed = (100.0 * n_changed / n_reviewed) if n_reviewed else 0.0

    with st.expander("Validation progress (current filters)", expanded=True):
        st.caption(f"Strategy selection: showing {sampled_n} clips from {strategy_scope_n} clips after the current filters.")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("Selected clips", total_in_scope)
        with m2:
            st.metric("Reviewed", f"{n_reviewed} ({pct_reviewed:.0f}%)")
        with m3:
            st.metric("Classifier correct", f"{n_correct} ({pct_correct:.0f}%)")
        with m4:
            st.metric("Changed of reviewed", f"{n_changed} ({pct_changed:.0f}%)")
        with m5:
            st.metric("Flagged uncertain", n_uncertain)
        with m6:
            st.metric("Groups below target", n_sparse)

        if "species_display_original" in df_view.columns:
            grp = (
                df_view.groupby("species_display_original", dropna=False)
                .agg(
                    detections=("species_display_original", "size"),
                    reviewed_n=("reviewed_flag", "sum"),
                    changed_n=("changed_flag", "sum"),
                    uncertain_n=("uncertain_flag_bool", "sum"),
                )
            )

            grp["pct_reviewed"] = (100.0 * grp["reviewed_n"] / grp["detections"]).round(1)
            grp["pct_changed_of_reviewed"] = np.where(
                grp["reviewed_n"] > 0,
                (100.0 * grp["changed_n"] / grp["reviewed_n"]).round(1),
                np.nan,
            )

            st.dataframe(
                grp.reset_index().rename(columns={"species_display_original": "species"}).sort_values("pct_reviewed", ascending=False),
                width="stretch",
            )

        with st.expander("Sampling preview for the active strategy", expanded=False):
            st.caption(_compact_preview_caption(goal, balance))
            if not strategy_preview_df.empty:
                st.dataframe(
                    _preview_display_df(strategy_preview_df),
                    width="stretch",
                    height=min(520, 44 + 32 * max(1, len(strategy_preview_df))),
                )
            else:
                st.write("No preview available for the current strategy.")

    df_view = df_view.sort_values(["basename", "species_display_original", "start_s"])
    grouped = df_view.groupby(["basename", "species_display_original"], dropna=False)
    groups: List[tuple[str, str]] = list(grouped.indices.keys())

    conf_sort = str(st.session_state.get("validate_conf_sort", "Strategy/default order"))
    if conf_sort in ("Confidence high to low", "Confidence low to high"):
        g_scores = {k: _group_max_prob(grouped.get_group(k)) for k in groups}

        def _confidence_sort_key(k):
            score = g_scores.get(k, np.nan)
            score = float(score) if np.isfinite(score) else -1.0
            score_key = -score if conf_sort == "Confidence high to low" else score
            return (score_key, str(k[0]).lower(), str(k[1]).lower())

        groups = sorted(groups, key=_confidence_sort_key)
    elif goal == "find_likely_mistakes":
        g_scores = {k: _group_max_prob(grouped.get_group(k)) for k in groups}
        groups = sorted(groups, key=lambda k: g_scores.get(k, -np.inf), reverse=False)
    elif goal == "review_strongest":
        g_scores = {k: _group_max_prob(grouped.get_group(k)) for k in groups}
        groups = sorted(groups, key=lambda k: g_scores.get(k, -np.inf), reverse=True)
    else:
        rng = np.random.default_rng(int(seed))
        g_shuffle = list(groups)
        rng.shuffle(g_shuffle)
        groups = g_shuffle

    total_cards = len(groups)
    total_pages = max(1, math.ceil(total_cards / int(NUM_PER_PAGE)))
    st.session_state["_validate_total_pages"] = total_pages

    page_raw = int(st.session_state.get("validate_page", 1))
    PAGE = max(1, min(page_raw, total_pages))

    if PAGE != int(st.session_state.get("validate_page", 1)):
        st.session_state["validate_page"] = PAGE
        st.session_state["validate_page_sync_pending"] = True

    start_idx = (PAGE - 1) * int(NUM_PER_PAGE)
    end_idx = min(total_cards, start_idx + int(NUM_PER_PAGE))
    page_keys = groups[start_idx:end_idx]
    st.caption(f"Showing {len(page_keys)} of {total_cards} spectrograms (page {PAGE} of {total_pages})")

    species_choices = sorted(
        pd.unique(
            pd.concat([
                df_all.get("species_name", pd.Series([], dtype=object)).astype(str),
                df_all.get("class", pd.Series([], dtype=object)).astype(str),
                df_all.get("validation_species", pd.Series([], dtype=object)).astype(str),
            ], ignore_index=True)
        ).tolist()
    )
    species_choices = [
        s.strip() for s in species_choices
        if s and s.strip() and s.strip().lower() not in ("nan", "none", "<na>", "[absent]", _ADD_SPECIES_OPTION.lower())
    ]
    species_choices = sorted(pd.unique(pd.Series(species_choices, dtype=object)).tolist(), key=lambda x: x.lower())
    species_choices.insert(0, "[absent]")
    species_select_options = species_choices + [_ADD_SPECIES_OPTION]

    n_rows = math.ceil(len(page_keys) / int(COLS_PER_ROW))
    for r in range(n_rows):
        cols = st.columns(int(COLS_PER_ROW))
        for c in range(int(COLS_PER_ROW)):
            gi = r * int(COLS_PER_ROW) + c
            if gi >= len(page_keys):
                break

            base, species_orig = page_keys[gi]
            gdf = grouped.get_group((base, species_orig)).copy()

            if "detection_probability" not in gdf.columns:
                gdf["detection_probability"] = gdf.apply(_best_prob_from_row, axis=1)

            apath = _resolve_audio_path(proj_root, gdf, df_all)
            gdf_plot = _rows_for_resolved_audio(proj_root, gdf, apath)

            all_detectable_boxes: List[Dict[str, float]] = []
            for row_index, row in gdf_plot.iterrows():
                b = {
                    "row_index": row_index,
                    "start_s": _num(row.get("start_s", row.get("detection_start_s"))),
                    "end_s": _num(row.get("end_s", row.get("detection_end_s"))),
                    "low_freq": _num(row.get("low_freq")),
                    "high_freq": _num(row.get("high_freq")),
                    "prob": _num(row.get("detection_probability")),
                }
                if np.isfinite(b["start_s"]) and np.isfinite(b["end_s"]) and b["end_s"] > b["start_s"]:
                    all_detectable_boxes.append(b)
            boxes = sorted(all_detectable_boxes, key=lambda b: (b["prob"] if np.isfinite(b["prob"]) else -1.0), reverse=True)[:10]
            displayed_indices = [b["row_index"] for b in boxes if b.get("row_index") in gdf.index]

            if displayed_indices:
                gdf_card = gdf.loc[displayed_indices].copy()

                # Use the current persisted dataframe for status, so reviewed/changed/uncertain
                # pills match the live table after reload.
                gdf_card_status = df_all.loc[displayed_indices].copy()
            else:
                gdf_card = gdf.iloc[0:0].copy()
                gdf_card_status = gdf_card.copy()

            n_displayed_det = int(len(gdf_card))

            max_cp = _group_max_prob(gdf_card if not gdf_card.empty else gdf_plot)
            title_html = (
                f"<div class='pam-card-header pam-card-title'><strong>{base}</strong>"
                f"<br>{species_orig}"
                f"<br>Displayed detections: {n_displayed_det}"
            )
            if np.isfinite(max_cp):
                title_html += f"<br>Max probability: {max_cp:.2f}"
            title_html += "</div>"

            with cols[c]:
                with st.container(border=True):
                    h1, h2 = st.columns([2.0, 1.0])
                    with h1:
                        st.markdown(title_html, unsafe_allow_html=True)
                    with h2:
                        _render_pills(gdf_card_status)
                    y, sr = np.array([], dtype=np.float32), 1
                    dur = 0.0
                    xmin, xmax = 0.0, 1.0
                    ymin, ymax = 0.0, 1.0
                    S_dB = np.zeros((2, 2))
                    times = np.arange(2, dtype=float)
                    freqs_hz = np.arange(2, dtype=float)
                    n_fft = _get_validate_n_fft(sr)
                    hop = max(1, n_fft // 8)

                    if not (apath and apath.exists()):
                        st.error("Audio not found")
                    else:
                        try:
                            sr_info, dur_info = _audio_info(apath)
                            if sr_info <= 0 or dur_info <= 0:
                                raise ValueError("Audio metadata could not be read")

                            sr = int(sr_info)
                            dur = float(dur_info)
                            n_fft = _get_validate_n_fft(sr)
                            hop = max(1, n_fft // 8)

                            time_state_key = _safe_widget_key("validate_time_window_state", base, species_orig)
                            time_key_start = _safe_widget_key("validate_time_xmin_input", base, species_orig)
                            time_key_end = _safe_widget_key("validate_time_xmax_input", base, species_orig)
                            stored_window = dict(st.session_state.get(time_state_key, {}))

                            default_start, default_end, default_signature = _default_detection_window(
                                boxes=boxes,
                                duration_s=dur,
                                default_single_window_s=float(st.session_state.get("validate_auto_zoom_window_s", 5.0)),
                                padding_s=2.0,
                            )

                            user_override = bool(stored_window.get("user_override", False))
                            previous_signature = stored_window.get("default_signature")
                            if (not user_override) or previous_signature != default_signature:
                                st.session_state[time_key_start] = float(default_start)
                                st.session_state[time_key_end] = float(default_end)
                                st.session_state[time_state_key] = {
                                    "xmin": float(default_start),
                                    "xmax": float(default_end),
                                    "user_override": False,
                                    "default_signature": default_signature,
                                }
                            else:
                                stored_start = float(min(max(0.0, _num(stored_window.get("xmin", st.session_state.get(time_key_start, default_start)))), float(dur)))
                                stored_end = float(min(max(0.0, _num(stored_window.get("xmax", st.session_state.get(time_key_end, default_end)))), float(dur)))
                                if stored_end <= stored_start:
                                    stored_start, stored_end = float(default_start), float(default_end)
                                if time_key_start not in st.session_state:
                                    st.session_state[time_key_start] = stored_start
                                if time_key_end not in st.session_state:
                                    st.session_state[time_key_end] = stored_end

                            tw1, tw2, tw3 = st.columns([1.0, 1.0, 0.16])
                            with tw3:
                                st.markdown("<div style='height:1.65rem'></div>", unsafe_allow_html=True)
                                reset_window = st.button(
                                    "↺",
                                    key=_safe_widget_key("validate_time_reset", base, species_orig),
                                    help="Reset x-axis to current default",
                                )
                            if reset_window:
                                st.session_state[time_key_start] = float(default_start)
                                st.session_state[time_key_end] = float(default_end)
                                st.session_state[time_state_key] = {
                                    "xmin": float(default_start),
                                    "xmax": float(default_end),
                                    "user_override": False,
                                    "default_signature": default_signature,
                                }
                                if hasattr(st, "rerun"):
                                    st.rerun()
                                elif hasattr(st, "experimental_rerun"):
                                    st.experimental_rerun()

                            with tw1:
                                x_start = st.number_input(
                                    "X min (s)",
                                    min_value=0.0,
                                    max_value=float(dur),
                                    step=0.1,
                                    format="%.3f",
                                    key=time_key_start,
                                    on_change=_mark_validate_time_window_override,
                                    args=(time_state_key,),
                                )
                            with tw2:
                                x_end = st.number_input(
                                    "X max (s)",
                                    min_value=0.0,
                                    max_value=float(dur),
                                    step=0.1,
                                    format="%.3f",
                                    key=time_key_end,
                                    on_change=_mark_validate_time_window_override,
                                    args=(time_state_key,),
                                )

                            if float(x_end) <= float(x_start):
                                x_start = float(default_start)
                                x_end = float(default_end)

                            xmin, xmax = max(0.0, float(x_start)), min(float(dur), float(x_end))
                            if xmax <= xmin:
                                xmin, xmax = float(default_start), float(default_end)
                            current_state = dict(st.session_state.get(time_state_key, {}))
                            st.session_state[time_state_key] = {
                                "xmin": float(xmin),
                                "xmax": float(xmax),
                                "user_override": bool(current_state.get("user_override", False)),
                                "default_signature": default_signature,
                            }

                            y, sr, actual_start, actual_end = _load_audio_window(apath, xmin, xmax, fallback_sr=sr)
                            if y.size == 0 or sr <= 0:
                                raise ValueError("Audio window could not be read")
                            xmin = float(actual_start)
                            xmax = float(actual_end)

                            if lock_freq and (fmax_khz > fmin_khz):
                                ymin = max(0.0, float(fmin_khz) * 1000.0)
                                ymax = float(fmax_khz) * 1000.0
                                nyq = 0.5 * sr * 0.98
                                ymax = min(ymax, nyq)
                            else:
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

                            spec = _compute_static_spectrogram_data(
                                y=y,
                                sr=sr,
                                n_fft=n_fft,
                                hop_length=hop,
                            )
                            S_dB = spec["S_dB"]
                            times = spec["times"] + float(xmin)
                            freqs_hz = spec["freqs_hz"]
                        except Exception as e:
                            st.error(f"Spectrogram setup error: {e}")
                            y, sr = np.array([], dtype=np.float32), 1
                            dur = 0.0
                            n_fft = _get_validate_n_fft(sr)
                            hop = max(1, n_fft // 8)
                            times = np.arange(2, dtype=float)
                            freqs_hz = np.arange(2, dtype=float)
                            S_dB = np.zeros((2, 2))
                            xmin, xmax = 0.0, 1.0

                        try:
                            fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=150, constrained_layout=False)
                            plot_xmin, plot_xmax = float(xmin), float(xmax)
                            if not (np.isfinite(plot_xmin) and np.isfinite(plot_xmax) and plot_xmax > plot_xmin):
                                plot_xmin, plot_xmax = 0.0, float(dur)
                            plot_span = max(1e-6, plot_xmax - plot_xmin)
                            plot_S = S_dB
                            if len(times) > 1:
                                time_mask = (times >= plot_xmin) & (times <= plot_xmax)
                                if int(np.count_nonzero(time_mask)) >= 2:
                                    plot_S = S_dB[:, time_mask]
                                else:
                                    left_idx = int(np.searchsorted(times, plot_xmin, side="left"))
                                    right_idx = int(np.searchsorted(times, plot_xmax, side="right"))
                                    left_idx = max(0, min(left_idx, len(times) - 2))
                                    right_idx = max(left_idx + 2, min(right_idx, len(times)))
                                    plot_S = S_dB[:, left_idx:right_idx]
                            extent = [0.0, plot_span, freqs_hz.min(), freqs_hz.max()]
                            ax.imshow(
                                plot_S,
                                origin="lower",
                                aspect="auto",
                                interpolation="nearest",
                                extent=extent,
                                vmin=S_dB.max() - 90,
                                vmax=S_dB.max(),
                            )
                            ax.set_xlim(0.0, plot_span)
                            ax.set_ylim(ymin, ymax)
                            ax.set_xlabel("Time (s)")
                            ax.set_ylabel("Frequency (kHz)")
                            ax.yaxis.set_major_formatter(FuncFormatter(lambda ytick, pos: f"{ytick/1000:.0f}"))
                            tick_count = 6 if plot_span >= 5.0 else 5
                            tick_positions = np.linspace(0.0, plot_span, tick_count)
                            ax.set_xticks(tick_positions)
                            ax.set_xticklabels([f"{plot_xmin + t:.1f}" for t in tick_positions])

                            for b in boxes:
                                x0, x1 = b["start_s"], b["end_s"]
                                if not (np.isfinite(x0) and np.isfinite(x1)):
                                    continue
                                clipped_x0 = max(float(x0), plot_xmin)
                                clipped_x1 = min(float(x1), plot_xmax)
                                if clipped_x1 <= clipped_x0:
                                    continue
                                rel_x0 = clipped_x0 - plot_xmin
                                rel_width = clipped_x1 - clipped_x0
                                prob = b["prob"]
                                ax.add_patch(
                                    Rectangle(
                                        (rel_x0, ymin),
                                        rel_width,
                                        ymax - ymin,
                                        facecolor=(1, 1, 1, 0.06),
                                        edgecolor=(1, 1, 1, 0.12),
                                        linewidth=0.6,
                                    )
                                )
                                if np.isfinite(prob):
                                    xm = rel_x0 + rel_width * 0.5
                                    ym = ymin + 0.88 * (ymax - ymin)
                                    ax.text(
                                        xm,
                                        ym,
                                        f"{prob:.2f}",
                                        ha="center",
                                        va="center",
                                        color="white",
                                        fontsize=9,
                                        bbox=dict(
                                            boxstyle="round,pad=0.18",
                                            fc=(0, 0, 0, 0.55),
                                            ec=(1, 1, 1, 0.25),
                                            lw=0.5,
                                        ),
                                    )

                            st.pyplot(fig, width="stretch", clear_figure=True)
                            plt.close(fig)
                        except Exception as e:
                            st.error(f"Spectrogram error: {e}")

                        if st.button(
                            "Open interactive spectrogram",
                            key=_safe_widget_key("open_interactive_plotly", base, species_orig),
                            width="stretch",
                        ):
                            st.session_state["validate_interactive_card"] = {
                                "base": base,
                                "species_orig": species_orig,
                                "xmin": float(xmin),
                                "xmax": float(xmax),
                                "ymin": float(ymin),
                                "ymax": float(ymax),
                                "n_fft": int(n_fft),
                                "hop_length": int(hop),
                            }
                            _interactive_spectrogram_dialog()

                        try:
                            y_seg = y
                            low_edge = _estimate_low_edge_hz_for_group(gdf)
                            te_auto = _choose_te_for_group(low_edge, sr)
                            use_te_override_flag = bool(st.session_state.get("validate_use_te_override", False))
                            if use_te_override_flag:
                                te_val = int(st.session_state.get("validate_te_override", te_auto or 1))
                                te = max(1, te_val)
                            else:
                                te = max(1, int(te_auto))

                            y_play, psr = _apply_time_expansion_for_playback(y_seg, sr, te)
                            tmp_wav = _tmp_audio_path(proj_root, base, f"{species_orig}|{float(xmin):.3f}-{float(xmax):.3f}", int(te), int(psr), int(y_play.size))
                            if not tmp_wav.exists() or tmp_wav.stat().st_size == 0:
                                sf.write(str(tmp_wav), y_play, int(psr), format="WAV", subtype="PCM_16")
                            st.audio(str(tmp_wav))
                        except Exception as e:
                            st.error(f"Playback error: {e}")

                    with st.form(key=_safe_widget_key("validate_card_review_form", base, species_orig), border=False):
                        card_form_values: Dict[int, Dict[str, object]] = {}

                        with st.expander("Edit detections (species)"):
                            acoustic_lookup: Dict[int, Dict[str, str]] = {}

                            if y.size > 0 and sr > 1:
                                requested_metric_fft = int(n_fft)
                                gdf_metric_source = _offset_detection_times(gdf, float(xmin))
                                metric_fft = _card_metric_fft(
                                    gdf=gdf_metric_source,
                                    y=y,
                                    sr=sr,
                                    requested_n_fft=requested_metric_fft,
                                )

                                if metric_fft is not None:
                                    metric_hop = max(1, metric_fft // 8)

                                    gdf_for_metrics = gdf_metric_source.copy().sort_values("start_s").reset_index(drop=True)

                                    for ridx_metric, row_metric in gdf_for_metrics.iterrows():
                                        start_s_metric = _num(row_metric.get("start_s", row_metric.get("detection_start_s")))
                                        end_s_metric = _num(row_metric.get("end_s", row_metric.get("detection_end_s")))
                                        low_freq_metric = _num(row_metric.get("low_freq"))
                                        high_freq_metric = _num(row_metric.get("high_freq"))
                                        prob_metric = _num(row_metric.get("detection_probability"))

                                        metrics = _acoustic_metrics_for_detection(
                                            y=y,
                                            sr=sr,
                                            start_s=start_s_metric,
                                            end_s=end_s_metric,
                                            low_freq=low_freq_metric,
                                            high_freq=high_freq_metric,
                                            n_fft=metric_fft,
                                            hop_length=metric_hop,
                                        )

                                        fft_note = ""
                                        if metric_fft != requested_metric_fft:
                                            fft_note = f" • FFT {metric_fft}"

                                        acoustic_lookup[int(ridx_metric)] = {
                                            "duration": _fmt_ms(metrics.get("duration_s", np.nan)),
                                            "peak": _fmt_khz(metrics.get("peak_freq_hz", np.nan)),
                                            "centroid": _fmt_khz(metrics.get("centroid_hz", np.nan)),
                                            "prob": f"{prob_metric:.2f}" if np.isfinite(prob_metric) else "—",
                                            "fft_note": fft_note,
                                        }

                            gdf_with_idx = gdf_card.copy()
                            gdf_with_idx["__orig_index"] = gdf_with_idx.index
                            rgdf = gdf_with_idx.reset_index(drop=True)

                            for ridx, row in rgdf.iterrows():
                                ts = row.get("start_s", row.get("detection_start_s", np.nan))
                                ts_str = f"{float(ts):.2f}s" if np.isfinite(_num(ts)) else "—"

                                cur_sp_row = str(row.get("species_name", "") or "").strip()
                                cur_pl_row = str(row.get("presence_label", "") or "").lower()
                                current_species_choice = "[absent]" if (cur_pl_row != "present" or cur_sp_row.strip() == "") else cur_sp_row
                                select_key = f"sp_{base}_{species_orig}_{ridx}"
                                new_species_key = f"{select_key}_new"
                                options_for_row = list(species_select_options)
                                if current_species_choice not in options_for_row and current_species_choice != "[absent]":
                                    options_for_row.insert(1, current_species_choice)
                                try:
                                    idx_choice = options_for_row.index(current_species_choice)
                                except ValueError:
                                    idx_choice = 0

                                row_left, row_right = st.columns([4.0, 1.2])

                                with row_left:
                                    if select_key in st.session_state:
                                        if st.session_state.get(select_key) not in options_for_row:
                                            st.session_state[select_key] = current_species_choice
                                        selected_species_choice = st.selectbox(
                                            f"Detection {ridx+1} @ {ts_str}",
                                            options=options_for_row,
                                            key=select_key,
                                        )
                                    else:
                                        selected_species_choice = st.selectbox(
                                            f"Detection {ridx+1} @ {ts_str}",
                                            options=options_for_row,
                                            index=idx_choice,
                                            key=select_key,
                                        )
                                    new_species_value = st.session_state.get(new_species_key, "")
                                    if selected_species_choice == _ADD_SPECIES_OPTION:
                                        new_species_value = st.text_input(
                                            "New species",
                                            key=new_species_key,
                                            placeholder="Type species name",
                                        )

                                with row_right:
                                    current_uncertain = _bool_from_any(row.get("uncertain_flag", ""))
                                    uncertain_value = st.checkbox(
                                        "Uncertain",
                                        value=current_uncertain,
                                        key=f"unc_{base}_{species_orig}_{ridx}",
                                    )

                                note_key = f"note_{base}_{species_orig}_{ridx}"
                                existing_note = str(row.get("validation_notes", "") or "")
                                if note_key not in st.session_state:
                                    st.session_state[note_key] = existing_note
                                with st.expander("Notes", expanded=bool(existing_note.strip())):
                                    note_value = st.text_area(
                                        "Optional note for this detection",
                                        key=note_key,
                                        height=80,
                                        placeholder="Add context for uncertainty, species changes, or other validation decisions.",
                                    )

                                card_form_values[int(ridx)] = {
                                    "species_key": select_key,
                                    "species_value": selected_species_choice,
                                    "new_species_key": new_species_key,
                                    "new_species_value": new_species_value,
                                    "uncertain_key": f"unc_{base}_{species_orig}_{ridx}",
                                    "uncertain_value": bool(uncertain_value),
                                    "note_key": note_key,
                                    "note_value": note_value,
                                }

                                summary = acoustic_lookup.get(int(ridx))
                                if summary:
                                    st.markdown(
                                        f"""
                                        <div style="
                                            margin:-0.10rem 0 0.75rem 0.1rem;
                                            font-size:0.98rem;
                                            color:#374151;
                                            line-height:1.45;
                                        ">
                                            <span style="font-weight:600;">{summary['duration']}</span>
                                            <span style="color:#9ca3af;"> • </span>
                                            <span><strong>{summary['peak']}</strong> peak energy</span>
                                            <span style="color:#9ca3af;"> • </span>
                                            <span><strong>{summary['centroid']}</strong> centroid</span>
                                            <span style="color:#9ca3af;"> • </span>
                                            <span>p=<strong>{summary['prob']}</strong></span>
                                            <span style="color:#6b7280;">{summary.get('fft_note', '')}</span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                        st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
                        card_review_submitted = st.form_submit_button(
                            "Mark card as reviewed",
                            key=_safe_widget_key("mark_reviewed_form", base, species_orig),
                            width="stretch",
                            type="primary",
                        )


                    if card_review_submitted:
                            selected_indices = list(gdf_card.index)
                            updated_df, _, _ = _commit_card(
                                proj_root,
                                df_all,
                                base,
                                species_orig,
                                selected_indices=selected_indices,
                                submitted_values=card_form_values,
                            )
                            out = proj_root / "data_normalised" / "detections_validated.csv"
                            out.parent.mkdir(parents=True, exist_ok=True)
                            updated_df.to_csv(out, index=False)

                            st.session_state["_force_validate_dataset"] = "Validated (published)"
                            st.session_state["active_dataset_label"] = "Validated (published)"
                            st.session_state["active_dataset_path"] = str(out)
                            st.session_state["pa_df_det"] = updated_df
                            df_all = updated_df.copy()

                            if hasattr(st, "rerun"):
                                st.rerun()
                            elif hasattr(st, "experimental_rerun"):
                                st.experimental_rerun()

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    nav_left, nav_mid, nav_right = st.columns([1.2, 1.2, 4])

    with nav_left:
        st.button(
            "Previous page",
            width="stretch",
            disabled=PAGE <= 1,
            on_click=_go_to_previous_validate_page,
        )

    with nav_mid:
        st.button(
            "Next page",
            width="stretch",
            disabled=PAGE >= total_pages,
            on_click=_go_to_next_validate_page,
        )

    st.session_state["pa_df_det"] = df_all.copy()

    st.divider()
    st.subheader("Tracked species changes (saved)")

    if not df_all.empty:
        orig_sp_all = df_all.get("species_name_original", df_all.get("species_name", "")).astype(str)
        orig_pl_all = df_all.get("presence_label_original", df_all.get("presence_label", "")).astype(str).str.lower()
        cur_sp_all = df_all.get("species_name", "").astype(str)
        cur_pl_all = df_all.get("presence_label", "").astype(str).str.lower()
        change_mask_all = (orig_sp_all != cur_sp_all) | (orig_pl_all != cur_pl_all)

        if change_mask_all.any():
            changed_df = df_all.loc[change_mask_all, [
                col for col in [
                    "detection_id",
                    "basename",
                    "species_name_original",
                    "presence_label_original",
                    "species_name",
                    "presence_label",
                    "uncertain_flag",
                    "validation_notes",
                ] if col in df_all.columns
            ]].copy()
            st.dataframe(changed_df, width="stretch")
        else:
            st.write("No saved species changes yet.")
    else:
        st.write("No saved species changes yet.")

    UNWANTED = [
        "validation_method", "user_changed", "user_changed_by", "user_changed_at",
        "FinalLabelEffective", "species_display", "species_display_original",
        "changed_flag", "reviewed_flag", "uncertain_flag_bool",
        "source_file", "FinalLabel", "class",
        "class_prob", "UserLabel", "is_present", "Changed", "lat", "lon",
        "filename_stem", "dt", "time_of_day", "tod_ts",
        "__strategy_parent", "__strategy_bin", "__strategy_stratum", "__strategy_priority",
    ]

    st.divider()
    st.subheader("Download validated data")
    st.markdown(
        "CSV exports the validated detections only.  \
"
        "Excel exports the validated detections plus a validation strategy summary."
    )

    user_name = (
        str(st.session_state.get("auth_user") or st.session_state.get("user_name") or "")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or "reviewer"
    )
    export_filename = _make_export_filename(proj_root, user_name)

    export_df = df_all.copy()

    for c in ["validation_state", "validation_label", "validation_species", "validated_by", "validated_at", "uncertain_flag", "validation_notes"]:
        if c not in export_df.columns:
            export_df[c] = ""

    export_df = export_df.drop(columns=UNWANTED, errors="ignore")

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    strategy_export_df = _strategy_export_summary_df(df_all.copy(), proj_root, user_name)
    xlsx_bytes = _validated_workbook_bytes(export_df, strategy_export_df)

    dl_cols = st.columns(2)
    with dl_cols[0]:
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name=export_filename,
            mime="text/csv",
        )
    with dl_cols[1]:
        st.download_button(
            "Download Excel workbook",
            data=xlsx_bytes,
            file_name=_make_export_xlsx_filename(export_filename),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

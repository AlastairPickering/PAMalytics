# scripts/pages/1_Validate.py
from __future__ import annotations

import math
import hashlib
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle

# Page config
try:
    st.set_page_config(layout="wide", page_title="Validate")
except Exception:
    pass


# utilities

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


def _get_active_dataset_or_load(sources: dict, df_passed: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Resolve the active detections dataframe with the following preference:

    1. Dataframe passed in by host (e.g. Dashboard) this session.
    2. In-memory session copy (st.session_state["pa_df_det"]).
    3. If active_dataset_path is set:
       - if it points to detections_normalised.csv but a detections_validated.csv
         exists in the same folder, prefer the validated file.
       - otherwise load active_dataset_path.
    4. Fallback to project files:
       - data_normalised/detections_validated.csv if it exists
       - else data_normalised/detections_normalised.csv
    """
    # 1) Host-provided df
    if isinstance(df_passed, pd.DataFrame) and not df_passed.empty:
        return df_passed.copy()

    # 2) In-memory dets for this session
    if isinstance(st.session_state.get("pa_df_det"), pd.DataFrame) and not st.session_state["pa_df_det"].empty:
        return st.session_state["pa_df_det"].copy()

    proj_root = Path(sources.get("project") or sources.get("project_root") or ".")
    dn = proj_root / "data_normalised"

    # 3) Active dataset path from session
    active_path = st.session_state.get("active_dataset_path")
    if isinstance(active_path, str) and active_path.strip():
        ap = Path(active_path)
        if ap.exists():
            # If active path is still the normalised file but a validated file exists,
            # silently upgrade to the validated file so we do not lose user work.
            if ap.name == "detections_normalised.csv":
                vpath = ap.parent / "detections_validated.csv"
                if vpath.exists():
                    try:
                        return pd.read_csv(vpath)
                    except Exception:
                        pass
            # Otherwise, just use the active path
            try:
                return pd.read_csv(ap)
            except Exception:
                pass

    # 4) Fallback: prefer validated, else normalised
    for p in [dn / "detections_validated.csv", dn / "detections_normalised.csv"]:
        try:
            if p.exists():
                return pd.read_csv(p)
        except Exception:
            continue

    return pd.DataFrame()


def _ensure_validation_ready(df_in: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise fields and derive display state from presence_label + species_name."""
    df = df_in.copy()

    # Canonical columns
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

    # File/paths
    if "path" not in df.columns and "file_path" in df.columns:
        df["path"] = df["file_path"]

    if "basename" not in df.columns:
        src = df.get("file_id", df.get("source_file", ""))
        df["basename"] = src.astype(str).map(lambda p: Path(p).name)

    if "filename_stem" not in df.columns:
        df["filename_stem"] = df["basename"].astype(str).map(lambda s: Path(s).stem.lower())

    # Start/end aliases
    if "start_s" not in df.columns and "detection_start_s" in df.columns:
        df["start_s"] = pd.to_numeric(df["detection_start_s"], errors="coerce")
    if "end_s" not in df.columns and "detection_end_s" in df.columns:
        df["end_s"] = pd.to_numeric(df["detection_end_s"], errors="coerce")

    # Probability
    if "detection_probability" not in df.columns:
        df["detection_probability"] = df.apply(_best_prob_from_row, axis=1)

    # Originals for audit/reset (only create if missing)
    if "species_name_original" not in df.columns:
        df["species_name_original"] = df["species_name"]
    if "presence_label_original" not in df.columns:
        df["presence_label_original"] = df["presence_label"]

    # Validation/admin fields
    for c, default in [
        ("validation_state", ""), ("validation_label", ""), ("validation_species", ""),
        ("validated_by", ""), ("validated_at", ""), ("validation_method", ""),
        ("user_changed", ""), ("user_changed_by", ""), ("user_changed_at", "")
    ]:
        if c not in df.columns:
            df[c] = default

    # Effective present/absent from presence_label
    pleff = df["presence_label"].astype(str).str.strip().str.lower()
    df["FinalLabelEffective"] = np.where(pleff == "present", "present", "absent")

    # Species display for UI (absent is not a species)
    sp = df["species_name"].astype(str)
    df["species_display"] = np.where(
        (df["FinalLabelEffective"] != "present") | (sp.str.strip() == ""),
        "[absent]",
        sp
    )

    return df


def _resolve_audio_path(row_or_df, df_all: pd.DataFrame) -> Optional[Path]:
    """Prefer explicit path in row/group; else resolve by filename stem elsewhere in project data."""
    if isinstance(row_or_df, pd.Series):
        rows = [row_or_df]
    else:
        rows = [row_or_df.iloc[0]] if len(row_or_df) else []

    for r in rows:
        for col in ("file_path", "path"):
            p = r.get(col)
            if isinstance(p, str) and p.strip() and Path(p).exists():
                return Path(p)

    cand_cols = [c for c in ("file_path", "path") if c in df_all.columns]
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
            if isinstance(q, str) and q.strip() and Path(q).exists():
                return Path(q)

    for col in cand_cols:
        q = rows2[col].dropna().astype(str).head(1)
        if not q.empty and Path(q.iloc[0]).exists():
            return Path(q.iloc[0])

    return None


def _decimate_mean(y: np.ndarray, sr: int, te: int) -> Tuple[np.ndarray, int]:
    """Integer time-expansion by mean decimation; preserves full duration in playback rate."""
    te = max(1, int(te))
    if te == 1 or y.size == 0:
        return y.astype(np.float32, copy=False), int(sr)
    n = (y.size // te) * te
    if n <= 0:
        return y.astype(np.float32, copy=False), int(sr)
    yy = y[:n].reshape(-1, te).mean(axis=1)
    return yy.astype(np.float32, copy=False), int(sr // te)


def _choose_te_guard(sr: int, high_hz: Optional[float]) -> int:
    """Nyquist guard; ensure downsampling avoids aliasing for given high bound."""
    if high_hz is None or not np.isfinite(high_hz) or high_hz <= 0:
        return 1
    nyq = 0.5 * sr
    limit = 0.9 * nyq
    if high_hz <= limit:
        return 1
    return int(max(1, math.ceil(high_hz / limit)))


def _estimate_peak_hz_for_group(gdf: pd.DataFrame, sr: int) -> Optional[float]:
    """Peak frequency estimate for time-expansion targeting."""
    if "detection_probability" not in gdf.columns:
        gdf = gdf.assign(detection_probability=gdf.apply(_best_prob_from_row, axis=1))

    try:
        idx = int(gdf["detection_probability"].astype(float).fillna(-1.0).idxmax())
        row = gdf.loc[idx]
    except Exception:
        row = gdf.iloc[0]

    lf = _num(row.get("low_freq"))
    hf = _num(row.get("high_freq"))
    if np.isfinite(lf) and np.isfinite(hf) and hf > lf:
        return 0.5 * (lf + hf)

    nyq = 0.5 * sr
    return min(12_000.0, 0.45 * nyq)


def _choose_te_for_peak(peak_hz: float) -> int:
    """Target ~11–12 kHz audible peak."""
    if not (isinstance(peak_hz, (int, float)) and np.isfinite(peak_hz) and peak_hz > 0):
        return 1
    te = int(max(1, round(peak_hz / 11_000.0)))
    return min(te, 16)


def _group_max_prob(gdf: pd.DataFrame) -> float:
    ps = pd.to_numeric(gdf.get("detection_probability"), errors="coerce")
    return float(ps.max()) if ps.notna().any() else -np.inf


def _tmp_audio_path(proj_root: Path, base: str, species_line: str, te: int, sr: int, n: int) -> Path:
    """Stable temp WAV path so the browser streams the full clip reliably."""
    ws = proj_root / "workspace" / "tmp_audio"
    ws.mkdir(parents=True, exist_ok=True)
    key = f"{base}|{species_line}|te={te}|sr={sr}|n={n}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return ws / f"play_{h}.wav"


def _now_iso() -> str:
    try:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    except Exception:
        return ""


def _user_name() -> str:
    return str(st.session_state.get("user_name") or os.environ.get("USER") or os.environ.get("USERNAME") or "")


# in-session overrides & filter state

def _ov_init():
    if "val_overrides" not in st.session_state:
        st.session_state["val_overrides"] = {}


def _ov_set_many(ids: List[str], state: str):
    _ov_init()
    for d in ids:
        if not d:
            continue
        st.session_state["val_overrides"][str(d)] = {"state": state, "species": "", "label": ""}


def _ov_get_state(detection_id: str) -> str:
    _ov_init()
    rec = st.session_state["val_overrides"].get(str(detection_id))
    return (rec or {}).get("state", "")


def _init_filter_state():
    defaults = {
        "validate_num_per_page": 10,
        "validate_cols_per_row": 2,
        "validate_page": 1,
        "validate_show_label": "present",
        "validate_min_prob": 0.0,
        "validate_sort_by": "probability: high → low",
        "validate_lock_freq": False,
        "validate_fmin_khz": 15.0,
        "validate_fmax_khz": 90.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _attach_validation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean flags for whether a detection has been reviewed and changed.

    Definitions:
      - changed_flag  = (species_name / presence_label differ from originals)
      - reviewed_flag = changed_flag OR validation_state_effective ∈ {correct, incorrect}
    """
    df = df.copy()

    # Originals: if missing, fall back to current values
    orig_sp = df.get("species_name_original", df.get("species_name", ""))
    orig_pl = df.get("presence_label_original", df.get("presence_label", ""))

    orig_sp = orig_sp.astype(str)
    orig_pl = orig_pl.astype(str).str.lower()

    cur_sp = df.get("species_name", "").astype(str)
    cur_pl = df.get("presence_label", "").astype(str).str.lower()

    changed_flag = (cur_sp != orig_sp) | (cur_pl != orig_pl)

    # Effective validation state
    eff = df.get("validation_state_effective")
    if eff is None:
        eff = df.get("validation_state", "")
    eff = eff.astype(str).str.lower()

    validated_flag = eff.isin(["correct", "incorrect"])

    df["changed_flag"] = changed_flag
    df["reviewed_flag"] = changed_flag | validated_flag

    return df


# page entrypoint

def render_validation(detections: Optional[pd.DataFrame], sources: dict) -> None:
    _init_filter_state()
    st.header("Validation")

    # Load the same dataset the Dashboard uses
    df_loaded = _get_active_dataset_or_load(sources, detections)
    if df_loaded is None or df_loaded.empty:
        st.warning("Validation cannot start because the analysis dataset is not initialised. Open the PAM Dashboard first.")
        return

    proj_root = Path(sources.get("project") or sources.get("project_root") or ".")
    df_all = _ensure_validation_ready(df_loaded)
    _ov_init()

    # Layout controls (persist via session_state keys)
    top1, top2, top3 = st.columns([1, 1, 1])
    with top1:
        NUM_PER_PAGE = st.number_input(
            "Spectrograms per page",
            min_value=4,
            max_value=40,
            step=2,
            key="validate_num_per_page",
        )
    with top2:
        COLS_PER_ROW = st.slider(
            "Columns per row",
            min_value=2,
            max_value=5,
            key="validate_cols_per_row",
        )
    with top3:
        PAGE = st.number_input(
            "Page",
            min_value=1,
            step=1,
            key="validate_page",
        )

    # Filters (collapsible, state persists via keys)
    with st.expander("Advanced filters", expanded=False):
        r1c1, r1c2, r1c3 = st.columns([1, 1, 1])
        with r1c1:
            show_label = st.selectbox(
                "Show clips labelled",
                ["present", "absent", "all", "user changed only"],
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
            sort_by = st.selectbox(
                "Sort by",
                ["probability: high → low", "probability: low → high", "basename"],
                key="validate_sort_by",
            )

        frow1, frow2, frow3, frow4, frow5 = st.columns([0.9, 0.7, 0.2, 0.9, 2.3])
        with frow1:
            lock_freq = st.checkbox(
                "Lock frequency (kHz)",
                key="validate_lock_freq",
            )
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
            st.caption("")
        with frow4:
            fmax_khz = st.number_input(
                "Max",
                min_value=1.0,
                max_value=250.0,
                step=1.0,
                disabled=not lock_freq,
                key="validate_fmax_khz",
            )
        with frow5:
            st.caption("")

        # group filter (species / recorder / site / detector)
        group_candidates = []
        label_map: Dict[str, str] = {}
        for label, col in [
            ("Species", "species_display"),
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
            group_label = st.selectbox(
                "Filter by group",
                group_options,
                key="validate_group_label",
            )

            if group_label != "[none]":
                group_col = label_map[group_label]
                all_vals = (
                    df_all[group_col]
                    .dropna()
                    .astype(str)
                    .sort_values()
                    .unique()
                )
                selected_vals = st.multiselect(
                    "Only show these values",
                    options=list(all_vals),
                    key="validate_group_values",
                )
                st.session_state["validate_group_col"] = group_col
            else:
                selected_vals = []
                st.session_state["validate_group_col"] = ""
        else:
            # no suitable columns to group on
            st.session_state["validate_group_col"] = ""

    # Read current group filter from session
    group_col = st.session_state.get("validate_group_col", "")
    group_values = st.session_state.get("validate_group_values", [])

    # Effective validation state with overrides
    eff_state = df_all["validation_state"].astype(str).str.lower() if "validation_state" in df_all.columns else pd.Series([""] * len(df_all))
    if "detection_id" in df_all.columns:
        ov_map = st.session_state["val_overrides"]
        if ov_map:
            det_ids = df_all["detection_id"].astype(str)
            eff_state = det_ids.map(lambda d: ov_map.get(d, {}).get("state", "") or "").where(
                eff_state.eq(""), eff_state
            )
    df_all = df_all.assign(validation_state_effective=eff_state)

    # Attach reviewed/changed flags (based on originals vs current + validation_state)
    df_all = _attach_validation_flags(df_all)

    # Filtering
    df_view = df_all.copy()

    # Apply group filter first (species / recorder / site / detector)
    if group_col and group_values:
        df_view = df_view[df_view[group_col].astype(str).isin(group_values)]

    # Label filter
    if show_label in ("present", "absent"):
        df_view = df_view[df_view["FinalLabelEffective"] == show_label]
    elif show_label == "user changed only":
        df_view = df_view[df_view["changed_flag"].astype(bool)]

    # Probability filter
    df_view["detection_probability"] = pd.to_numeric(df_view["detection_probability"], errors="coerce").fillna(0.0)
    df_view = df_view[df_view["detection_probability"] >= float(min_prob)]
    if df_view.empty:
        st.info("No clips match the current filters.")
        return

    # Validation progress summary for current filters
    total_in_scope = len(df_view)
    reviewed_mask = df_view["reviewed_flag"].astype(bool)
    changed_mask = df_view["changed_flag"].astype(bool)

    n_reviewed = int(reviewed_mask.sum())
    n_changed = int(changed_mask.sum())
    eff_local = df_view.get("validation_state_effective", df_view.get("validation_state", "")).astype(str).str.lower()
    correct_mask = eff_local.eq("correct")
    n_correct = int(correct_mask.sum())

    pct_reviewed = (100.0 * n_reviewed / total_in_scope) if total_in_scope else 0.0
    pct_correct = (100.0 * n_correct / n_reviewed) if n_reviewed else 0.0
    pct_changed = (100.0 * n_changed / n_reviewed) if n_reviewed else 0.0

    with st.expander("Validation progress (current filters)", expanded=True):
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Detections in scope", total_in_scope)
        with m2:
            st.metric("Reviewed", f"{n_reviewed} ({pct_reviewed:.0f}%)")
        with m3:
            st.metric("Marked correct", f"{n_correct} ({pct_correct:.0f}%)")
        with m4:
            st.metric("Changed of reviewed", f"{n_changed} ({pct_changed:.0f}%)")

        # Per-species breakdown
        if "species_display" in df_view.columns:
            if "detection_id" in df_view.columns:
                grp = (
                    df_view
                    .groupby("species_display", dropna=False)
                    .agg(
                        detections=("detection_id", "size"),
                        reviewed_n=("reviewed_flag", "sum"),
                        changed_n=("changed_flag", "sum"),
                    )
                )
            else:
                grp = (
                    df_view
                    .groupby("species_display", dropna=False)
                    .agg(
                        detections=("species_display", "size"),
                        reviewed_n=("reviewed_flag", "sum"),
                        changed_n=("changed_flag", "sum"),
                    )
                )

            grp["pct_reviewed"] = (100.0 * grp["reviewed_n"] / grp["detections"]).round(1)
            grp["pct_changed_of_reviewed"] = np.where(
                grp["reviewed_n"] > 0,
                (100.0 * grp["changed_n"] / grp["reviewed_n"]).round(1),
                np.nan,
            )

            st.dataframe(
                grp.reset_index().rename(columns={"species_display": "species"}).sort_values(
                    "pct_reviewed", ascending=False
                ),
                use_container_width=True,
            )

    # Grouping like dashboard: (basename, species_display)
    df_view = df_view.sort_values(["basename", "species_display", "start_s"])
    grouped = df_view.groupby(["basename", "species_display"], dropna=False)
    groups: List[tuple[str, str]] = list(grouped.indices.keys())

    # Sorting groups
    if sort_by.startswith("probability"):
        g_scores = {k: _group_max_prob(grouped.get_group(k)) for k in groups}
        reverse = (sort_by == "probability: high → low")
        groups = sorted(groups, key=lambda k: g_scores.get(k, -np.inf), reverse=reverse)
    else:
        groups = sorted(groups, key=lambda k: (k[0], k[1]))

    # Pagination
    total_cards = len(groups)
    start_idx = (int(PAGE) - 1) * int(NUM_PER_PAGE)
    end_idx = min(total_cards, start_idx + int(NUM_PER_PAGE))
    page_keys = groups[start_idx:end_idx]
    st.caption(f"Showing {len(page_keys)} of {total_cards} spectrograms (page {PAGE})")

    # Page-level bulk validate
    page_ids: List[str] = []
    for base, species_line in page_keys:
        gdf = grouped.get_group((base, species_line))
        if "detection_id" in gdf.columns:
            page_ids.extend([str(x) for x in gdf["detection_id"].astype(str).tolist()])

    bL, bR = st.columns([1, 5])
    with bL:
        if st.button("Validate all visible as correct"):
            _ov_set_many([d for d in page_ids if d], state="correct")
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()

    # Species choice list with UI "[absent]" command first
    species_choices = sorted(
        pd.unique(
            pd.concat([
                df_all.get("species_name", pd.Series([], dtype=object)).astype(str),
                df_all.get("class", pd.Series([], dtype=object)).astype(str)
            ], ignore_index=True)
        ).tolist()
    )
    species_choices = [s for s in species_choices if s and s.lower() not in ("nan", "[absent]")]
    species_choices.insert(0, "[absent]")

    sp_updates: List[Dict[str, str]] = []

    # Render spectrogram grid
    n_rows = math.ceil(len(page_keys) / int(COLS_PER_ROW))
    for r in range(n_rows):
        cols = st.columns(int(COLS_PER_ROW))
        for c in range(int(COLS_PER_ROW)):
            gi = r * int(COLS_PER_ROW) + c
            if gi >= len(page_keys):
                break

            base, species_line = page_keys[gi]
            gdf = grouped.get_group((base, species_line)).copy()

            if "detection_probability" not in gdf.columns:
                gdf["detection_probability"] = gdf.apply(_best_prob_from_row, axis=1)

            n_det = int(len(gdf))
            max_cp = _group_max_prob(gdf)
            title_html = (
                f"<div style='margin-bottom:2px'><strong>{base}</strong>"
                f"<br>{species_line}"
                f"<br>Detections: {n_det}"
            )
            if np.isfinite(max_cp):
                title_html += f"<br>Max probability: {max_cp:.2f}"
            title_html += "</div>"

            with cols[c]:
                h1, h2 = st.columns([2.0, 1.0])
                with h1:
                    st.markdown(title_html, unsafe_allow_html=True)
                with h2:
                    card_ids = [str(x) for x in gdf.get("detection_id", pd.Series([], dtype=object)).astype(str).tolist()]
                    if st.button("Validate all", key=f"bulk_spec_correct_{hash((base, species_line))}"):
                        _ov_set_many([d for d in card_ids if d], state="correct")
                        if hasattr(st, "rerun"):
                            st.rerun()
                        elif hasattr(st, "experimental_rerun"):
                            st.experimental_rerun()

                apath = _resolve_audio_path(gdf, df_all)
                if not (apath and apath.exists()):
                    st.error("Audio not found")
                    y, sr = np.array([], dtype=np.float32), 1
                else:
                    try:
                        y, sr = librosa.load(str(apath), sr=None, mono=True)
                    except Exception as e:
                        st.error(f"Audio read error: {e}")
                        y, sr = np.array([], dtype=np.float32), 1

                # Detection windows (top-10 by prob)
                boxes: List[Dict[str, float]] = []
                for _, row in gdf.iterrows():
                    b = {
                        "start_s": _num(row.get("start_s", row.get("detection_start_s"))),
                        "end_s":   _num(row.get("end_s",   row.get("detection_end_s"))),
                        "low_freq":_num(row.get("low_freq")),
                        "high_freq":_num(row.get("high_freq")),
                        "prob":    _num(row.get("detection_probability")),
                    }
                    if (np.isfinite(b["start_s"]) and np.isfinite(b["end_s"]) and b["end_s"] > b["start_s"]):
                        boxes.append(b)

                if boxes:
                    boxes = sorted(
                        boxes,
                        key=lambda b: (b["prob"] if np.isfinite(b["prob"]) else -1.0),
                        reverse=True
                    )[:10]

                if apath and y.size > 0:
                    # Frequency window
                    if lock_freq and (fmax_khz > fmin_khz):
                        ymin = max(0.0, float(fmin_khz) * 1000.0)
                        ymax = float(fmax_khz) * 1000.0
                        nyq = 0.5 * sr * 0.98
                        ymax = min(ymax, nyq)
                    else:
                        highs = [b["high_freq"] for b in boxes if np.isfinite(b["high_freq"])]
                        lows  = [b["low_freq"]  for b in boxes if np.isfinite(b["low_freq"])]
                        if highs and lows and max(highs) > min(lows):
                            fmin, fmax = min(lows), max(highs)
                        else:
                            fmin, fmax = 0.0, 0.5 * sr
                        span = max(1.0, (fmax - fmin))
                        pad = max(4_000.0, 0.30 * span)
                        nyq = 0.5 * sr * 0.98
                        ymin = max(0.0, fmin - pad)
                        ymax = min(nyq, fmax + pad)

                    # Spectrogram
                    try:
                        n_fft = 8192 if sr > 48_000 else 4096
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
                        times = np.arange(2); freqs_hz = np.arange(2)
                        S_dB = np.zeros((2, 2)); xmin, xmax = 0, 1; ymin, ymax = 0, 1

                    try:
                        fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=280, constrained_layout=False)
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
                        ax.yaxis.set_major_formatter(FuncFormatter(lambda ytick, pos: f"{ytick/1000:.0f}"))

                        # Windows + inline probability labels
                        for b in boxes:
                            x0, x1 = b["start_s"], b["end_s"]
                            prob = b["prob"]
                            ax.add_patch(
                                Rectangle(
                                    (x0, ymin),
                                    x1 - x0,
                                    ymax - ymin,
                                    facecolor=(1, 1, 1, 0.06),
                                    edgecolor=(1, 1, 1, 0.12),
                                    linewidth=0.6,
                                )
                            )
                            if np.isfinite(prob):
                                xm = (x0 + x1) * 0.5
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

                        st.pyplot(fig, use_container_width=True, clear_figure=True)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Spectrogram error: {e}")

                    # Playback TE from peak
                    try:
                        highs = [b["high_freq"] for b in boxes if np.isfinite(b["high_freq"])]
                        max_high = max(highs) if highs else None
                        te_guard = _choose_te_guard(sr, max_high)
                        peak_hz = _estimate_peak_hz_for_group(gdf, sr)
                        te_peak = _choose_te_for_peak(peak_hz)
                        te = max(te_guard, te_peak)

                        y_play, psr = _decimate_mean(y, sr, te)
                        peak = float(np.max(np.abs(y_play))) if y_play.size else 0.0
                        if peak > 0:
                            y_play = (y_play / peak * 0.98).astype(np.float32)

                        tmp_wav = _tmp_audio_path(proj_root, base, species_line, int(te), int(psr), int(y_play.size))
                        sf.write(str(tmp_wav), y_play, int(psr), format="WAV", subtype="PCM_16")
                        st.audio(str(tmp_wav))
                    except Exception as e:
                        st.error(f"Playback error: {e}")

                # In-place editor
                with st.expander("Edit detections (species & validation)"):
                    rgdf = gdf.reset_index(drop=True)
                    for ridx, row in rgdf.iterrows():
                        det_id = str(row.get("detection_id", f"{base}#{ridx}"))
                        ts = row.get("start_s", row.get("detection_start_s", np.nan))
                        ts_str = f"{float(ts):.2f}s" if np.isfinite(_num(ts)) else "—"

                        # Current species/presence -> UI choice
                        cur_sp = str(row.get("species_name", "") or "")
                        cur_pl = str(row.get("presence_label", "") or "").lower()
                        current_species_choice = "[absent]" if (cur_pl != "present" or cur_sp.strip() == "") else cur_sp
                        try:
                            idx_choice = species_choices.index(current_species_choice)
                        except ValueError:
                            idx_choice = 0  # "[absent]"

                        eff_state_row = _ov_get_state(det_id) or str(row.get("validation_state", "")).lower()
                        state_opt = ["", "correct", "incorrect"]
                        state_lbl = ["Unknown", "Correct", "Incorrect"]
                        state_idx = state_opt.index(eff_state_row) if eff_state_row in state_opt else 0

                        cc1, cc2 = st.columns([1.3, 0.9])
                        with cc1:
                            new_sp_choice = st.selectbox(
                                f"Detection {ridx+1} @ {ts_str}",
                                options=species_choices,
                                index=idx_choice,
                                key=f"sp_{base}_{species_line}_{ridx}"
                            )
                        with cc2:
                            new_state_lbl = st.selectbox(
                                "Validation",
                                state_lbl,
                                index=state_idx,
                                key=f"val_{base}_{species_line}_{ridx}"
                            )
                            new_state = state_opt[state_lbl.index(new_state_lbl)]

                        # Species change capture
                        if new_sp_choice != current_species_choice:
                            sp_updates.append({
                                "detection_id": det_id,
                                "new_species_choice": new_sp_choice
                            })

                        # Validation state overrides (in-session)
                        if new_state != eff_state_row:
                            _ov_set_many([det_id], state=new_state)

    st.divider()

    # Pending changes + save
    st.subheader("Pending species changes")

    if sp_updates:
        upd_df = (
            pd.DataFrame(sp_updates)
            .groupby("detection_id", as_index=False)
            .last()
        )
        st.dataframe(upd_df, use_container_width=True)
    else:
        upd_df = pd.DataFrame(columns=["detection_id", "new_species_choice"])
        st.write("No pending species changes.")

    left, right = st.columns([1, 3])

    def _apply_updates_and_write(det: pd.DataFrame, out_path: Path, upd_df: pd.DataFrame) -> tuple[int, int]:
        """
        Apply both:
          - species / presence changes from upd_df
          - validation_state = correct / incorrect from val_overrides

        Species changes:
          - update species_name / presence_label
          - set user_changed, user_changed_by, user_changed_at
          - DO NOT auto-assign validation_state (user does that explicitly)

        Returns:
          (n_species_changes, n_validation_flags)
        """
        det = det.copy()

        # Ensure expected columns exist
        for col in (
            "species_name", "presence_label",
            "species_name_original", "presence_label_original",
            "user_changed", "user_changed_by", "user_changed_at",
            "validation_state", "validated_by", "validated_at",
        ):
            if col not in det.columns:
                if col.endswith("_original"):
                    base = col.replace("_original", "")
                    det[col] = det.get(base, "")
                else:
                    det[col] = ""

        # detection_id -> indices
        key_to_idx: Dict[str, List[int]] = {}
        if "detection_id" in det.columns:
            for i, r in det.iterrows():
                did = str(r.get("detection_id"))
                if did and did.lower() != "nan":
                    key_to_idx.setdefault(did, []).append(i)

        user_id = st.session_state.get("user_id") or st.session_state.get("username") or _user_name()
        now_iso = _now_iso()

        # Apply species / presence changes from the pending table
        species_applied = 0
        for rec in upd_df.to_dict(orient="records"):
            detid = rec["detection_id"]
            choice = rec["new_species_choice"]
            idxs = key_to_idx.get(detid, [])
            for i in idxs:
                # Set originals if blank (original classifier output)
                if str(det.at[i, "species_name_original"]).strip() == "":
                    det.at[i, "species_name_original"] = det.at[i, "species_name"]
                if str(det.at[i, "presence_label_original"]).strip() == "":
                    det.at[i, "presence_label_original"] = det.at[i, "presence_label"]

                if choice == "[absent]":
                    new_species = ""
                    new_presence = "absent"
                else:
                    new_species = choice
                    new_presence = "present"

                prev_sp = str(det.at[i, "species_name"])
                prev_pl = str(det.at[i, "presence_label"]).lower()

                changed = (prev_sp != new_species) or (prev_pl != new_presence)

                det.at[i, "species_name"] = new_species
                det.at[i, "presence_label"] = new_presence

                if changed:
                    det.at[i, "user_changed"] = user_id or "1"
                    det.at[i, "user_changed_by"] = user_id
                    det.at[i, "user_changed_at"] = now_iso
                    species_applied += 1

        # Apply validation flags (correct / incorrect) from in-session overrides
        ov_map: Dict[str, Dict[str, str]] = st.session_state.get("val_overrides", {})
        validation_applied = 0

        for detid, ov in ov_map.items():
            state = str((ov or {}).get("state", "")).lower()
            if state not in ("correct", "incorrect"):
                continue

            idxs = key_to_idx.get(detid, [])
            for i in idxs:
                prev = str(det.at[i, "validation_state"]).lower()
                if prev == state:
                    continue  # no change

                det.at[i, "validation_state"] = state
                det.at[i, "validated_by"] = user_id
                det.at[i, "validated_at"] = now_iso
                validation_applied += 1

        # Write updated dataframe
        out_path.parent.mkdir(parents=True, exist_ok=True)
        det.to_csv(out_path, index=False)

        # Update session so Dashboard uses validated immediately
        st.session_state["active_dataset_label"] = "validated"
        st.session_state["active_dataset_path"] = str(out_path)
        st.session_state["pa_df_det"] = det  # keep in memory for immediate page reload

        # Clear pending overrides for a clean state after save
        st.session_state["val_overrides"] = {}

        return species_applied, validation_applied

    with left:
        if st.button("Save changes & validations"):
            try:
                det = df_all.copy()
                out = proj_root / "data_normalised" / "detections_validated.csv"
                n_species, n_val = _apply_updates_and_write(det, out, upd_df)
                st.success(
                    f"Applied {n_species} species change(s) and "
                    f"{n_val} validation flag(s). Saved to: {out}"
                )
                if hasattr(st, "rerun"):
                    st.rerun()
                elif hasattr(st, "experimental_rerun"):
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to save updates: {e}")

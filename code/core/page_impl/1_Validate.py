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
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or ""
    )


# Dataset loading

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

    # Originals for audit/reset
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

    # Species display current
    sp = df["species_name"].astype(str)
    df["species_display"] = np.where(
        (df["FinalLabelEffective"] != "present") | (sp.str.strip() == ""),
        "[absent]",
        sp
    )

    # Species display original (card key)
    sp0 = df["species_name_original"].astype(str)
    pl0 = df["presence_label_original"].astype(str).str.strip().str.lower()
    df["species_display_original"] = np.where(
        (pl0 != "present") | (sp0.str.strip() == ""),
        "[absent]",
        sp0
    )

    return df


# Audio path + TE helpers

def _resolve_audio_path(row_or_df, df_all: pd.DataFrame) -> Optional[Path]:
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


def _group_max_prob(gdf: pd.DataFrame) -> float:
    ps = pd.to_numeric(gdf.get("detection_probability"), errors="coerce")
    return float(ps.max()) if ps.notna().any() else -np.inf


def _tmp_audio_path(proj_root: Path, base: str, species_line: str, te: int, sr: int, n: int) -> Path:
    ws = proj_root / "workspace" / "tmp_audio"
    ws.mkdir(parents=True, exist_ok=True)
    key = f"{base}|{species_line}|te={te}|sr={sr}|n={n}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return ws / f"play_{h}.wav"


# Filter + UI state

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
        "validate_use_te_override": False,
        "validate_te_override": 10,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _card_key(base: str, species_orig: str) -> str:
    return f"{base}||{species_orig}"


def _card_change_counts(gdf: pd.DataFrame) -> Tuple[int, int]:
    if gdf.empty:
        return 0, 0
    cur_sp = gdf.get("species_name", "").astype(str)
    cur_pl = gdf.get("presence_label", "").astype(str).str.lower()
    orig_sp = gdf.get("species_name_original", cur_sp).astype(str)
    orig_pl = gdf.get("presence_label_original", cur_pl).astype(str).str.lower()
    changed = (cur_sp != orig_sp) | (cur_pl != orig_pl)
    return int(changed.sum()), int(len(gdf))


def _card_classifier_label_and_colour(changed: int, total: int, reviewed: bool) -> Tuple[str, str]:
    if total == 0:
        return "Classifier: not assessed", "#777777"
    if not reviewed:
        return "Classifier: not assessed", "#777777"
    if changed == 0:
        return "Classifier: all correct", "#2e7d32"
    if changed == total:
        return "Classifier: all incorrect", "#c62828"
    return "Classifier: mixed", "#ef6c00"


def _render_pills(gdf: pd.DataFrame):
    changed, total = _card_change_counts(gdf)
    val_state = gdf.get("validation_state", pd.Series([""] * len(gdf))).astype(str).str.lower()
    reviewed = bool(total) and val_state.replace({"nan": ""}).ne("").all()

    review_colour = "#2e7d32" if reviewed else "#777777"
    review_text = "Reviewed" if reviewed else "Not reviewed"

    cls_label, cls_colour = _card_classifier_label_and_colour(changed, total, reviewed)

    pills_html = f"""
    <div style="display:flex; gap:0.4rem; flex-wrap:wrap; justify-content:flex-end;">
      <span style="padding:0.15rem 0.55rem; border-radius:999px;
                   background-color:{review_colour}; color:white; font-size:0.72rem;">
        {review_text}
      </span>
      <span style="padding:0.15rem 0.55rem; border-radius:999px;
                   background-color:{cls_colour}; color:white; font-size:0.72rem;">
        {cls_label}
      </span>
    </div>
    """
    st.markdown(pills_html, unsafe_allow_html=True)


# Card commit logic

def _commit_card(
    proj_root: Path,
    df_all: pd.DataFrame,
    base: str,
    species_orig: str,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Apply species/presence changes for a single card and derive validation_state.

    Returns:
      updated_df, (n_changed_in_card, n_total_in_card)
    """
    det = df_all.copy()

    mask_card = (
        det["basename"].astype(str).eq(base)
        & det["species_display_original"].astype(str).eq(species_orig)
    )
    card_rows = det.loc[mask_card].copy()
    if card_rows.empty:
        return det, 0, 0

    # Stable ordering for mapping to widget keys
    card_rows = card_rows.sort_values("start_s")
    card_rows["__orig_index"] = card_rows.index
    rgdf = card_rows.reset_index(drop=True)

    user_id = st.session_state.get("user_id") or st.session_state.get("username") or _user_name()
    now_iso = _now_iso()

    changed_cnt = 0
    total_cnt = len(rgdf)

    for ridx, row in rgdf.iterrows():
        idx = int(row["__orig_index"])
        key = f"sp_{base}_{species_orig}_{ridx}"
        choice = st.session_state.get(key, None)

        # If widget not created (never opened expander), we treat as no change.
        if choice is None:
            continue

        if choice == "[absent]":
            new_species = ""
            new_presence = "absent"
        else:
            new_species = choice
            new_presence = "present"

        prev_sp = str(det.at[idx, "species_name"])
        prev_pl = str(det.at[idx, "presence_label"]).lower()

        if str(det.at[idx, "species_name_original"]).strip() == "":
            det.at[idx, "species_name_original"] = prev_sp
        if str(det.at[idx, "presence_label_original"]).strip() == "":
            det.at[idx, "presence_label_original"] = prev_pl

        det.at[idx, "species_name"] = new_species
        det.at[idx, "presence_label"] = new_presence

        changed_here = (prev_sp != new_species) or (prev_pl != new_presence)
        if changed_here:
            changed_cnt += 1
            det.at[idx, "user_changed"] = user_id or "1"
            det.at[idx, "user_changed_by"] = user_id
            det.at[idx, "user_changed_at"] = now_iso

    # After species/presence updates, recompute validation_state for card rows
    card_rows_updated = det.loc[mask_card].copy()
    card_rows_updated = card_rows_updated.sort_values("start_s")
    cur_sp = card_rows_updated["species_name"].astype(str)
    cur_pl = card_rows_updated["presence_label"].astype(str).str.lower()
    orig_sp = card_rows_updated["species_name_original"].astype(str)
    orig_pl = card_rows_updated["presence_label_original"].astype(str).str.lower()
    changed_mask = (cur_sp != orig_sp) | (cur_pl != orig_pl)

    for (i, changed_here) in zip(card_rows_updated.index, changed_mask):
        det.at[i, "validation_state"] = "incorrect" if changed_here else "correct"
        det.at[i, "validated_by"] = user_id
        det.at[i, "validated_at"] = now_iso

    return det, int(changed_mask.sum()), total_cnt


# Page entrypoint

def render_validation(detections: Optional[pd.DataFrame], sources: dict) -> None:
    _init_filter_state()
    st.header("Validation")

    proj_root = Path(sources.get("project") or sources.get("project_root") or ".")

    # Dataset selection (Original vs Validated)
    df_default, ds_label, ds_choices, ds_paths = _dataset_choice_validate(sources)
    if ds_label == "None" or df_default.empty:
        st.warning("Validation cannot start because the analysis dataset is not initialised. Ingest data first.")
        return

    ds_labels = list(ds_choices.keys())
    ds_index = ds_labels.index(ds_label) if ds_label in ds_labels else 0

    ds_col, _ = st.columns([1.4, 3])
    with ds_col:
        dataset_label = st.selectbox("Dataset", ds_labels, index=ds_index, key="validate_dataset_selector")

    if dataset_label != ds_label:
        df_default = ds_choices[dataset_label].copy()

    st.session_state["active_dataset_label"] = dataset_label
    st.session_state["active_dataset_path"] = str(ds_paths.get(dataset_label, ""))
    st.session_state["pa_df_det"] = df_default.copy()

    df_all = _ensure_validation_ready(df_default)

    # Layout controls
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

    # Advanced filters
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

        frow1, frow2, frow3, frow4, frow5 = st.columns([0.9, 0.7, 0.9, 0.9, 0.9])
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
            fmax_khz = st.number_input(
                "Max",
                min_value=1.0,
                max_value=250.0,
                step=1.0,
                disabled=not lock_freq,
                key="validate_fmax_khz",
            )
        with frow4:
            use_te_override = st.checkbox(
                "Set Time Expansion Factor",
                key="validate_use_te_override",
            )
        with frow5:
            te_override = st.number_input(
                "TE factor",
                min_value=1,
                max_value=32,
                step=1,
                key="validate_te_override",
                disabled=not use_te_override,
            )

        # Group filter
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
            st.session_state["validate_group_col"] = ""

    group_col = st.session_state.get("validate_group_col", "")
    group_values = st.session_state.get("validate_group_values", [])

    # Changed/reviewed flags
    orig_sp_all = df_all.get("species_name_original", df_all.get("species_name", "")).astype(str)
    orig_pl_all = df_all.get("presence_label_original", df_all.get("presence_label", "")).astype(str).str.lower()
    cur_sp_all = df_all.get("species_name", "").astype(str)
    cur_pl_all = df_all.get("presence_label", "").astype(str).str.lower()
    df_all["changed_flag"] = (orig_sp_all != cur_sp_all) | (orig_pl_all != cur_pl_all)

    val_state_all = df_all.get("validation_state", pd.Series([""] * len(df_all))).astype(str).str.lower()
    df_all["reviewed_flag"] = val_state_all.replace({"nan": ""}).ne("")

    # Apply filters
    df_view = df_all.copy()

    if group_col and group_values:
        df_view = df_view[df_view[group_col].astype(str).isin(group_values)]

    if show_label in ("present", "absent"):
        if show_label == "present":
            df_view = df_view[orig_pl_all.eq("present")]
        else:
            df_view = df_view[orig_pl_all.ne("present")]
    elif show_label == "user changed only":
        df_view = df_view[df_view["changed_flag"].astype(bool)]

    df_view["detection_probability"] = pd.to_numeric(df_view["detection_probability"], errors="coerce").fillna(0.0)
    df_view = df_view[df_view["detection_probability"] >= float(min_prob)]
    if df_view.empty:
        st.info("No clips match the current filters.")
        st.session_state["pa_df_det"] = df_all.copy()
        return

    # Summary metrics (detection-level)
    total_in_scope = len(df_view)
    reviewed_mask = df_view["reviewed_flag"].astype(bool)
    changed_mask = df_view["changed_flag"].astype(bool) & reviewed_mask

    val_state_local = df_view.get("validation_state", pd.Series([""] * len(df_view))).astype(str).str.lower()
    correct_mask = reviewed_mask & val_state_local.eq("correct")

    n_reviewed = int(reviewed_mask.sum())
    n_changed = int(changed_mask.sum())
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
            st.metric("Classifier correct", f"{n_correct} ({pct_correct:.0f}%)")
        with m4:
            st.metric("Changed of reviewed", f"{n_changed} ({pct_changed:.0f}%)")

        if "species_display_original" in df_view.columns:
            if "detection_id" in df_view.columns:
                grp = (
                    df_view
                    .groupby("species_display_original", dropna=False)
                    .agg(
                        detections=("detection_id", "size"),
                        reviewed_n=("reviewed_flag", "sum"),
                        changed_n=("changed_flag", "sum"),
                    )
                )
            else:
                grp = (
                    df_view
                    .groupby("species_display_original", dropna=False)
                    .agg(
                        detections=("species_display_original", "size"),
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
                grp.reset_index().rename(columns={"species_display_original": "species"}).sort_values(
                    "pct_reviewed", ascending=False
                ),
                use_container_width=True,
            )

    # Grouping: (basename, species_display_original)
    df_view = df_view.sort_values(["basename", "species_display_original", "start_s"])
    grouped = df_view.groupby(["basename", "species_display_original"], dropna=False)
    groups: List[tuple[str, str]] = list(grouped.indices.keys())

    if sort_by.startswith("probability"):
        g_scores = {k: _group_max_prob(grouped.get_group(k)) for k in groups}
        reverse = (sort_by == "probability: high → low")
        groups = sorted(groups, key=lambda k: g_scores.get(k, -np.inf), reverse=reverse)
    else:
        groups = sorted(groups, key=lambda k: (k[0], k[1]))

    total_cards = len(groups)
    start_idx = (int(PAGE) - 1) * int(NUM_PER_PAGE)
    end_idx = min(total_cards, start_idx + int(NUM_PER_PAGE))
    page_keys = groups[start_idx:end_idx]
    st.caption(f"Showing {len(page_keys)} of {total_cards} spectrograms (page {PAGE})")

    # Species dropdown choices (for editing)
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

    # Spectrogram grid
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

            n_det = int(len(gdf))
            max_cp = _group_max_prob(gdf)
            title_html = (
                f"<div style='margin-bottom:2px'><strong>{base}</strong>"
                f"<br>{species_orig}"
                f"<br>Detections: {n_det}"
            )
            if np.isfinite(max_cp):
                title_html += f"<br>Max probability: {max_cp:.2f}"
            title_html += "</div>"

            with cols[c]:
                # Header + pills + mark-reviewed button in top-right column
                h1, h2 = st.columns([2.0, 1.0])
                with h1:
                    st.markdown(title_html, unsafe_allow_html=True)
                with h2:
                    _render_pills(gdf)
                    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
                    if st.button(
                        "Mark card as reviewed",
                        key=f"mark_reviewed_{hash((base, species_orig))}",
                        use_container_width=True,
                    ):
                        updated_df, _, _ = _commit_card(proj_root, df_all, base, species_orig)
                        out = proj_root / "data_normalised" / "detections_validated.csv"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        updated_df.to_csv(out, index=False)

                        st.session_state["active_dataset_label"] = "Validated (published)"
                        st.session_state["active_dataset_path"] = str(out)
                        st.session_state["pa_df_det"] = updated_df

                        if hasattr(st, "rerun"):
                            st.rerun()
                        elif hasattr(st, "experimental_rerun"):
                            st.experimental_rerun()

                # Audio and spectrogram
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
                        times = np.arange(2)
                        freqs_hz = np.arange(2)
                        S_dB = np.zeros((2, 2))
                        xmin, xmax = 0, 1
                        ymin, ymax = 0, 1

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

                    # Playback with TE – always use full clip (no cropping)
                    try:
                        y_seg = y  # full recording

                        low_edge = _estimate_low_edge_hz_for_group(gdf)
                        te_auto = _choose_te_for_group(low_edge, sr)
                        use_te_override_flag = bool(st.session_state.get("validate_use_te_override", False))
                        if use_te_override_flag:
                            te_val = int(st.session_state.get("validate_te_override", te_auto or 1))
                            te = max(1, te_val)
                        else:
                            te = max(1, int(te_auto))

                        y_play, psr = _apply_time_expansion_for_playback(y_seg, sr, te)
                        tmp_wav = _tmp_audio_path(proj_root, base, species_orig, int(te), int(psr), int(y_play.size))
                        sf.write(str(tmp_wav), y_play, int(psr), format="WAV", subtype="PCM_16")
                        st.audio(str(tmp_wav))
                    except Exception as e:
                        st.error(f"Playback error: {e}")

                # In-place species editor
                with st.expander("Edit detections (species)"):
                    gdf_with_idx = gdf.copy()
                    gdf_with_idx["__orig_index"] = gdf_with_idx.index
                    rgdf = gdf_with_idx.reset_index(drop=True)

                    for ridx, row in rgdf.iterrows():
                        ts = row.get("start_s", row.get("detection_start_s", np.nan))
                        ts_str = f"{float(ts):.2f}s" if np.isfinite(_num(ts)) else "—"

                        cur_sp_row = str(row.get("species_name", "") or "")
                        cur_pl_row = str(row.get("presence_label", "") or "").lower()
                        current_species_choice = "[absent]" if (cur_pl_row != "present" or cur_sp_row.strip() == "") else cur_sp_row
                        try:
                            idx_choice = species_choices.index(current_species_choice)
                        except ValueError:
                            idx_choice = 0

                        st.selectbox(
                            f"Detection {ridx+1} @ {ts_str}",
                            options=species_choices,
                            index=idx_choice,
                            key=f"sp_{base}_{species_orig}_{ridx}",
                        )

    # Update in-memory copy for other pages
    st.session_state["pa_df_det"] = df_all.copy()

    # Pending changes table
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
                ] if col in df_all.columns
            ]].copy()
            st.dataframe(changed_df, use_container_width=True)
        else:
            st.write("No saved species changes yet.")
    else:
        st.write("No saved species changes yet.")

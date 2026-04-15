from __future__ import annotations

import math
import hashlib
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


def _safe_widget_key(prefix: str, *parts: object) -> str:
    s = prefix + "|" + "|".join(str(p) for p in parts)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


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

        current_presence = str(row.get("presence_label", "") or "").strip().lower()
        current_species = str(row.get("species_name", "") or "")

        choice = st.session_state.get(sp_key, None)
        if choice is None:
            choice = "[absent]" if (current_presence != "present" or current_species.strip() == "") else current_species

        if choice == "[absent]":
            out.at[idx, "species_name"] = ""
            out.at[idx, "presence_label"] = "absent"
        else:
            out.at[idx, "species_name"] = str(choice)
            out.at[idx, "presence_label"] = "present"

        current_unc = st.session_state.get(unc_key, _bool_from_any(row.get("uncertain_flag", "")))
        out.at[idx, "uncertain_flag"] = "1" if bool(current_unc) else ""

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
    with right:
        st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
        if st.button("Change strategy", key="open_validate_strategy_modal", width="stretch"):
            st.session_state["validate_strategy_modal_open"] = True


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
) -> Tuple[pd.DataFrame, int, int]:
    det = df_all.copy()

    det = _force_string_cols(det, [
        "species_name", "presence_label",
        "species_name_original", "presence_label_original",
        "validation_state", "validation_label", "validation_species",
        "validated_by", "validated_at", "validation_method",
        "user_changed", "user_changed_by", "user_changed_at",
        "uncertain_flag",
    ])

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
    changed_mask = (cur_sp != orig_sp) | (cur_pl != orig_pl)

    for i, changed_here in zip(card_rows_updated.index, changed_mask):
        current_sp = str(det.at[i, "species_name"] or "")
        current_pl = str(det.at[i, "presence_label"] or "").strip().lower()
        original_sp = str(det.at[i, "species_name_original"] or "")
        original_pl = str(det.at[i, "presence_label_original"] or "").strip().lower()

        if changed_here:
            det.at[i, "user_changed"] = user_id or "1"
            det.at[i, "user_changed_by"] = user_id
            det.at[i, "user_changed_at"] = now_iso

        is_uncertain = _bool_from_any(det.at[i, "uncertain_flag"])
        if is_uncertain:
            det.at[i, "validation_state"] = "uncertain"
        else:
            det.at[i, "validation_state"] = "incorrect" if changed_here else "correct"

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
        "validate_lock_freq": False,
        "validate_fmin_khz": 15.0,
        "validate_fmax_khz": 90.0,
        "validate_use_te_override": False,
        "validate_te_override": 10,
        "validate_use_fft_override": False,
        "validate_fft_size": 4096,
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

    review_colour = "#2e7d32" if reviewed else "#777777"
    review_text = "Reviewed" if reviewed else "Not reviewed"

    cls_label, cls_colour = _card_classifier_label_and_colour(changed, total, reviewed)

    pills_html = (
        "<div style='display:flex; gap:0.4rem; flex-wrap:wrap; justify-content:flex-end; align-items:center;'>"
        f"<span style='padding:0.15rem 0.55rem; border-radius:999px; background-color:{review_colour}; color:white; font-size:0.72rem;'>{review_text}</span>"
        f"<span style='padding:0.15rem 0.55rem; border-radius:999px; background-color:{cls_colour}; color:white; font-size:0.72rem;'>{cls_label}</span>"
    )

    if uncertain_n > 0:
        tooltip = f"{uncertain_n} uncertain detection" + ("s" if uncertain_n != 1 else "")
        pills_html += (
            f"<span title='{tooltip}' style='padding:0.10rem 0.40rem; border-radius:999px; background-color:#f9a825; color:white; font-size:0.78rem; font-weight:700;'>! {uncertain_n}</span>"
        )

    pills_html += "</div>"
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
                    st.session_state["validate_strategy_modal_open"] = False
                    st.session_state["validate_strategy_dont_auto_show"] = bool(dont_show)
                    st.session_state["validate_strategy_prompt_seen"] = True
                    _save_strategy_state(proj_root)
                    if hasattr(st, "rerun"):
                        st.rerun()
                    elif hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()

        if st.session_state.get("validate_strategy_modal_open", False):
            _strategy_dialog()

    _render_strategy_summary_bar(df_all)

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
        r1c1, r1c2 = st.columns([1, 1])
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
    df_all["changed_flag"] = (orig_sp_all != cur_sp_all) | (orig_pl_all != cur_pl_all)

    val_state_all = df_all.get("validation_state", pd.Series([""] * len(df_all))).astype(str).str.lower()
    df_all["reviewed_flag"] = val_state_all.replace({"nan": "", "<na>": ""}).ne("")
    df_all["uncertain_flag_bool"] = df_all.get(
        "uncertain_flag",
        pd.Series([""] * len(df_all), index=df_all.index)
    ).map(_bool_from_any)

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

    if goal == "find_likely_mistakes":
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
                df_all.get("class", pd.Series([], dtype=object)).astype(str)
            ], ignore_index=True)
        ).tolist()
    )
    species_choices = [s for s in species_choices if s and s.lower() not in ("nan", "[absent]")]
    species_choices.insert(0, "[absent]")

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
                f"<br>Selected detections: {n_det}"
            )
            if np.isfinite(max_cp):
                title_html += f"<br>Max probability: {max_cp:.2f}"
            title_html += "</div>"

            with cols[c]:
                h1, h2 = st.columns([2.0, 1.0])
                with h1:
                    st.markdown(title_html, unsafe_allow_html=True)
                with h2:
                    _render_pills(gdf)
                    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
                    if st.button("Mark card as reviewed", key=_safe_widget_key("mark_reviewed", base, species_orig), width="stretch"):
                        selected_indices = gdf.index.tolist()
                        updated_df, _, _ = _commit_card(
                            proj_root,
                            df_all,
                            base,
                            species_orig,
                            selected_indices=selected_indices,
                        )
                        out = proj_root / "data_normalised" / "detections_validated.csv"
                        out.parent.mkdir(parents=True, exist_ok=True)
                        updated_df.to_csv(out, index=False)

                        st.session_state["_force_validate_dataset"] = "Validated (published)"
                        st.session_state["active_dataset_label"] = "Validated (published)"
                        st.session_state["active_dataset_path"] = str(out)
                        st.session_state["pa_df_det"] = updated_df

                        if hasattr(st, "rerun"):
                            st.rerun()
                        elif hasattr(st, "experimental_rerun"):
                            st.experimental_rerun()

                apath = _resolve_audio_path(proj_root, gdf, df_all)
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
                        "end_s": _num(row.get("end_s", row.get("detection_end_s"))),
                        "low_freq": _num(row.get("low_freq")),
                        "high_freq": _num(row.get("high_freq")),
                        "prob": _num(row.get("detection_probability")),
                    }
                    if np.isfinite(b["start_s"]) and np.isfinite(b["end_s"]) and b["end_s"] > b["start_s"]:
                        boxes.append(b)

                if boxes:
                    boxes = sorted(boxes, key=lambda b: (b["prob"] if np.isfinite(b["prob"]) else -1.0), reverse=True)[:10]

                n_fft = _get_validate_n_fft(sr)
                hop = max(1, n_fft // 8)

                if apath and y.size > 0:
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

                    try:
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

                        st.pyplot(fig, width="stretch", clear_figure=True)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Spectrogram error: {e}")

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
                        tmp_wav = _tmp_audio_path(proj_root, base, species_orig, int(te), int(psr), int(y_play.size))
                        sf.write(str(tmp_wav), y_play, int(psr), format="WAV", subtype="PCM_16")
                        st.audio(str(tmp_wav))
                    except Exception as e:
                        st.error(f"Playback error: {e}")

                with st.expander("Edit detections (species)"):
                    acoustic_lookup: Dict[int, Dict[str, str]] = {}

                    if y.size > 0 and sr > 1:
                        requested_metric_fft = int(n_fft)
                        metric_fft = _card_metric_fft(
                            gdf=gdf,
                            y=y,
                            sr=sr,
                            requested_n_fft=requested_metric_fft,
                        )

                        if metric_fft is not None:
                            metric_hop = max(1, metric_fft // 8)

                            gdf_for_metrics = gdf.copy().sort_values("start_s").reset_index(drop=True)

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

                        row_left, row_right = st.columns([4.0, 1.2])

                        with row_left:
                            st.selectbox(
                                f"Detection {ridx+1} @ {ts_str}",
                                options=species_choices,
                                index=idx_choice,
                                key=f"sp_{base}_{species_orig}_{ridx}",
                            )

                        with row_right:
                            current_uncertain = _bool_from_any(row.get("uncertain_flag", ""))
                            st.checkbox(
                                "Uncertain",
                                value=current_uncertain,
                                key=f"unc_{base}_{species_orig}_{ridx}",
                            )

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
    st.write(
        "Download the currently selected dataset as a CSV file. "
        "If a validated dataset exists and is selected above, that will be exported; "
        "otherwise the current in-memory detections are exported."
    )

    user_name = (
        str(st.session_state.get("auth_user") or st.session_state.get("user_name") or "")
        or os.environ.get("USER")
        or os.environ.get("USERNAME")
        or "reviewer"
    )
    export_filename = _make_export_filename(proj_root, user_name)

    export_df = df_all.copy()

    for c in ["validation_state", "validation_label", "validation_species", "validated_by", "validated_at", "uncertain_flag"]:
        if c not in export_df.columns:
            export_df[c] = ""

    export_df = export_df.drop(columns=UNWANTED, errors="ignore")

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=export_filename,
        mime="text/csv",
    )
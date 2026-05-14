from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import os
import numpy as np
import pandas as pd

from schema import CORE_COLUMNS as PAMA_CORE, LEGACY_MAP, normalise_schema, drop_mapped_columns


RECOMMENDED: List[str] = ["recorder_id", "date_time"]


def _read_all_csvs(root: Path) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for p in Path(root).rglob("*.csv"):
        try:
            df = pd.read_csv(p, low_memory=False)
            if not df.empty:
                df["_source_csv"] = str(p)
                parts.append(df)
        except Exception:
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _safe_path_str(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ("", "nan", "none", "<na>"):
        return ""
    return s


def _safe_basename(x) -> str:
    s = _safe_path_str(x)
    return os.path.basename(s) if s else ""


def _valid_audio_filename(x) -> bool:
    s = _safe_path_str(x)
    if not s:
        return False
    return Path(s).suffix.lower() in (".wav", ".flac", ".mp3")


def _derive_audio_ref(df: pd.DataFrame) -> pd.Series:
    """
    Resolve the audio reference for each row.

    Supported pathways:
    1) New clip pathway: clip_id is the actual clip filename
    2) Legacy pathway: the CSV filename itself corresponds to the audio filename
    """
    out = pd.Series([""] * len(df), index=df.index, dtype="object")

    if "clip_id" in df.columns:
        vals = df["clip_id"].map(_safe_basename)
        need = out.eq("")
        out.loc[need] = vals.loc[need]

    if "_source_csv" in df.columns:
        def from_csv(x) -> str:
            s = _safe_path_str(x)
            if not s:
                return ""
            b = os.path.basename(s)
            return b[:-4] if b.lower().endswith(".csv") else b

        need = out.eq("")
        out.loc[need] = df.loc[need, "_source_csv"].map(from_csv)

    return out.fillna("")


def _derive_file_id(df: pd.DataFrame) -> pd.Series:
    return _derive_audio_ref(df)


def _pick_times(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    cand_pairs = [
        ("start_time", "end_time"),
        ("start_s", "end_s"),
        ("onset_s", "offset_s"),
        ("start", "end"),
    ]
    for a, b in cand_pairs:
        if a in df.columns and b in df.columns:
            return (
                pd.to_numeric(df[a], errors="coerce"),
                pd.to_numeric(df[b], errors="coerce"),
            )
    n = len(df)
    return pd.Series([np.nan] * n, index=df.index), pd.Series([np.nan] * n, index=df.index)


def _clean_batdetect_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.head(0).copy()

    refs = _derive_audio_ref(df)
    valid_mask = refs.map(_valid_audio_filename)

    bad = df.loc[~valid_mask].copy()
    good = df.loc[valid_mask].copy()

    if not bad.empty:
        bad["_ingest_drop_reason"] = "invalid_or_missing_audio_filename"

    return good, bad


def _index_audio_recursive(audio_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, str]] = []
    for root, _, names in os.walk(audio_root):
        for nm in names:
            if os.path.splitext(nm)[1].lower() in (".wav", ".flac", ".mp3"):
                full = Path(root) / nm
                rows.append({"filename": nm, "path": str(full.resolve())})
    mp = pd.DataFrame(rows)
    if mp.empty:
        return mp
    mp["_name_lc"] = mp["filename"].astype(str).str.strip().str.lower()
    mp["_stem_lc"] = mp["_name_lc"].str.replace(r"\.[^.]+$", "", regex=True)
    stem_counts = mp["_stem_lc"].value_counts()
    mp["_stem_unique"] = mp["_stem_lc"].isin(stem_counts[stem_counts == 1].index)
    return mp


def _attach_paths_by_filename(df: pd.DataFrame, audio_root: Optional[Path]) -> pd.Series:
    if not audio_root or not Path(audio_root).exists():
        return pd.Series([""] * len(df), index=df.index)
    mp = _index_audio_recursive(Path(audio_root))
    if mp.empty:
        return pd.Series([""] * len(df), index=df.index)

    name_map = dict(zip(mp["_name_lc"], mp["path"]))
    fid_lc = df["file_id"].astype(str).str.strip().str.lower()
    out = fid_lc.map(name_map)

    need = out.isna() | (out.astype(str).str.strip() == "")
    if need.any():
        uniq = mp.loc[mp["_stem_unique"], ["_stem_lc", "path"]]
        stem_map = dict(zip(uniq["_stem_lc"], uniq["path"]))
        fid_stem = fid_lc.str.replace(r"\.[^.]+$", "", regex=True)
        out.loc[need] = fid_stem.loc[need].map(stem_map)

    return out.fillna("")


def _drop_legacy_mapped_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    reverse: Dict[str, List[str]] = {}
    for leg, canon in LEGACY_MAP.items():
        reverse.setdefault(canon, []).append(leg)

    cols_lc = {c.lower(): c for c in df.columns}
    to_drop: List[str] = []

    for canon in PAMA_CORE:
        if canon not in df.columns:
            continue
        for legacy_lc in reverse.get(canon, []):
            if legacy_lc in cols_lc:
                orig = cols_lc[legacy_lc]
                if orig != canon and orig in df.columns:
                    to_drop.append(orig)

    for extra in ("start_time", "end_time"):
        if extra in df.columns and "detection_start_s" in df.columns:
            to_drop.append(extra)

    to_drop = sorted(set(to_drop))
    if to_drop:
        df = df.drop(columns=to_drop, errors="ignore")
    return df


def _finalise_order(df: pd.DataFrame) -> pd.DataFrame:
    for rec in RECOMMENDED:
        if rec not in df.columns:
            df[rec] = ""
    ordered = [c for c in PAMA_CORE if c in df.columns] + [c for c in RECOMMENDED if c in df.columns]
    rest = [c for c in df.columns if c not in set(ordered)]
    return df[ordered + rest]


def _should_rebase_to_clip_time(df: pd.DataFrame) -> pd.Series:
    has_clip_id = pd.Series(False, index=df.index)
    has_clip_start = pd.Series(False, index=df.index)

    if "clip_id" in df.columns:
        has_clip_id = df["clip_id"].map(_valid_audio_filename)

    if "clip_start_time" in df.columns:
        clip_start = pd.to_numeric(df["clip_start_time"], errors="coerce")
        has_clip_start = clip_start.notna()

    return has_clip_id & has_clip_start


def _derive_recorder_id(df: pd.DataFrame) -> pd.Series:
    out = pd.Series([""] * len(df), index=df.index, dtype="object")

    if "H2" in df.columns:
        vals = df["H2"].astype(str).fillna("").str.strip()
        need = out.eq("")
        out.loc[need] = vals.loc[need]

    if "recorder_id" in df.columns:
        vals = df["recorder_id"].astype(str).fillna("").str.strip()
        need = out.eq("")
        out.loc[need] = vals.loc[need]

    return out.fillna("")


def _derive_date_time(df: pd.DataFrame) -> pd.Series:
    """
    Preserve the input day-month order.
    Output is a stable string: YYYY-MM-DD HH:MM:SS
    """
    out = pd.Series([""] * len(df), index=df.index, dtype="object")

    if "datetime" in df.columns:
        raw = df["datetime"].astype(str).fillna("").str.strip()
        dt = pd.to_datetime(raw, errors="coerce")
        vals = dt.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
        need = out.eq("")
        out.loc[need] = vals.loc[need]

    if "date_time" in df.columns:
        raw = df["date_time"].astype(str).fillna("").str.strip()
        dt = pd.to_datetime(raw, errors="coerce", dayfirst=True)
        vals = dt.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
        need = out.eq("")
        out.loc[need] = vals.loc[need]

    return out.fillna("")


def ingest_batdetect2(
    csv_root: Path,
    audio_root: Optional[Path] = None,
    det_thresh: float = 0.50,
    class_thresh: float = 0.20,
    te_factor_default: float = 1.0,
    keep_only_present: bool = True,
    prob_source: Optional[str] = None,
    presence_rule: str = "det_or_class",
) -> pd.DataFrame:
    csv_root = Path(csv_root)
    if not csv_root.exists():
        return pd.DataFrame()

    raw = _read_all_csvs(csv_root)
    if raw.empty:
        return raw

    for col in ("det_prob", "class_prob", "score", "probability"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    df = raw.copy()

    df, dropped_bad_rows = _clean_batdetect_rows(df)
    if df.empty:
        return pd.DataFrame()

    # Preserve metadata early from raw rows
    recorder_id_raw = _derive_recorder_id(df)
    date_time_raw = _derive_date_time(df)

    # file_id
    df["file_id"] = _derive_file_id(df)

    # times
    s_native, e_native = _pick_times(df)
    if "te_factor" in df.columns:
        te = pd.to_numeric(df["te_factor"], errors="coerce").replace(0, np.nan).fillna(float(te_factor_default))
    else:
        te = pd.Series(float(te_factor_default), index=df.index)

    s_real = (s_native / te).astype(float)
    e_real = (e_native / te).astype(float)

    rebase_mask = _should_rebase_to_clip_time(df)
    if "clip_start_time" in df.columns:
        clip_start = pd.to_numeric(df["clip_start_time"], errors="coerce")
    else:
        clip_start = pd.Series(np.nan, index=df.index)

    df["detection_start_s"] = s_real
    df["detection_end_s"] = e_real

    df.loc[rebase_mask, "detection_start_s"] = s_real.loc[rebase_mask] - clip_start.loc[rebase_mask]
    df.loc[rebase_mask, "detection_end_s"] = e_real.loc[rebase_mask] - clip_start.loc[rebase_mask]

    # species_name
    if "class" in df.columns:
        df["species_name"] = df["class"].astype(str)
    elif "species" in df.columns:
        df["species_name"] = df["species"].astype(str)

    # probability
    if prob_source in {"det_prob", "class_prob", "score", "probability"} and prob_source in df.columns:
        df["detection_probability"] = pd.to_numeric(df[prob_source], errors="coerce")
    else:
        for k in ("det_prob", "class_prob", "score", "probability"):
            if k in df.columns:
                df["detection_probability"] = pd.to_numeric(df[k], errors="coerce")
                break
        else:
            df["detection_probability"] = np.nan

    df["detection_probability"] = df["detection_probability"].clip(lower=0.0, upper=1.0)

    # presence
    det_p = df.get("det_prob")
    cls_p = df.get("class_prob")
    det_pass = det_p.ge(float(det_thresh)) if det_p is not None else pd.Series(False, index=df.index)
    cls_pass = cls_p.ge(float(class_thresh)) if cls_p is not None else pd.Series(False, index=df.index)

    if presence_rule == "det_only":
        present_mask = det_pass.fillna(False)
    elif presence_rule == "class_only":
        present_mask = cls_pass.fillna(False)
    else:
        present_mask = (det_pass | cls_pass).fillna(False)

    df["presence_label"] = np.where(present_mask, "present", "absent")
    if keep_only_present:
        df = df.loc[df["presence_label"] == "present"].copy()
        recorder_id_raw = recorder_id_raw.loc[df.index]
        date_time_raw = date_time_raw.loc[df.index]

    # file paths
    df["file_path"] = _attach_paths_by_filename(df, audio_root)

    # detection_id
    def _det_id(row) -> str:
        f = str(row.get("file_id", ""))
        s = _safe_float(row.get("detection_start_s"), np.nan)
        e = _safe_float(row.get("detection_end_s"), np.nan)
        if np.isnan(s) or np.isnan(e):
            return f"{f}:nan-nan"
        return f"{f}:{s:.3f}-{e:.3f}"

    df["detection_id"] = df.apply(_det_id, axis=1)

    # normalise core schema
    df = normalise_schema(df, build_detection_id=True)

    # Drop only raw columns that truly map into core schema.
    mapped_sources: List[str] = []

    for cand in ("clip_id",):
        if cand in raw.columns:
            mapped_sources.append(cand)

    for cand in ("start_time", "end_time", "start_s", "end_s", "onset_s", "offset_s", "start", "end", "clip_start_time", "clip_end_time"):
        if cand in raw.columns:
            mapped_sources.append(cand)

    for cand in ("class", "species"):
        if cand in raw.columns:
            mapped_sources.append(cand)

    for cand in ("det_prob", "class_prob", "score", "probability"):
        if cand in raw.columns:
            mapped_sources.append(cand)

    df = drop_mapped_columns(df, mapped_sources)
    df = _drop_legacy_mapped_columns(df)

    # Reapply recommended metadata after cleanup so it cannot be dropped.
    df["recorder_id"] = recorder_id_raw.reindex(df.index).fillna("").astype(str)
    df["date_time"] = date_time_raw.reindex(df.index).fillna("").astype(str)

    df = _finalise_order(df)

    if not dropped_bad_rows.empty:
        print(f"[batdetect2] Dropped {len(dropped_bad_rows)} row(s) with invalid or missing audio filename values.")

    return df
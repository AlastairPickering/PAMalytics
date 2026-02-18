# scripts/adapters/batdetect2.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import os
import numpy as np
import pandas as pd

# Single source of truth
from pamalytics.scripts.schema import CORE_COLUMNS as PAMA_CORE, LEGACY_MAP, normalise_schema, drop_mapped_columns


RECOMMENDED: List[str] = ["recorder_id", "date_time"]


# Helpers
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


def _derive_file_id(df: pd.DataFrame) -> pd.Series:
    for cand in ("wav_filename", "filename", "audio_filename", "source_file", "file", "name"):
        if cand in df.columns:
            return df[cand].astype(str).map(lambda s: os.path.basename(s))
    def from_csv(s: str) -> str:
        b = os.path.basename(s)
        return b[:-4] if b.lower().endswith(".csv") else b
    return df.get("_source_csv", "").astype(str).map(from_csv)


def _pick_times(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    CAND_PAIRS = [
        ("start_time", "end_time"), 
        ("start_s", "end_s"),
        ("onset_s", "offset_s"),
        ("start", "end"),
    ]
    for a, b in CAND_PAIRS:
        if a in df.columns and b in df.columns:
            return (pd.to_numeric(df[a], errors="coerce"),
                    pd.to_numeric(df[b], errors="coerce"))
    n = len(df)
    return pd.Series([np.nan] * n), pd.Series([np.nan] * n)


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

    # Reverse map: canonical -> [legacy names]
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

    # BD2 extras
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


# Main entry
def ingest_batdetect2(
    csv_root: Path,
    audio_root: Optional[Path] = None,
    det_thresh: float = 0.50,
    class_thresh: float = 0.20,
    te_factor_default: float = 1.0,
    keep_only_present: bool = True,
    prob_source: Optional[str] = None,       # None|'det_prob'|'class_prob'|'score'|'probability'
    presence_rule: str = "det_or_class",     # 'det_or_class'|'det_only'|'class_only'
) -> pd.DataFrame:
    """
    Mapping controls:
      - prob_source: choose which column becomes canonical detection_probability.
                     If None, prefer in order det_prob > class_prob > score > probability.
      - presence_rule: pick which fields define 'present'.
    """
    csv_root = Path(csv_root)
    if not csv_root.exists():
        return pd.DataFrame()

    raw = _read_all_csvs(csv_root)
    if raw.empty:
        return raw

    # Ensure numeric
    for col in ("det_prob", "class_prob", "score", "probability"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    df = raw.copy()

    # file_id
    df["file_id"] = _derive_file_id(df)

    # times (native) -> real seconds via TE
    s_native, e_native = _pick_times(df)
    if "te_factor" in df.columns:
        te = pd.to_numeric(df["te_factor"], errors="coerce").replace(0, np.nan).fillna(float(te_factor_default))
    else:
        te = pd.Series(float(te_factor_default), index=df.index)
    df["detection_start_s"] = (s_native / te).astype(float)
    df["detection_end_s"]   = (e_native / te).astype(float)

    # species_name
    if "class" in df.columns:
        df["species_name"] = df["class"].astype(str)
    elif "species" in df.columns:
        df["species_name"] = df["species"].astype(str)

    # detection_probability mapping
    if prob_source in {"det_prob", "class_prob", "score", "probability"} and prob_source in df.columns:
        df["detection_probability"] = pd.to_numeric(df[prob_source], errors="coerce")
    else:
        # auto preference
        for k in ("det_prob", "class_prob", "score", "probability"):
            if k in df.columns:
                df["detection_probability"] = pd.to_numeric(df[k], errors="coerce")
                break
        else:
            df["detection_probability"] = np.nan
    # clamp to [0,1]
    df["detection_probability"] = df["detection_probability"].clip(lower=0.0, upper=1.0)

    # presence_label according to presence_rule
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

    # file_path via audio_root (exact filename + unique-stem fallback)
    df["file_path"] = _attach_paths_by_filename(df, audio_root)

    # detection_id
    def _det_id(row) -> str:
        f = str(row.get("file_id", ""))
        s = _safe_float(row.get("detection_start_s"), np.nan)
        e = _safe_float(row.get("detection_end_s"),   np.nan)
        if np.isnan(s) or np.isnan(e):
            return f"{f}:nan-nan"
        return f"{f}:{s:.3f}-{e:.3f}"
    df["detection_id"] = df.apply(_det_id, axis=1)

    # Normalise (types, missing cores, label lower-casing, id backfill)
    df = normalise_schema(df, build_detection_id=True)

    # Columns that fed into the canonical fields for BD2
    mapped_sources = []

    # file id sources
    for cand in ("wav_filename", "filename", "audio_filename", "source_file", "file", "name"):
        if cand in raw.columns:
            mapped_sources.append(cand)

    # time sources
    for cand in ("start_time", "end_time", "start_s", "end_s", "onset_s", "offset_s", "start", "end"):
        if cand in raw.columns:
            mapped_sources.append(cand)

    # species / class sources
    for cand in ("class", "species"):
        if cand in raw.columns:
            mapped_sources.append(cand)

    # probability / score sources
    for cand in ("det_prob", "class_prob", "score", "probability"):
        if cand in raw.columns:
            mapped_sources.append(cand)

    # Drop mapped sources exactly like the manual path
    df = drop_mapped_columns(df, mapped_sources)

    # Then your existing duplicate-cleanup + ordering
    df = _drop_legacy_mapped_columns(df)
    df = _finalise_order(df)
    return df


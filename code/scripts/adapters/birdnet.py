# scripts/adapters/birdnet.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import os
import numpy as np
import pandas as pd

from schema import CORE_COLUMNS as PAMA_CORE, LEGACY_MAP, normalise_schema, drop_mapped_columns

RECOMMENDED: List[str] = ["recorder_id", "date_time"]


# Helpers
def _read_all_csvs(root: Path) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(root, low_memory=False)
            if not df.empty:
                df["_source_csv"] = str(root)
                parts.append(df)
        except Exception:
            pass
    else:
        for p in root.rglob("*.csv"):
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
    """BirdNET: prefer basename of the 'file' column."""
    if "file" in df.columns:
        return df["file"].astype(str).map(os.path.basename)
    # Fallbacks, just in case
    for cand in ("wav_filename", "filename", "audio_filename", "source_file", "name"):
        if cand in df.columns:
            return df[cand].astype(str).map(os.path.basename)

    def from_csv(s: str) -> str:
        b = os.path.basename(s)
        return b[:-4] if b.lower().endswith(".csv") else b

    return df.get("_source_csv", "").astype(str).map(from_csv)


def _time_to_seconds(col: pd.Series) -> pd.Series:
    """
    Convert BirdNET start_time/end_time to seconds.
    Handles numeric seconds directly, and simple 'HH:MM:SS.ss' or 'MM:SS.ss' strings.
    """
    s = col
    # Try numeric first
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.8:
        return num

    def _parse_one(x):
        if isinstance(x, (int, float)):
            return float(x)
        try:
            txt = str(x).strip()
            if not txt:
                return np.nan
            if ":" not in txt:
                return float(txt)
            parts = [float(p) for p in txt.split(":")]
            if len(parts) == 3:
                h, m, sec = parts
                return h * 3600.0 + m * 60.0 + sec
            if len(parts) == 2:
                m, sec = parts
                return m * 60.0 + sec
            if len(parts) == 1:
                return parts[0]
        except Exception:
            return np.nan
        return np.nan

    return s.apply(_parse_one)


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
        # fall back to the original BirdNET 'file' column if present
        if "file" in df.columns:
            return df["file"].astype(str)
        return pd.Series([""] * len(df), index=df.index)

    mp = _index_audio_recursive(Path(audio_root))
    if mp.empty:
        if "file" in df.columns:
            return df["file"].astype(str)
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

    out = out.fillna("")
    # fall back to BirdNET 'file' path for any remaining blanks
    if "file" in df.columns:
        src = df["file"].astype(str)
        missing = out.astype(str).str.strip().eq("")
        out.loc[missing] = src.loc[missing]
    return out


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

    # BirdNET-specific time aliases
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
def ingest_birdnet(
    csv_root: Path,
    audio_root: Optional[Path] = None,
    min_conf: float = 0.0,
    keep_only_present: bool = True,
) -> pd.DataFrame:
    """
    Ingest BirdNET CSVs with columns:
      common_name, scientific_name, start_time, end_time, confidence, label, file

    Mapping:
      - file_id               <- basename(file)
      - detection_start_s     <- start_time (seconds or HH:MM:SS)
      - detection_end_s       <- end_time   (seconds or HH:MM:SS)
      - species_name          <- scientific_name (fallback: common_name)
      - detection_probability <- confidence (clipped to [0,1])
      - presence_label        <- confidence >= min_conf ? 'present' : 'absent'
      - file_path             <- resolved via audio_root, fallback to file
    """
    csv_root = Path(csv_root)
    if not csv_root.exists():
        return pd.DataFrame()

    raw = _read_all_csvs(csv_root)
    if raw.empty:
        return raw

    df = raw.copy()

    # file_id from basename of the BirdNET file path
    df["file_id"] = _derive_file_id(df)

    # times -> seconds
    if "start_time" in df.columns:
        df["detection_start_s"] = _time_to_seconds(df["start_time"])
    else:
        df["detection_start_s"] = np.nan

    if "end_time" in df.columns:
        df["detection_end_s"] = _time_to_seconds(df["end_time"])
    else:
        df["detection_end_s"] = np.nan

    # species_name
    if "scientific_name" in df.columns:
        df["species_name"] = df["scientific_name"].astype(str)
    elif "common_name" in df.columns:
        df["species_name"] = df["common_name"].astype(str)

    # detection_probability from confidence
    if "confidence" in df.columns:
        df["detection_probability"] = pd.to_numeric(df["confidence"], errors="coerce")
    else:
        df["detection_probability"] = np.nan
    df["detection_probability"] = df["detection_probability"].clip(lower=0.0, upper=1.0)

    # presence_label
    if "detection_probability" in df.columns:
        present_mask = df["detection_probability"].ge(float(min_conf))
    else:
        present_mask = pd.Series(True, index=df.index)

    df["presence_label"] = np.where(present_mask, "present", "absent")

    if keep_only_present:
        df = df.loc[df["presence_label"] == "present"].copy()

    # file_path via audio_root (exact filename + unique-stem fallback, then fallback to BirdNET file path)
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

    # BirdNET raw columns used to build canonical fields
    mapped_sources = []

    # file id sources
    for cand in ("file", "wav_filename", "filename", "audio_filename", "source_file", "name"):
        if cand in raw.columns:
            mapped_sources.append(cand)

    # time sources
    for cand in ("start_time", "end_time"):
        if cand in raw.columns:
            mapped_sources.append(cand)

    # species sources
    for cand in ("scientific_name", "common_name"):
        if cand in raw.columns:
            mapped_sources.append(cand)

    # probability / label sources
    if "confidence" in raw.columns:
        mapped_sources.append("confidence")
    if "label" in raw.columns:
        mapped_sources.append("label")

    # Drop mapped sources using the shared helper
    df = drop_mapped_columns(df, mapped_sources)

    # Existing BirdNET-specific duplicate drop + ordering
    df = _drop_legacy_mapped_columns(df)
    df = _finalise_order(df)
    return df


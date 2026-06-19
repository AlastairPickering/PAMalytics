# scripts/adapters/birdnet.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import os
import re
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



def _clean_identity_value(x) -> str:
    
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return s


def _key_text(x) -> str:
    return _clean_identity_value(x).replace("\\", "/").lower()


def _choose_single_path(matches: pd.DataFrame) -> str:
    if matches is None or matches.empty:
        return ""
    paths = matches["path"].dropna().astype(str).drop_duplicates()
    return paths.iloc[0] if len(paths) == 1 else ""


def _path_suffix_candidates(rel_folder: str, audio_name: str) -> List[str]:
    rel_folder = _key_text(rel_folder).strip("/")
    audio_name = Path(_clean_identity_value(audio_name)).name.lower()
    if not audio_name:
        return []
    out: List[str] = []
    if rel_folder:
        parts = [p for p in rel_folder.split("/") if p]
        for i in range(len(parts)):
            out.append("/".join(parts[i:] + [audio_name]))
    out.append(audio_name)
    seen = set()
    keep = []
    for item in out:
        if item and item not in seen:
            seen.add(item)
            keep.append(item)
    return keep



def _sqlite_query_paths(index_db: Path, sql: str, params: Tuple[object, ...]) -> List[str]:
    import sqlite3
    if not index_db or not Path(index_db).exists():
        return []
    try:
        with sqlite3.connect(str(index_db)) as conn:
            rows = conn.execute(sql, params).fetchall()
        return list(dict.fromkeys([str(r[0]) for r in rows if r and _clean_identity_value(r[0])]))
    except Exception:
        return []


def _match_audio_paths_sqlite(index_db: Path, raw_value: str, source_csv: str = "", csv_base: Optional[Path] = None) -> List[str]:
    raw = _clean_identity_value(raw_value)
    if not raw or not index_db or not Path(index_db).exists():
        return []

    raw_rel_lc = _key_text(raw).strip("/")
    raw_name_lc = Path(raw).name.lower()
    raw_stem_lc = re.sub(r"\.[^.]+$", "", raw_name_lc)

    if raw.startswith("\\\\") or raw.startswith("//") or (len(raw) >= 3 and raw[1] == ":" and raw[2] in ("\\", "/")) or Path(raw).is_absolute():
        m = _sqlite_query_paths(index_db, "SELECT path FROM audio_files WHERE path_lc = ?", (raw.replace("\\", "/").lower(),))
        if m:
            return m

    m = _sqlite_query_paths(index_db, "SELECT path FROM audio_files WHERE rel_lc = ?", (raw_rel_lc,))
    if m:
        return m

    if "/" in raw_rel_lc:
        m = _sqlite_query_paths(index_db, "SELECT path FROM audio_files WHERE rel_lc = ? OR rel_lc LIKE ?", (raw_rel_lc, "%/" + raw_rel_lc))
        if m:
            return m

    if source_csv and csv_base is not None:
        try:
            source_folder = Path(str(source_csv)).expanduser().resolve().parent
            try:
                rel_folder = source_folder.relative_to(csv_base).as_posix()
            except Exception:
                rel_folder = source_folder.name
            for cand in _path_suffix_candidates(rel_folder, raw_name_lc):
                m = _sqlite_query_paths(index_db, "SELECT path FROM audio_files WHERE rel_lc = ? OR rel_lc LIKE ?", (cand, "%/" + cand))
                if m:
                    return m
        except Exception:
            pass

    m = _sqlite_query_paths(index_db, "SELECT path FROM audio_files WHERE filename_lc = ?", (raw_name_lc,))
    if m:
        return m

    if raw_stem_lc:
        m = _sqlite_query_paths(index_db, "SELECT path FROM audio_files WHERE stem_lc = ?", (raw_stem_lc,))
        if m:
            return m

    return []


def _match_audio_row(mp: pd.DataFrame, raw_value: str, source_csv: str = "", csv_base: Optional[Path] = None) -> str:
    raw = _clean_identity_value(raw_value)
    if not raw:
        return ""
    raw_path_lc = str(Path(raw).expanduser()).lower()
    raw_rel_lc = _key_text(raw)
    raw_name_lc = Path(raw).name.lower()
    raw_stem_lc = re.sub(r"\.[^.]+$", "", raw_name_lc)

    exact = mp.loc[mp["path_lc"].eq(raw_path_lc)]
    hit = _choose_single_path(exact)
    if hit:
        return hit

    exact_rel = mp.loc[mp["rel_lc"].eq(raw_rel_lc)]
    hit = _choose_single_path(exact_rel)
    if hit:
        return hit

    if "/" in raw_rel_lc:
        suffix_rel = mp.loc[mp["rel_lc"].eq(raw_rel_lc) | mp["rel_lc"].str.endswith("/" + raw_rel_lc)]
        hit = _choose_single_path(suffix_rel)
        if hit:
            return hit

    if source_csv and csv_base is not None:
        try:
            source_folder = Path(str(source_csv)).expanduser().resolve().parent
            try:
                rel_folder = source_folder.relative_to(csv_base).as_posix()
            except Exception:
                rel_folder = source_folder.name
            for cand in _path_suffix_candidates(rel_folder, raw_name_lc):
                m = mp.loc[mp["rel_lc"].eq(cand) | mp["rel_lc"].str.endswith("/" + cand)]
                hit = _choose_single_path(m)
                if hit:
                    return hit
        except Exception:
            pass

    same_name = mp.loc[mp["_name_lc"].eq(raw_name_lc)]
    hit = _choose_single_path(same_name)
    if hit:
        return hit

    same_stem = mp.loc[mp["_stem_lc"].eq(raw_stem_lc)]
    hit = _choose_single_path(same_stem)
    if hit:
        return hit

    return ""


def _make_file_key(row) -> str:
    for c in ("file_path", "file_path_original", "file_id", "file", "_source_csv"):
        v = _clean_identity_value(row.get(c, ""))
        if v:
            return _key_text(v)
    return ""


def _make_detection_id(row) -> str:
    f = _make_file_key(row)
    s = _safe_float(row.get("detection_start_s"), np.nan)
    e = _safe_float(row.get("detection_end_s"), np.nan)
    species = _key_text(row.get("species_name", ""))
    if np.isnan(s) or np.isnan(e):
        base = f"{f}:nan-nan:{species}"
    else:
        base = f"{f}:{s:.3f}-{e:.3f}:{species}"
    return base


def _ensure_unique_detection_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "detection_id" not in df.columns:
        return df
    d = df["detection_id"].astype(str)
    n = d.groupby(d).cumcount()
    dup = d.duplicated(keep=False)
    if dup.any():
        df.loc[dup, "detection_id"] = d.loc[dup] + ":" + n.loc[dup].astype(str)
    return df

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
    audio_root = Path(audio_root).expanduser().resolve()

    for root, _, names in os.walk(audio_root):
        for nm in names:
            if os.path.splitext(nm)[1].lower() in (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".aif", ".aiff"):
                full = (Path(root) / nm).resolve()
                try:
                    rel = full.relative_to(audio_root).as_posix()
                except Exception:
                    rel = full.name

                rows.append({
                    "filename": nm,
                    "path": str(full),
                    "path_lc": str(full).lower(),
                    "rel_lc": rel.lower(),
                })

    mp = pd.DataFrame(rows)

    if mp.empty:
        return mp

    mp["_name_lc"] = mp["filename"].astype(str).str.strip().str.lower()
    mp["_stem_lc"] = mp["_name_lc"].str.replace(r"\.[^.]+$", "", regex=True)

    name_counts = mp["_name_lc"].value_counts()
    stem_counts = mp["_stem_lc"].value_counts()
    rel_counts = mp["rel_lc"].value_counts()

    mp["_name_unique"] = mp["_name_lc"].isin(name_counts[name_counts == 1].index)
    mp["_stem_unique"] = mp["_stem_lc"].isin(stem_counts[stem_counts == 1].index)
    mp["_rel_unique"] = mp["rel_lc"].isin(rel_counts[rel_counts == 1].index)

    return mp

def _unique_paths(matches: pd.DataFrame) -> List[str]:
    if matches is None or matches.empty or "path" not in matches.columns:
        return []
    return matches["path"].dropna().astype(str).drop_duplicates().tolist()


def _match_audio_paths(mp: pd.DataFrame, raw_value: str, source_csv: str = "", csv_base: Optional[Path] = None) -> List[str]:
    raw = _clean_identity_value(raw_value)
    if not raw:
        return []
    raw_path_lc = str(Path(raw).expanduser()).lower()
    raw_rel_lc = _key_text(raw)
    raw_name_lc = Path(raw).name.lower()
    raw_stem_lc = re.sub(r"\.[^.]+$", "", raw_name_lc)

    exact = _unique_paths(mp.loc[mp["path_lc"].eq(raw_path_lc)])
    if exact:
        return exact

    exact_rel = _unique_paths(mp.loc[mp["rel_lc"].eq(raw_rel_lc)])
    if exact_rel:
        return exact_rel

    if "/" in raw_rel_lc:
        suffix_rel = _unique_paths(mp.loc[mp["rel_lc"].eq(raw_rel_lc) | mp["rel_lc"].str.endswith("/" + raw_rel_lc)])
        if suffix_rel:
            return suffix_rel

    if source_csv and csv_base is not None:
        try:
            source_folder = Path(str(source_csv)).expanduser().resolve().parent
            try:
                rel_folder = source_folder.relative_to(csv_base).as_posix()
            except Exception:
                rel_folder = source_folder.name
            for cand in _path_suffix_candidates(rel_folder, raw_name_lc):
                m = _unique_paths(mp.loc[mp["rel_lc"].eq(cand) | mp["rel_lc"].str.endswith("/" + cand)])
                if m:
                    return m
        except Exception:
            pass

    same_name = _unique_paths(mp.loc[mp["_name_lc"].eq(raw_name_lc)])
    if same_name:
        return same_name

    if raw_stem_lc:
        same_stem = _unique_paths(mp.loc[mp["_stem_lc"].eq(raw_stem_lc)])
        if same_stem:
            return same_stem

    return []


def _expand_rows_by_audio_matches(
    df: pd.DataFrame,
    audio_root: Optional[Path],
    csv_root: Optional[Path] = None,
    audio_index_db: Optional[Path] = None,
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["file_path"] = ""
        return out

    use_sqlite = bool(audio_index_db and Path(audio_index_db).exists())
    mp = None
    if not use_sqlite:
        if not audio_root or not Path(audio_root).exists():
            out = df.copy()
            out["file_path"] = ""
            return out
        mp = _index_audio_recursive(Path(audio_root))
        if mp.empty:
            out = df.copy()
            out["file_path"] = ""
            return out

    raw_file = df["file"].astype(str).str.strip() if "file" in df.columns else df["file_id"].astype(str).str.strip()

    csv_base = None
    if csv_root is not None:
        try:
            csv_root_p = Path(csv_root).expanduser().resolve()
            csv_base = csv_root_p.parent if csv_root_p.is_file() else csv_root_p
        except Exception:
            csv_base = None

    rows = []
    cache: Dict[Tuple[str, str], List[str]] = {}
    for idx, row in df.iterrows():
        source_csv = str(row.get("_source_csv", ""))
        raw_value = str(raw_file.at[idx])
        key = (raw_value, source_csv)
        if key not in cache:
            if use_sqlite:
                cache[key] = _match_audio_paths_sqlite(Path(audio_index_db), raw_value, source_csv=source_csv, csv_base=csv_base)
            else:
                cache[key] = _match_audio_paths(mp, raw_value, source_csv=source_csv, csv_base=csv_base)
        matches = cache[key]
        if matches:
            for path in matches:
                r = row.copy()
                r["file_path"] = path
                rows.append(r)
        else:
            r = row.copy()
            r["file_path"] = ""
            rows.append(r)

    return pd.DataFrame(rows).reset_index(drop=True)


def _attach_paths_by_filename(
    df: pd.DataFrame,
    audio_root: Optional[Path],
    csv_root: Optional[Path] = None,
) -> pd.Series:
    out = pd.Series([""] * len(df), index=df.index)

    if not audio_root or not Path(audio_root).exists():
        return out

    mp = _index_audio_recursive(Path(audio_root))
    if mp.empty:
        return out

    raw_file = df["file"].astype(str).str.strip() if "file" in df.columns else df["file_id"].astype(str).str.strip()

    csv_base = None
    if csv_root is not None:
        try:
            csv_root_p = Path(csv_root).expanduser().resolve()
            csv_base = csv_root_p.parent if csv_root_p.is_file() else csv_root_p
        except Exception:
            csv_base = None

    cache: Dict[Tuple[str, str], str] = {}
    for idx in df.index:
        source_csv = str(df.at[idx, "_source_csv"]) if "_source_csv" in df.columns else ""
        raw_value = str(raw_file.at[idx])
        key = (raw_value, source_csv)
        if key not in cache:
            cache[key] = _match_audio_row(mp, raw_value, source_csv=source_csv, csv_base=csv_base)
        out.at[idx] = cache[key]

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
    audio_index_db: Optional[Path] = None,
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

    df = _expand_rows_by_audio_matches(df, audio_root, csv_root=csv_root, audio_index_db=audio_index_db)
    df["file_key"] = df.apply(_make_file_key, axis=1)
    df["detection_id"] = df.apply(_make_detection_id, axis=1)

    # Normalise (types, missing cores, label lower-casing, id backfill)
    df = normalise_schema(df, build_detection_id=False)
    df["file_key"] = df.apply(_make_file_key, axis=1)
    df["detection_id"] = df.apply(_make_detection_id, axis=1)
    df = _ensure_unique_detection_ids(df)

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


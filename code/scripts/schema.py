# scripts/schema.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional
import pandas as pd
import numpy as np

# Canonical PAMalytics columns
CORE_COLUMNS: Iterable[str] = (
    "file_id",                 # filename OR path string from the detector
    "file_path",               # absolute path to audio (filled later by audio map; may be empty at ingest)
    "detection_id",            # unique id per detection (built if missing)
    "detection_start_s",       # float seconds
    "detection_end_s",         # float seconds
    "presence_label",          # "present" / "absent" (verbatim, lower-case)
    "species_name",            # string (classifier’s class/species label)
    "detection_probability",   # float in [0,1] (best-effort)
)

# Legacy → canonical name candidates (first match wins)
LEGACY_MAP: Dict[str, str] = {
    # file id
    "source_file":           "file_id",
    "filename":              "file_id",
    "file":                  "file_id",
    "filepath":              "file_id",
    "path_in_results":       "file_id",

    # path to audio (optional at ingest)
    "path":                  "file_path",
    "audio_path":            "file_path",

    # times
    "start":                 "detection_start_s",
    "start_s":               "detection_start_s",
    "start_time_s":          "detection_start_s",
    "begin":                 "detection_start_s",
    "onset":                 "detection_start_s",

    "end":                   "detection_end_s",
    "end_s":                 "detection_end_s",
    "end_time_s":            "detection_end_s",
    "offset":                "detection_end_s",
    "duration":              "detection_end_s",      
    "duration_s":            "detection_end_s",

    # label / presence
    "label":                 "presence_label",       # adapters should already canonicalise to present/absent
    "presence":              "presence_label",
    "present":               "presence_label",

    # class/species
    "class":                 "species_name",
    "species":               "species_name",

    # probabilities / scores
    "prob":                  "detection_probability",
    "probability":           "detection_probability",
    "score":                 "detection_probability",
    "class_prob":            "detection_probability",
    "det_prob":              "detection_probability",

    # pre-existing id
    "id":                    "detection_id",
    "detection_uuid":        "detection_id",
}

def _lc_map(cols: Iterable[str]) -> Dict[str, str]:
    """Lowercase → original name lookup."""
    return {c.lower(): c for c in cols}

def _first_available(cols_lc: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in cols_lc:
            return cols_lc[c]
    return None

def normalise_schema(
    df_in: pd.DataFrame,
    *,
    build_detection_id: bool = True,
) -> pd.DataFrame:
    """
    Return a copy of df_in with PAMalytics’ canonical columns ensured:
      - Renames legacy columns to canonical names when unambiguous
      - Ensures missing canonical columns exist (empty)
      - Type-coerces core fields (floats for times/prob, lower-case labels)
      - Builds `detection_id` if missing: "{file_id}#{start:.3f}-{end:.3f}"
    All non-core columns are passed through unchanged.
    """
    df = df_in.copy()

    # Rename legacy columns when needed
    cols_lc = _lc_map(df.columns)
    for legacy_lc, canon in LEGACY_MAP.items():
        if legacy_lc in cols_lc and canon not in df.columns:
            df.rename(columns={cols_lc[legacy_lc]: canon}, inplace=True)

    # Ensure all core columns exist (empty if not present)
    for c in CORE_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA

    # Type coercions
    def _to_float(s) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    df["detection_start_s"]   = _to_float(df["detection_start_s"])
    df["detection_end_s"]     = _to_float(df["detection_end_s"])
    df["detection_probability"]= _to_float(df["detection_probability"])

    # Normalise presence labels, but only if they look like strings
    if "presence_label" in df.columns:
        ser = df["presence_label"].astype(str).str.strip().str.lower()
        # Keep user corrections like "present"/"absent"; treat common truthy/falsey
        truthy  = {"1","true","yes","y","t","present"}
        falsy   = {"0","false","no","n","f","absent"}
        ser = ser.mask(ser.isin(truthy), "present")
        ser = ser.mask(ser.isin(falsy),  "absent")
        df["presence_label"] = ser

    # Ensure file_id is string
    df["file_id"] = df["file_id"].astype(str)

    # file_path can legitimately be empty at ingest; leave as-is otherwise
    if "file_path" in df.columns:
        # normalise blanks to <NA>
        s = df["file_path"].astype(str).str.strip()
        df["file_path"] = s.mask(s == "", pd.NA)

    # Detection ID builder (only when missing)
    if build_detection_id and "detection_id" in df.columns:
        need = df["detection_id"].isna() | (df["detection_id"].astype(str).str.strip() == "")
        if need.any():
            def _mk(row) -> Optional[str]:
                try:
                    f = str(row["file_id"])
                    a = float(row["detection_start_s"])
                    b = float(row["detection_end_s"])
                    if not (np.isfinite(a) and np.isfinite(b)):
                        return None
                    return f"{f}#{a:.3f}-{b:.3f}"
                except Exception:
                    return None
            df.loc[need, "detection_id"] = df.loc[need].apply(_mk, axis=1)

    return df


# Columns that are often only intermediate mapping sources and safe to drop
# once canonical columns are in place.
_HELPER_LEGACY_MAPPED = {
    "source_file", "start_s", "end_s", "score", "label", "timestamp_utc",
}

def drop_mapped_columns(
    df_in: pd.DataFrame,
    mapped_sources: Iterable[str],
) -> pd.DataFrame:
    """
    Drop any columns that were used as sources for the canonical schema
    (plus a small helper set), while keeping all canonical columns from
    CORE_COLUMNS and any other metadata untouched.

    This is the same behaviour as the manual ingestion path: once a
    column has been mapped into `file_id`, `detection_start_s`, etc.,
    you do not want the original raw columns hanging around as duplicates.
    """
    df = df_in.copy()
    core_set = set(CORE_COLUMNS)

    # Start from the adapter/manual mapping list…
    mapped = {c for c in mapped_sources if c}
    # …plus the helper columns that were used in the manual path
    mapped |= _HELPER_LEGACY_MAPPED
    # Never drop canonical columns themselves
    mapped -= core_set

    keep_cols = [c for c in df.columns if c not in mapped]
    return df[keep_cols]
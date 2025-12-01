# studio/utils.py
from pathlib import Path
import pandas as pd
import json, os
from schema import normalise_schema  # scripts/schema.py, path already added by the launcher


def project_path(folder: Path, *keys: str) -> Path:
    """Resolve a project-relative path using project.json, ensuring the dirs exist."""
    folder = Path(folder)
    pj = folder / "project.json"
    if not pj.exists():
        raise FileNotFoundError(f"project.json not found under: {folder}")
    data = json.loads(pj.read_text(encoding="utf-8"))
    paths = data.get("paths") or {}
    if not keys:
        return Path(paths.get("root", str(folder))).resolve()
    base = (folder / paths[keys[0]]).resolve()
    base.mkdir(parents=True, exist_ok=True)
    for k in keys[1:]:
        base = (base / k).resolve()
    return base

def analysis_keys(df, col="source_file"):
    out = df.copy()
    out["_basename"]   = out[col].astype(str).apply(lambda p: os.path.basename(p).strip())
    out["_name_lower"] = out["_basename"].str.lower()
    out["_stem_lower"] = out["_name_lower"].apply(lambda s: os.path.splitext(s)[0])
    return out

def build_analysis_dataset(proj_path: Path, use_stem_fallback: bool = True):
    """
    Returns (df, notes) where df contains *all* original columns PLUS the canonical PAMalytics columns.
    """
    import pandas as pd

    def _first(df, *cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    def _ensure_float(s):
        return pd.to_numeric(s, errors="coerce")

    norm     = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
    enriched = project_path(proj_path, "data_normalised") / "detections_enriched.csv"
    audio_csv = project_path(proj_path, "workspace") / "audio_paths.csv"

    notes: list[str] = []
    det = None
    if enriched.exists():
        try:
            tmp = pd.read_csv(enriched, low_memory=False)
            if tmp is not None and not tmp.empty:
                det = tmp
        except Exception:
            det = None
    if det is None:
        if not norm.exists():
            return None, ["No detections found."]
        try:
            det = pd.read_csv(norm, low_memory=False)
            if det is None or det.empty:
                return None, ["Detections are empty."]
        except Exception:
            return None, ["Detections could not be read."]

    df = det.copy()

    # ensure minimal identifiers before schema pass
    c_file_id = _first(df, "file_id", "source_file", "filename", "file", "path")
    if c_file_id is None:
        return None, ["Detections lack any file identifier column (expected one of file_id/source_file/filename/file/path)."]
    df["file_id"] = df[c_file_id].astype(str)

    c_start = _first(df, "detection_start_s", "start_s", "start", "begin", "onset", "start_time_s", "start_sec")
    c_end   = _first(df, "detection_end_s",   "end_s",   "end",   "offset", "end_time_s",   "end_sec", "duration", "duration_s")
    if c_start is None or c_end is None:
        return None, ["Missing detection start/end columns. Map these in Data mapping first."]
    start_vals = _ensure_float(df[c_start])
    if c_end.lower() in {"duration", "duration_s"}:
        end_vals = start_vals + _ensure_float(df[c_end])
    else:
        end_vals = _ensure_float(df[c_end])
    df["detection_start_s"] = start_vals
    df["detection_end_s"]   = end_vals

    c_lbl = _first(df, "presence_label", "FinalLabel", "label")
    if c_lbl is None:
        df["presence_label"] = "present"
        notes.append("No presence label found; defaulted to 'present' for all rows.")
    else:
        df["presence_label"] = df[c_lbl].astype(str).str.strip()

    c_species = _first(df, "species_name", "class", "species")
    if c_species is not None:
        df["species_name"] = df[c_species].astype(str)

    c_prob = _first(df, "detection_probability", "class_prob", "probability", "score", "det_prob")
    if c_prob is not None:
        df["detection_probability"] = pd.to_numeric(df[c_prob], errors="coerce")

    c_existing_path = _first(df, "file_path", "path", "audio_path")
    if c_existing_path is not None:
        df["file_path"] = df[c_existing_path].astype(str)
    else:
        df["file_path"] = ""

    if audio_csv.exists():
        try:
            mp = pd.read_csv(audio_csv)
            if not mp.empty and {"filename", "path"}.issubset(mp.columns):
                _mp = mp.copy()
                _mp["_filename_lc"] = _mp["filename"].astype(str).str.strip().str.lower()
                _mp["_stem_lc"]     = _mp["_filename_lc"].str.replace(r"\.[^.]+$", "", regex=True)

                _fid_lc  = df["file_id"].astype(str).str.strip().str.lower()
                _stem_lc = _fid_lc.str.replace(r"\.[^.]+$", "", regex=True)

                name_to_path = dict(zip(_mp["_filename_lc"], _mp["path"]))

                need = (df["file_path"].astype(str).str.strip() == "")
                if need.any():
                    df.loc[need, "file_path"] = _fid_lc[need].map(name_to_path)

                if use_stem_fallback:
                    still = (df["file_path"].astype(str).str.strip() == "")
                    if still.any():
                        stem_counts = _mp["_stem_lc"].value_counts()
                        uniq_stems  = set(stem_counts[stem_counts == 1].index)
                        stem_to_path = dict(zip(
                            _mp.loc[_mp["_stem_lc"].isin(uniq_stems), "_stem_lc"],
                            _mp.loc[_mp["_stem_lc"].isin(uniq_stems), "path"]
                        ))
                        df.loc[still, "file_path"] = _stem_lc[still].map(stem_to_path)
        except Exception:
            pass

    if callable(normalise_schema):
        try:
            df = normalise_schema(df, build_detection_id=True)
        except Exception:
            pass

    has_path = df["file_path"].astype(str).str.strip().ne("")
    matched = df.loc[has_path].copy()
    if matched.empty:
        return None, ["No matched detections with audio file paths. Complete Audio mapping or re-check filenames."]

    return matched, notes

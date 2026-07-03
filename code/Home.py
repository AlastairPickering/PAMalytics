import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, NamedTuple
import uuid
import os
import platform
import subprocess
import sys

# Paths
STUDIO_ROOT = Path(__file__).resolve().parent      # code/
REPO_ROOT   = STUDIO_ROOT.parent                   # repo root
SCRIPTS_DIR = STUDIO_ROOT / "scripts"              # code/scripts

# Ensure scripts/ is importable
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# schema: import central normaliser
try:
    from schema import normalise_schema  # code/scripts/schema.py
except Exception:
    normalise_schema = None  # type: ignore

# Streamlit / UI
import streamlit as st  # noqa


def hide_chrome(hide_sidebar: bool = True, hide_header: bool = True) -> None:
    css = ["<style id='pa-chrome'>"]
    if hide_sidebar:
        css += [
            'section[data-testid="stSidebar"] { display: none !important; }',
            'div[data-testid="stSidebarNav"] { display: none !important; }',
            '@media (min-width: 0px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }',
        ]
    else:
        css += [
            'section[data-testid="stSidebar"], aside[data-testid="stSidebar"] {',
            '  display: block !important;',
            '  visibility: visible !important;',
            '  transform: none !important;',
            '  opacity: 1 !important;',
            '}',
            'div[data-testid="stSidebarNav"] {',
            '  display: block !important;',
            '  visibility: visible !important;',
            '}',
            '[data-testid="collapsedControl"] {',
            '  opacity: 1 !important;',
            '  visibility: visible !important;',
            '  pointer-events: auto !important;',
            '  z-index: 1000 !important;',
            '  right: 8px !important;',
            '  top: 8px !important;',
            '}',
            '[data-testid="collapsedControl"] button, [data-testid="collapsedControl"] svg {',
            '  opacity: 1 !important;',
            '}',
        ]
    if hide_header:
        css += [
            'header { visibility: hidden !important; }',
            'footer { visibility: hidden !important; }',
            '#MainMenu { visibility: hidden !important; }',
        ]
    else:
        css += [
            'header { visibility: visible !important; }',
            'footer { visibility: visible !important; }',
            '#MainMenu { visibility: visible !important; }',
        ]
    css += ["</style>"]
    st.markdown("\n".join(css), unsafe_allow_html=True, help=None)


def chip(text: str, kind: str = "info") -> str:
    colours = {"ready": "#16a34a", "pending": "#d97706", "empty": "#6b7280", "error": "#dc2626", "info": "#3b82f6"}
    return f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;background:{colours.get(kind, "#3b82f6")};color:white;font-size:12px">{text}</span>'


def _btn(label: str, key: Optional[str] = None) -> bool:
    return st.button(label, key=key or label)


def nav_row(left_label: str, left_route: str,
            right_label: Optional[str] = None, right_route: Optional[str] = None,
            key_prefix: str = "nav"):
    c1, c2 = st.columns([1, 1])
    if left_label and c1.button(left_label, key=f"{key_prefix}_left"):
        st.session_state.route = left_route
        st.rerun()
    if right_label and c2.button(right_label, key=f"{key_prefix}_right"):
        st.session_state.route = right_route
        st.rerun()


# Project model
PROJECTS_ROOT = STUDIO_ROOT / "projects"
PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
AUTH_FILE = STUDIO_ROOT / ".auth.json"


@dataclass
class ProjectManifest:
    project_id: str
    name: str
    use_case: str
    tz: str = "UTC"
    created_by: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(dt_timezone.utc).isoformat())
    last_opened: Optional[str] = None
    paths: Optional[dict] = None
    status: Optional[dict] = None
    provenance: Optional[dict] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _slug(name: str) -> str:
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name.strip())
    return s[:64] or "project"


def _default_status() -> Dict[str, str]:
    return {
        "import_results":  "empty",
        "audio_resolver":  "empty",
        "metadata_joins":  "empty",
        "analysis":        "empty",
        "export":          "empty",
    }


def _default_paths(folder: Path) -> Dict[str, str]:
    return {
        "root":            ".",
        "data_raw":        "data_raw/",
        "data_normalised": "data_normalised/",
        "metadata":        "metadata/",
        "exports":         "exports/",
        "logs":            "logs/",
        "workspace":       "workspace/",
    }


def create_project(name: str, use_case: str, created_by: Optional[str]) -> Path:
    folder = PROJECTS_ROOT / _slug(name)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = ProjectManifest(
        project_id=str(uuid.uuid4()),
        name=name,
        use_case=use_case,
        created_by=created_by or "user",
        paths=_default_paths(folder),
        status=_default_status(),
        provenance={"app": "pamalytics_studio", "version": "0.3.0"},
        last_opened=datetime.now(dt_timezone.utc).isoformat(),
    )
    for p in manifest.paths.values():
        (Path(folder) / p).mkdir(parents=True, exist_ok=True)
    (Path(folder) / "project.json").write_text(manifest.to_json(), encoding="utf-8")
    return Path(folder)


def list_projects() -> List[Path]:
    return sorted(
        [p for p in PROJECTS_ROOT.iterdir() if (p / "project.json").exists()],
        key=lambda p: p.stat().st_mtime, reverse=True
    )


def load_project(folder: Path) -> dict:
    return json.loads((folder / "project.json").read_text(encoding="utf-8"))


def save_project(folder: Path, data: Dict[str, Any]) -> None:
    (folder / "project.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def touch_last_opened(folder: Path) -> None:
    data = load_project(folder)
    data["last_opened"] = datetime.now(dt_timezone.utc).isoformat()
    save_project(folder, data)


def set_status(folder: Path, key: str, value: str) -> None:
    data = load_project(folder)
    if "status" not in data or not isinstance(data["status"], dict):
        data["status"] = _default_status()
    data["status"][key] = value
    save_project(folder, data)


def ensure_paths_schema(folder: Path) -> None:
    data = load_project(folder)
    paths = data.get("paths") or {}
    changed = False
    for k, v in _default_paths(folder).items():
        if k not in paths:
            paths[k] = v
            changed = True
    if changed:
        data["paths"] = paths
        save_project(folder, data)
        for rel in paths.values():
            (folder / rel).mkdir(parents=True, exist_ok=True)


def project_path(folder: Path, *keys: str) -> Path:
    ensure_paths_schema(folder)
    data = load_project(folder)
    base = (folder / data["paths"][keys[0]]).resolve()
    base.mkdir(parents=True, exist_ok=True)
    for k in keys[1:]:
        base = (base / k).resolve()
    return base


# Audio path helpers (project-portable)
def _is_abs_like(p: str) -> bool:
    """Heuristic that treats Windows drive/UNC paths as absolute even on POSIX."""
    p = (p or "").strip()
    if not p:
        return False
    if p.startswith("\\\\") or p.startswith("//"):
        return True
    if len(p) >= 3 and p[1] == ":" and p[2] in ("\\", "/"):
        return True
    try:
        return Path(p).is_absolute()
    except Exception:
        return False


def resolve_project_path(proj_path: Path, maybe_rel: str) -> Path:
    """Resolve a stored path (relative-to-project or absolute) to an absolute Path."""
    s = (maybe_rel or "").strip()
    if not s:
        return Path("")
    if _is_abs_like(s):
        return Path(s)
    return (proj_path / s).resolve()


def resolve_input_audio_path(audio_root: Optional[Path], p: str) -> Path:
    s = (p or "").strip()
    if not s:
        return Path("")

    candidate = Path(s).expanduser()

    if _is_abs_like(s):
        return candidate

    if audio_root:
        try:
            cand = (Path(audio_root).expanduser() / s).resolve()
            if cand.exists():
                return cand
        except Exception:
            pass

    return Path("")



def _pa_clean_value(x) -> str:
    import pandas as pd
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return s


def make_file_key(value: str) -> str:
    return _pa_clean_value(value).replace("\\", "/").lower()


def _pa_choose_single(values) -> str:
    vals = [str(v) for v in values if _pa_clean_value(v)]
    vals = list(dict.fromkeys(vals))
    return vals[0] if len(vals) == 1 else ""


def _pa_suffix_candidates(folder: str, name: str) -> List[str]:
    folder = make_file_key(folder).strip("/")
    name = Path(_pa_clean_value(name)).name.lower()
    if not name:
        return []
    out = []
    if folder:
        parts = [p for p in folder.split("/") if p]
        for i in range(len(parts)):
            out.append("/".join(parts[i:] + [name]))
    out.append(name)
    return list(dict.fromkeys([x for x in out if x]))


def _resolve_indexed_audio_value(wav_index, value: str, source_file: str = "", results_root: Optional[Path] = None) -> str:
    import re as _re
    raw = _pa_clean_value(value)
    if wav_index is None or getattr(wav_index, "empty", True) or not raw:
        return ""
    raw_path_lc = str(Path(raw).expanduser()).lower()
    raw_rel_lc = make_file_key(raw)
    raw_name_lc = Path(raw).name.lower()
    raw_stem_lc = _re.sub(r"\.[^.]+$", "", raw_name_lc)

    m = wav_index.loc[wav_index["path_lc"].eq(raw_path_lc), "path"]
    hit = _pa_choose_single(m)
    if hit:
        return hit

    m = wav_index.loc[wav_index["rel_lc"].eq(raw_rel_lc), "path"]
    hit = _pa_choose_single(m)
    if hit:
        return hit

    if "/" in raw_rel_lc:
        m = wav_index.loc[wav_index["rel_lc"].eq(raw_rel_lc) | wav_index["rel_lc"].str.endswith("/" + raw_rel_lc), "path"]
        hit = _pa_choose_single(m)
        if hit:
            return hit

    if source_file and results_root is not None:
        try:
            src_parent = Path(source_file).expanduser().resolve().parent
            try:
                folder = src_parent.relative_to(Path(results_root).expanduser().resolve()).as_posix()
            except Exception:
                folder = src_parent.name
            for cand in _pa_suffix_candidates(folder, raw_name_lc):
                m = wav_index.loc[wav_index["rel_lc"].eq(cand) | wav_index["rel_lc"].str.endswith("/" + cand), "path"]
                hit = _pa_choose_single(m)
                if hit:
                    return hit
        except Exception:
            pass

    m = wav_index.loc[wav_index["basename_lc"].eq(raw_name_lc), "path"]
    hit = _pa_choose_single(m)
    if hit:
        return hit

    m = wav_index.loc[wav_index["stem_lc"].eq(raw_stem_lc), "path"]
    hit = _pa_choose_single(m)
    if hit:
        return hit

    return ""


def _pa_unique_paths(values) -> List[str]:
    try:
        return values.dropna().astype(str).drop_duplicates().tolist()
    except Exception:
        return []


def _resolve_indexed_audio_values(wav_index, value: str, source_file: str = "", results_root: Optional[Path] = None) -> List[str]:
    import re as _re
    raw = _pa_clean_value(value)
    if wav_index is None or getattr(wav_index, "empty", True) or not raw:
        return []
    raw_path_lc = str(Path(raw).expanduser()).lower()
    raw_rel_lc = make_file_key(raw)
    raw_name_lc = Path(raw).name.lower()
    raw_stem_lc = _re.sub(r"\.[^.]+$", "", raw_name_lc)

    m = _pa_unique_paths(wav_index.loc[wav_index["path_lc"].eq(raw_path_lc), "path"])
    if m:
        return m

    m = _pa_unique_paths(wav_index.loc[wav_index["rel_lc"].eq(raw_rel_lc), "path"])
    if m:
        return m

    if "/" in raw_rel_lc:
        m = _pa_unique_paths(wav_index.loc[wav_index["rel_lc"].eq(raw_rel_lc) | wav_index["rel_lc"].str.endswith("/" + raw_rel_lc), "path"])
        if m:
            return m

    if source_file and results_root is not None:
        try:
            src_parent = Path(source_file).expanduser().resolve().parent
            try:
                folder = src_parent.relative_to(Path(results_root).expanduser().resolve()).as_posix()
            except Exception:
                folder = src_parent.name
            for cand in _pa_suffix_candidates(folder, raw_name_lc):
                m = _pa_unique_paths(wav_index.loc[wav_index["rel_lc"].eq(cand) | wav_index["rel_lc"].str.endswith("/" + cand), "path"])
                if m:
                    return m
        except Exception:
            pass

    m = _pa_unique_paths(wav_index.loc[wav_index["basename_lc"].eq(raw_name_lc), "path"])
    if m:
        return m

    if raw_stem_lc:
        m = _pa_unique_paths(wav_index.loc[wav_index["stem_lc"].eq(raw_stem_lc), "path"])
        if m:
            return m

    return []


def _pa_detection_id(row) -> str:
    import numpy as _np
    f = make_file_key(row.get("file_key", "")) or make_file_key(row.get("file_path", "")) or make_file_key(row.get("file_path_original", "")) or make_file_key(row.get("file_id", ""))
    try:
        s = float(row.get("detection_start_s"))
        e = float(row.get("detection_end_s"))
        species = make_file_key(row.get("species_name", ""))
        if not (_np.isfinite(s) and _np.isfinite(e)):
            return f"{f}:nan-nan:{species}"
        return f"{f}:{s:.3f}-{e:.3f}:{species}"
    except Exception:
        return f"{f}:nan-nan:{make_file_key(row.get('species_name', ''))}"


def _pa_rebuild_file_keys_and_detection_ids(df):
    if df is None or getattr(df, "empty", True):
        return df
    source = df["file_path"] if "file_path" in df.columns else df["file_id"]
    df["file_key"] = source.astype(str).map(make_file_key)
    if "file_path_original" in df.columns:
        missing = df["file_key"].astype(str).str.strip().eq("")
        if missing.any():
            df.loc[missing, "file_key"] = df.loc[missing, "file_path_original"].astype(str).map(make_file_key)
    df["detection_id"] = df.apply(_pa_detection_id, axis=1)
    d = df["detection_id"].astype(str)
    n = d.groupby(d).cumcount()
    dup = d.duplicated(keep=False)
    if dup.any():
        df.loc[dup, "detection_id"] = d.loc[dup] + ":" + n.loc[dup].astype(str)
    return df

def stage_audio_into_project(
    proj_path: Path,
    src_path: Path,
    *,
    dest_root: Optional[Path] = None,
    audio_root: Optional[Path] = None,
) -> str:
    """
    Copy an audio file into the project folder so the project is portable.
    Returns a POSIX-style path relative to the project root, or "" if staging fails.
    """
    import shutil

    try:
        if not src_path or not Path(src_path).exists():
            return ""
        proj_abs = proj_path.resolve()
        dest_root = (dest_root or (proj_abs / "data_raw" / "audio")).resolve()
        dest_root.mkdir(parents=True, exist_ok=True)

        src_abs = Path(src_path).resolve()
        rel_inside = None
        if audio_root:
            try:
                rel_inside = src_abs.relative_to(Path(audio_root).resolve())
            except Exception:
                rel_inside = None

        dest = (dest_root / rel_inside) if rel_inside is not None else (dest_root / src_abs.name)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if not dest.exists():
            shutil.copy2(src_abs, dest)

        try:
            return dest.relative_to(proj_abs).as_posix()
        except Exception:
            return str(dest)
    except Exception:
        return ""


# App config
st.set_page_config(page_title="PAMalytics Studio", layout="wide", initial_sidebar_state="collapsed")
hide_chrome(True, True)


# Auth helpers + Sign out
def _is_logged_in() -> bool:
    return bool(st.session_state.get("auth_user"))


def _sign_out():
    for k in list(st.session_state.keys()):
        if str(k) in {"auth_user", "route", "current_project", "pa_page"} or str(k).startswith((
            "bd2_", "manual_", "import_", "audio_", "metadata_", "ds_", "dataset_", "val_", "filters_", "pa_"
        )):
            st.session_state.pop(k, None)
    try:
        AUTH_FILE.write_text(json.dumps({"remember": False, "user": ""}), encoding="utf-8")
    except Exception:
        pass
    st.session_state.route = "login"
    st.rerun()


def _top_right_signout_button(label: str = "Sign out"):
    _, col_btn = st.columns([1, 0.18])
    with col_btn:
        st.write("")
        if st.button(label, key="pa_signout_topright"):
            _sign_out()


# Session
ss = st.session_state
ss.setdefault("auth_user", None)
ss.setdefault("route", "login")
ss.setdefault("current_project", None)
ss.setdefault("pa_page", "Dashboard")

# Import state
ss.setdefault("import_params", {})
ss.setdefault("import_preview_ready", False)
ss.setdefault("import_preview_df", None)
ss.setdefault("import_last_saved", None)
ss.setdefault("import_notes", [])

# Reset import state when switching projects
if ss.get("import_project") != ss.get("current_project"):
    ss["import_project"] = ss.get("current_project")
    ss["import_params"] = {}
    ss["import_preview_ready"] = False
    ss["import_preview_df"] = None
    ss["import_last_saved"] = None
    ss["import_notes"] = []
    ss["manual_df_linked"] = None
    ss["bd2_ingest_ready"] = False
    ss["bn_ingest_ready"] = False
    ss["manual_ingest_ready"] = False

# Audio state
ss.setdefault("audio_dirs", [])
ss.setdefault("audio_map_df", None)
ss.setdefault("audio_map_preview", None)
ss.setdefault("audio_save_path", None)

# Coverage preference
ss.setdefault("use_stem_fallback", True)
ss.setdefault("prefill_audio_dir", None)
ss.setdefault("metadata_optional", True)

# Remember-me
AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
if ss.get("auth_user") is None and AUTH_FILE.exists():
    try:
        data = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        if data.get("remember") and data.get("user"):
            ss.auth_user = data["user"]
            ss.route = "hub"
    except Exception:
        pass


# Coverage helpers
class Coverage(NamedTuple):
    matched_rows: int
    total_rows: int
    matched_unique_files: int
    total_unique_files: int


def analysis_keys(df, col="source_file"):
    import pandas as pd
    import os as _os
    out = df.copy()
    out["_basename"] = out[col].astype(str).apply(lambda p: _os.path.basename(p).strip())
    out["_name_lower"] = out["_basename"].str.lower()
    out["_stem_lower"] = out["_name_lower"].apply(lambda s: _os.path.splitext(s)[0])
    return out


def compute_audio_coverage(detections_csv: Path, mapping: Any, use_stem_fallback: bool = True) -> Coverage:
    import pandas as pd
    import os as _os
    det = pd.read_csv(detections_csv)
    if det.empty or "source_file" not in det.columns:
        return Coverage(0, 0, 0, 0)
    det = analysis_keys(det)
    total_rows = int(len(det))
    det_files = det[["_basename", "_name_lower", "_stem_lower"]].drop_duplicates()
    total_unique_files = int(len(det_files))
    if hasattr(mapping, "to_dict"):
        mp = mapping.copy()
    elif isinstance(mapping, (str, Path)):
        try:
            mp = pd.read_csv(mapping)
        except Exception:
            return Coverage(0, total_rows, 0, total_unique_files)
    else:
        return Coverage(0, total_rows, 0, total_unique_files)
    if mp.empty or "filename" not in mp.columns:
        return Coverage(0, total_rows, 0, total_unique_files)
    mp = mp.assign(_filename=mp["filename"].astype(str).str.strip())
    mp["_name_lower"] = mp["_filename"].str.lower()
    mp["_stem_lower"] = mp["_name_lower"].apply(lambda s: _os.path.splitext(s)[0])
    name_set = set(mp["_name_lower"].unique())
    name_match_mask = det["_name_lower"].isin(name_set)
    if use_stem_fallback:
        stem_counts = mp["_stem_lower"].value_counts()
        unique_stems = set(stem_counts[stem_counts == 1].index)
        stem_match_mask = det["_stem_lower"].isin(unique_stems)
        files_stem_match = det_files["_stem_lower"].isin(unique_stems)
    else:
        stem_match_mask = det["_stem_lower"].isin(set())
        files_stem_match = False
    final_match_mask = name_match_mask | (~name_match_mask & stem_match_mask)
    matched_rows = int(final_match_mask.sum())
    files_name_match = det_files["_name_lower"].isin(name_set)
    files_final = files_name_match | (~files_name_match & files_stem_match)
    matched_unique_files = int(files_final.sum())
    return Coverage(matched_rows, total_rows, matched_unique_files, total_unique_files)


def back_to_overview_bar(where: str = "top") -> None:
    c1, _ = st.columns([1, 9])
    if c1.button("⬅︎ Back to Overview", key=f"back_overview_{where}"):
        st.session_state.route = "overview"
        st.rerun()


def back_to_hub_bar(where: str = "top") -> None:
    c1, _ = st.columns([1, 9])
    if c1.button("⬅︎ Back to Project Hub", key=f"back_hub_{where}"):
        st.session_state.route = "hub"
        st.rerun()


def compute_import_stats(
    norm_csv: Path,
    audio_csv: Optional[Path],
    meta_csv: Optional[Path],
    use_stem_fallback: bool = True,
) -> Dict[str, Any]:
    import pandas as pd
    import os as _os

    stats = {
        "detections_rows": 0,
        "unique_files_in_detections": 0,
        "audio_files_indexed": 0,
        "detections_with_audio": 0,
        "metadata_join_rows": 0,
        "final_rows": 0,
    }

    if not norm_csv.exists():
        return stats

    det = pd.read_csv(norm_csv, low_memory=False)
    if det is None or det.empty:
        return stats

    if "file_id" not in det.columns:
        return stats

    det = det.copy()
    det["_basename"] = det["file_id"].astype(str).apply(lambda p: _os.path.basename(p).strip())
    det["_name_lower"] = det["_basename"].str.lower()
    det["_stem_lower"] = det["_name_lower"].apply(lambda s: _os.path.splitext(s)[0])

    stats["detections_rows"] = int(len(det))
    if "file_key" in det.columns:
        stats["unique_files_in_detections"] = int(det["file_key"].astype(str).str.strip().replace("", pd.NA).dropna().nunique())
    elif "file_path" in det.columns:
        stats["unique_files_in_detections"] = int(det["file_path"].astype(str).str.strip().replace("", pd.NA).dropna().nunique())
    else:
        stats["unique_files_in_detections"] = int(det["_basename"].nunique())

    if "file_path" in det.columns:
        fp = det["file_path"].astype(str)
        have_fp = fp.str.strip().ne("") & fp.notna()
        stats["detections_with_audio"] = int(have_fp.sum())

    if audio_csv and Path(audio_csv).exists():
        mp = pd.read_csv(audio_csv, low_memory=False)
        if mp is not None and not mp.empty and {"filename", "path"}.issubset(mp.columns):
            mp = mp.copy()
            mp["_filename_lc"] = mp["filename"].astype(str).str.strip().str.lower()
            mp["_stem_lc"] = mp["_filename_lc"].apply(lambda s: _os.path.splitext(s)[0])
            if "file_key" in mp.columns:
                stats["audio_files_indexed"] = int(mp["file_key"].astype(str).str.strip().replace("", pd.NA).dropna().nunique())
            elif "path" in mp.columns:
                stats["audio_files_indexed"] = int(mp["path"].astype(str).str.strip().replace("", pd.NA).dropna().nunique())
            else:
                stats["audio_files_indexed"] = int(mp["_filename_lc"].nunique())

            if "file_path" not in det.columns:
                name_set = set(mp["_filename_lc"].unique())
                mask = det["_name_lower"].isin(name_set)

                if use_stem_fallback:
                    stem_counts = mp["_stem_lc"].value_counts()
                    unique_stems = set(stem_counts[stem_counts == 1].index)
                    mask = mask | (~mask & det["_stem_lower"].isin(unique_stems))

                stats["detections_with_audio"] = int(mask.sum())
        else:
            stats["audio_files_indexed"] = 0

    if meta_csv and Path(meta_csv).exists():
        try:
            meta = pd.read_csv(meta_csv, low_memory=False)
            stats["metadata_join_rows"] = int(len(meta))
        except Exception:
            pass

    stats["final_rows"] = stats["detections_with_audio"]

    return stats


def render_audio_coverage(
    norm_csv: Path,
    audio_csv: Path,
    use_stem_fallback: bool = True,
    heading: str = "Audio coverage",
) -> None:
    """
    Render a standardised audio coverage block (detections, detections with audio, audio coverage %)
    for any ingestion route, if both the normalised detections and audio map exist.
    """
    if not norm_csv.exists() or not audio_csv.exists():
        return

    stats = compute_import_stats(
        norm_csv=norm_csv,
        audio_csv=audio_csv,
        meta_csv=None,
        use_stem_fallback=use_stem_fallback,
    )
    total = int(stats.get("detections_rows", 0))
    with_audio = int(stats.get("detections_with_audio", 0))

    if total <= 0:
        return

    pct = (100.0 * with_audio / total) if total else 0.0

    st.subheader(heading)
    c1, c2, c3 = st.columns(3)
    c1.metric("Detections", f"{total:,}")
    c2.metric("Detections with audio", f"{with_audio:,}")
    c3.metric("Audio coverage", f"{pct:.1f}%")


def render_norm_preview(norm_csv: Path, heading: str = "Preview mapped detections") -> None:
    """
    Render a preview of the saved normalised detections (canonical mapped table),
    used consistently across ingestion routes.
    """
    if not norm_csv.exists():
        return
    import pandas as pd
    with st.expander(heading, expanded=False):
        try:
            df_prev = pd.read_csv(norm_csv, low_memory=False)
            st.dataframe(df_prev.head(50), width='stretch')
            st.caption(f"Rows: {len(df_prev):,}")
        except Exception as e:
            st.error(f"Could not read normalised data: {e}")


# Safe delete / move-to-trash
from datetime import datetime as _dt
TRASH_DIR = PROJECTS_ROOT / ".trash"
TRASH_DIR.mkdir(parents=True, exist_ok=True)


def _trash_target_name(p: Path) -> Path:
    ts = _dt.now(dt_timezone.utc).strftime("%Y%m%d-%H%M%S")
    return TRASH_DIR / f"{p.name}__{ts}"


def move_project_to_trash(project_folder: Path) -> Path:
    if not project_folder.exists():
        raise FileNotFoundError(f"{project_folder} does not exist")
    target = _trash_target_name(project_folder)
    i = 1
    while target.exists():
        target = TRASH_DIR / f"{project_folder.name}__{i}"
        i += 1
    project_folder.rename(target)
    return target


# Schema helpers
PAMA_CORE = [
    "file_id", "file_path", "detection_id",
    "detection_start_s", "detection_end_s",
    "presence_label", "species_name", "detection_probability",
]
PAMA_OPTIONAL = ["recorder_id", "date_time"]


def _mk_detection_id(file_id: str, start: float, end: float) -> str:
    try:
        key = make_file_key(file_id)
        return f"{key}:{float(start):.3f}-{float(end):.3f}"
    except Exception:
        return f"{str(file_id)}:{start}-{end}"


def _to_canonical_names(df_in: "object") -> "object":
    import pandas as pd
    df = df_in.copy()
    if "file_id" not in df.columns:
        if "source_file" in df.columns:
            df["file_id"] = df["source_file"].astype(str)
        elif "filename" in df.columns:
            df["file_id"] = df["filename"].astype(str)
        else:
            df["file_id"] = ""
    if "detection_start_s" not in df.columns and "start_s" in df.columns:
        df["detection_start_s"] = pd.to_numeric(df["start_s"], errors="coerce")
    if "detection_end_s" not in df.columns and "end_s" in df.columns:
        df["detection_end_s"] = pd.to_numeric(df["end_s"], errors="coerce")
    if "presence_label" not in df.columns:
        if "label" in df.columns:
            df["presence_label"] = df["label"].astype(str)
        elif "FinalLabel" in df.columns:
            df["presence_label"] = df["FinalLabel"].astype(str)
        else:
            df["presence_label"] = ""
    if "species_name" not in df.columns:
        if "class" in df.columns:
            df["species_name"] = df["class"].astype(str)
        elif "species" in df.columns:
            df["species_name"] = df["species"].astype(str)
        else:
            df["species_name"] = ""
    if "detection_probability" not in df.columns:
        for cand in ("score", "class_prob", "probability", "det_prob"):
            if cand in df.columns:
                df["detection_probability"] = pd.to_numeric(df[cand], errors="coerce")
                break
        else:
            df["detection_probability"] = pd.NA
    if "file_path" not in df.columns:
        df["file_path"] = df["path"] if "path" in df.columns else ""
    if "detection_id" not in df.columns:
        if {"file_id", "detection_start_s", "detection_end_s"} <= set(df.columns):
            df["detection_id"] = [
                _mk_detection_id(fid, s, e)
                for fid, s, e in zip(df["file_id"], df["detection_start_s"], df["detection_end_s"])
            ]
        else:
            df["detection_id"] = ""
    return df


def build_analysis_dataset(proj_path: Path, use_stem_fallback: bool = True):
    """
    Returns (df, notes) where df contains all original columns plus the canonical PAMalytics columns.
    """
    import pandas as pd

    def _first(df, *cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    def _ensure_float(s):
        return pd.to_numeric(s, errors="coerce")

    norm = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
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

    c_file_id = _first(df, "file_id", "source_file", "filename", "file", "path")
    if c_file_id is None:
        return None, ["Detections lack any file identifier column (expected one of file_id/source_file/filename/file/path)."]
    df["file_id"] = df[c_file_id].astype(str)

    c_start = _first(df, "detection_start_s", "start_s", "start", "begin", "onset", "start_time_s", "start_sec")
    c_end = _first(df, "detection_end_s", "end_s", "end", "offset", "end_time_s", "end_sec", "duration", "duration_s")
    if c_start is None or c_end is None:
        return None, ["Missing detection start/end columns. Map these in Data mapping first."]
    start_vals = _ensure_float(df[c_start])
    if c_end.lower() in {"duration", "duration_s"}:
        end_vals = start_vals + _ensure_float(df[c_end])
    else:
        end_vals = _ensure_float(df[c_end])
    df["detection_start_s"] = start_vals
    df["detection_end_s"] = end_vals

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
        df["detection_probability"] = pd.to_numeric(
            c_prob if isinstance(c_prob, pd.Series) else df[c_prob],
            errors="coerce"
        )

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
                _mp["_stem_lc"] = _mp["_filename_lc"].str.replace(r"\.[^.]+$", "", regex=True)
                _mp["_file_key"] = (_mp["file_key"] if "file_key" in _mp.columns else _mp["path"]).astype(str).map(make_file_key)

                _fid_lc = df["file_id"].astype(str).str.strip().str.lower()
                _stem_lc = _fid_lc.str.replace(r"\.[^.]+$", "", regex=True)
                _df_key = (df["file_key"] if "file_key" in df.columns else df["file_path"]).astype(str).map(make_file_key)
                key_to_path = dict(zip(_mp["_file_key"], _mp["path"]))
                need = (df["file_path"].astype(str).str.strip() == "")
                if need.any():
                    df.loc[need, "file_path"] = _df_key[need].map(key_to_path)

                filename_counts = _mp["_filename_lc"].value_counts()
                unique_filenames = set(filename_counts[filename_counts == 1].index)

                name_to_path = dict(zip(
                    _mp.loc[_mp["_filename_lc"].isin(unique_filenames), "_filename_lc"],
                    _mp.loc[_mp["_filename_lc"].isin(unique_filenames), "path"]
                ))

                need = (df["file_path"].astype(str).str.strip() == "")
                if need.any():
                    df.loc[need, "file_path"] = _fid_lc[need].map(name_to_path)

                if use_stem_fallback:
                    still = (df["file_path"].astype(str).str.strip() == "")
                    if still.any():
                        stem_counts = _mp["_stem_lc"].value_counts()
                        uniq_stems = set(stem_counts[stem_counts == 1].index)
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

    # Parse date_time for dashboard filtering / plotting
    c_date_time = _first(df, "date_time", "datetime", "timestamp_utc")
    if c_date_time is not None:
        try:
            if c_date_time == "timestamp_utc":
                dt = pd.to_datetime(df[c_date_time], errors="coerce", utc=True)
            elif c_date_time == "datetime":
                dt = pd.to_datetime(df[c_date_time], errors="coerce", dayfirst=True)
            else:
                dt = pd.to_datetime(df[c_date_time], errors="coerce")

            df["date_time"] = dt

            if dt.notna().any():
                df["date"] = dt.dt.date.astype(str)
                df["hour"] = dt.dt.hour
        except Exception:
            notes.append("Could not parse date_time values for dashboard plots.")

    # Keep stored project-relative paths for portability, but provide an absolute path for playback.
    if "file_path" in df.columns:
        df["file_path_rel"] = df["file_path"].astype(str)
        df["file_path"] = df["file_path_rel"].apply(lambda s: str(resolve_project_path(proj_path, s)))

        # Optional fallback: if an original absolute path exists locally, use it when the staged file is missing.
        if "file_path_original" in df.columns:
            try:
                _abs = df["file_path"].astype(str)

                def _pick_abs(a: str) -> str:
                    try:
                        if a and Path(a).exists():
                            return a
                    except Exception:
                        pass
                    return a

                df["file_path"] = [_pick_abs(a) for a in _abs]
            except Exception:
                pass

    has_path = df["file_path"].astype(str).str.strip().ne("")
    matched = df.loc[has_path].copy()
    if matched.empty:
        return None, ["No matched detections with audio file paths. Complete Audio mapping or re-check filenames."]

    return matched, notes


# per-detection clip builder
def ensure_detection_clips(proj_path: Path, detections_df, audio_map_df=None):
    import soundfile as sf
    import pandas as pd

    clips_dir = project_path(proj_path, "workspace") / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    det = detections_df.copy()

    for idx, row in det.iterrows():
        try:
            start_s = float(row.get("detection_start_s", row.get("start_s")))
            end_s = float(row.get("detection_end_s", row.get("end_s")))
        except Exception:
            continue

        if not (end_s > start_s):
            continue

        src_value = str(row.get("file_path_rel", row.get("file_path", ""))).strip()
        if not src_value:
            continue

        src_path = resolve_project_path(proj_path, src_value)
        if not src_path.exists():
            continue

        try:
            info = sf.info(str(src_path))
            sr = info.samplerate
            total = info.frames

            start_f = max(0, int(start_s * sr))
            end_f = min(total, int(end_s * sr))
            if end_f <= start_f:
                continue

            frames = end_f - start_f
            y, _ = sf.read(str(src_path), start=start_f, frames=frames, dtype="float32", always_2d=False)

            safe_stem = Path(src_path).stem
            clip_name = f"{safe_stem}_{idx}_{start_f}_{end_f}.wav"
            clip_path = clips_dir / clip_name
            sf.write(clip_path, y, sr)

            rows.append({
                "filename": src_path.name,
                "clip_path": str(clip_path),
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": frames / sr,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


# Buttons / pickers
from pathlib import Path as _P
import os as _os

TABULAR_EXTS = {".csv", ".tsv", ".parquet"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aif", ".aiff"}


def pick_file_dialog(
    *,
    title: str = "Select a file",
    filetypes: Optional[List[Tuple[str, str]]] = None,
    initial_dir: Optional[_P] = None,
) -> Optional[str]:
    try:
        start_dir = None
        if initial_dir:
            try:
                cand = _P(initial_dir).expanduser()
                if cand.exists():
                    start_dir = str(cand)
            except Exception:
                start_dir = None
        system = platform.system().lower()
        if "darwin" in system or "mac" in system:
            prompt = json.dumps(title)
            default_clause = f" default location POSIX file {json.dumps(start_dir)}" if start_dir else ""
            script = f"set _file to POSIX path of (choose file with prompt {prompt}{default_clause})\nreturn _file"
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            path = res.stdout.strip()
            return path or None
        else:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            kwargs = {"title": title, "filetypes": filetypes or [("All files", "*.*")]}
            if start_dir:
                kwargs["initialdir"] = start_dir
            chosen = filedialog.askopenfilename(**kwargs)
            root.destroy()
            return chosen or None
    except Exception:
        return None


def pick_folder_dialog(title: str = "Select a folder") -> Optional[str]:
    try:
        system = platform.system().lower()
        if "darwin" in system or "mac" in system:
            script = f'set _folder to POSIX path of (choose folder with prompt "{title}")\nreturn _folder'
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            path = res.stdout.strip()
            return path or None
        else:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            folder = filedialog.askdirectory(title=title)
            root.destroy()
            return folder or None
    except Exception:
        return None


def _clean_dialog_path(p: str) -> str:
    p = (p or "").strip().strip('"').strip("'")
    if not p:
        return ""  # avoid normpath("") -> "."
    if p.startswith("{") and p.endswith("}"):  # occasional tkinter oddity
        p = p[1:-1]
    if p.startswith("\\\\?\\"):
        p = p[4:]
    return _os.path.normpath(p)


def _browse_into_file(state_key: str, title: str, filetypes: Optional[List[Tuple[str, str]]] = None, initial_dir: Optional[_P] = None) -> None:
    chosen = pick_file_dialog(title=title, filetypes=filetypes, initial_dir=initial_dir)
    if chosen:
        st.session_state[state_key] = _clean_dialog_path(chosen)


def _browse_into_folder(state_key: str, title: str) -> None:
    chosen = pick_folder_dialog(title=title)
    if chosen:
        st.session_state[state_key] = _clean_dialog_path(chosen)


def flexible_path_picker(
    label: str,
    state_key: str,
    *,
    allow_file: bool = True,
    allow_folder: bool = True,
    filetypes: Optional[List[Tuple[str, str]]] = None,
    placeholder: Optional[str] = None,
    initial_dir: Optional[_P] = None,
) -> Optional[_P]:
    st.session_state.setdefault(state_key, "")

    st.markdown(f"**{label}**")

    widths = [8]
    if allow_file:
        widths.append(1)
    if allow_folder:
        widths.append(1)

    cols = st.columns(widths)

    cols[0].text_input(
        f"{label} path",
        key=state_key,
        label_visibility="collapsed",
        placeholder=placeholder or (r"C:\path\to\file_or_folder" if _os.name == "nt" else "/path/to/file_or_folder"),
    )

    idx = 1
    if allow_file:
        cols[idx].button(
            "File…",
            key=f"{state_key}__browse_file",
            on_click=_browse_into_file,
            args=(state_key, f"Select {label.lower()}", filetypes, initial_dir),
        )
        idx += 1

    if allow_folder:
        cols[idx].button(
            "Folder…",
            key=f"{state_key}__browse_folder",
            on_click=_browse_into_folder,
            args=(state_key, f"Select {label.lower()}"),
        )

    val = _clean_dialog_path(str(st.session_state.get(state_key, "")))
    return _P(val) if val else None


def path_picker(label: str, state_key: str) -> Optional[_P]:
    return flexible_path_picker(
        label=label,
        state_key=state_key,
        allow_file=False,
        allow_folder=True,
        placeholder=(r"C:\path\to\folder" if _os.name == "nt" else "/path/to/folder"),
    )


def _list_result_files(path: Path) -> List[Path]:
    if not path or not path.exists():
        return []

    if path.is_file():
        return [path] if path.suffix.lower() in TABULAR_EXTS else []

    if path.is_dir():
        files: List[Path] = []
        for ext in sorted(TABULAR_EXTS):
            files.extend(path.rglob(f"*{ext}"))
        return sorted([p for p in files if p.is_file()])

    return []


def _read_result_inputs(path: Path):
    import pandas as pd

    files = _list_result_files(path)
    if not files:
        raise ValueError("No CSV, TSV or Parquet files were found in the selected location.")

    frames = []
    for f in files:
        if f.suffix.lower() == ".parquet":
            df = pd.read_parquet(f)
        elif f.suffix.lower() == ".tsv":
            df = pd.read_csv(f, sep="\t")
        else:
            try:
                df = pd.read_csv(f)
            except Exception:
                df = pd.read_csv(f, sep="\t")

        if df is not None and not df.empty:
            df = df.copy()
            df["_ingest_source_file"] = str(f)
            frames.append(df)

    if not frames:
        raise ValueError("All selected result files were empty.")

    return pd.concat(frames, ignore_index=True), files



def _audio_index_db_path(proj_path: Path) -> Path:
    return project_path(proj_path, "workspace") / "audio_index.sqlite"


def _audio_index_norm_path(p: str) -> str:
    return _pa_clean_value(p).replace("\\", "/").lower()


def _audio_index_root_abs(p: Optional[Path]) -> str:
    if not p:
        return ""
    try:
        return os.path.abspath(os.path.expanduser(str(p)))
    except Exception:
        return _pa_clean_value(str(p))


def _audio_index_read_meta(index_db: Path) -> Dict[str, str]:
    import sqlite3
    if not index_db or not Path(index_db).exists():
        return {}
    try:
        with sqlite3.connect(str(index_db)) as conn:
            tables = {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if "audio_index_meta" not in tables:
                return {}
            return {str(k): str(v) for k, v in conn.execute("SELECT key, value FROM audio_index_meta").fetchall()}
    except Exception:
        return {}


def _audio_index_required_columns_ok(conn) -> bool:
    try:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(audio_files)").fetchall()}
    except Exception:
        return False
    required = {"filename_lc", "stem_lc", "path", "path_lc", "rel_lc"}
    return required.issubset(cols)


def _audio_index_sample_paths(index_db: Path, limit: int = 5) -> Tuple[int, int]:
    import sqlite3
    tested = 0
    found = 0
    if not index_db or not Path(index_db).exists():
        return tested, found
    try:
        with sqlite3.connect(str(index_db)) as conn:
            rows = conn.execute("SELECT path FROM audio_files WHERE path IS NOT NULL AND TRIM(path) <> '' LIMIT ?", (int(limit),)).fetchall()
        for row in rows:
            tested += 1
            try:
                if Path(str(row[0])).exists():
                    found += 1
            except Exception:
                pass
    except Exception:
        pass
    return tested, found


def _audio_index_status(index_db: Path, audio_root: Optional[Path], allow_root_mismatch: bool = False, check_samples: bool = False) -> Dict[str, Any]:
    import sqlite3
    out = {
        "ready": False,
        "file_count": 0,
        "audio_root": "",
        "index_db": str(index_db) if index_db else "",
        "created_at": "",
        "built_at": "",
        "schema_version": "",
        "status": "",
        "reason": "No index selected.",
        "root_match": False,
        "sample_tested": 0,
        "sample_found": 0,
    }
    if not index_db or not Path(index_db).exists():
        out["reason"] = "Index file does not exist."
        return out
    try:
        root_abs = _audio_index_root_abs(audio_root)
        with sqlite3.connect(str(index_db)) as conn:
            tables = {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if "audio_files" not in tables:
                out["reason"] = "Index does not contain an audio_files table."
                return out
            if not _audio_index_required_columns_ok(conn):
                out["reason"] = "Index schema is missing required audio_files columns."
                return out
            meta = {}
            if "audio_index_meta" in tables:
                meta = {str(k): str(v) for k, v in conn.execute("SELECT key, value FROM audio_index_meta").fetchall()}
            cnt = int(conn.execute("SELECT COUNT(*) FROM audio_files").fetchone()[0] or 0)
        if cnt <= 0:
            out["reason"] = "Index contains no audio files."
            return out
        idx_root = meta.get("audio_root", "")
        root_match = bool(not root_abs or not idx_root or idx_root == root_abs)
        idx_status = meta.get("status", "complete")
        if idx_status != "complete":
            out["reason"] = f"Index status is {idx_status!r}, not 'complete'."
            return out
        if root_abs and idx_root and idx_root != root_abs and not allow_root_mismatch:
            out.update({
                "file_count": cnt,
                "audio_root": idx_root,
                "created_at": meta.get("created_at", ""),
                "built_at": meta.get("built_at", meta.get("created_at", "")),
                "schema_version": meta.get("schema_version", ""),
                "status": idx_status,
                "root_match": False,
                "reason": "Index was built for a different audio root.",
            })
            return out
        tested, found = (0, 0)
        if check_samples:
            tested, found = _audio_index_sample_paths(Path(index_db), limit=5)
            if tested and found == 0:
                out.update({
                    "file_count": cnt,
                    "audio_root": idx_root,
                    "created_at": meta.get("created_at", ""),
                    "built_at": meta.get("built_at", meta.get("created_at", "")),
                    "schema_version": meta.get("schema_version", ""),
                    "status": idx_status,
                    "root_match": root_match,
                    "sample_tested": tested,
                    "sample_found": found,
                    "reason": "Sample indexed paths are not accessible from this machine/session.",
                })
                return out
        out.update({
            "ready": True,
            "file_count": cnt,
            "audio_root": idx_root,
            "created_at": meta.get("created_at", ""),
            "built_at": meta.get("built_at", meta.get("created_at", "")),
            "schema_version": meta.get("schema_version", ""),
            "status": idx_status,
            "root_match": root_match,
            "sample_tested": tested,
            "sample_found": found,
            "reason": "Index is ready.",
        })
        return out
    except Exception as e:
        out["reason"] = f"Index validation failed: {e}"
        return out


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_audio_index_sqlite(audio_root: Path, index_db: Path, progress_callback=None, batch_size: int = 10000) -> Dict[str, Any]:
    import sqlite3
    import time

    root_abs = _audio_index_root_abs(audio_root)
    index_db = Path(index_db)
    index_db.parent.mkdir(parents=True, exist_ok=True)
    tmp_db = index_db.with_suffix(index_db.suffix + ".building")
    if tmp_db.exists():
        tmp_db.unlink()

    audio_exts = {e.lower() for e in AUDIO_EXTS}
    started = time.monotonic()
    count = 0
    batch = []

    conn = sqlite3.connect(str(tmp_db))
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=OFF")
        cur.execute("PRAGMA synchronous=OFF")
        cur.execute("PRAGMA temp_store=MEMORY")
        cur.execute("CREATE TABLE audio_files (id INTEGER PRIMARY KEY, audio_root TEXT NOT NULL, filename TEXT NOT NULL, filename_lc TEXT NOT NULL, stem_lc TEXT NOT NULL, suffix_lc TEXT NOT NULL, path TEXT NOT NULL, path_lc TEXT NOT NULL, rel_lc TEXT NOT NULL, parent_lc TEXT NOT NULL)")
        cur.execute("CREATE TABLE audio_index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        cur.execute("INSERT OR REPLACE INTO audio_index_meta (key, value) VALUES (?, ?)", ("schema_version", "1"))
        cur.execute("INSERT OR REPLACE INTO audio_index_meta (key, value) VALUES (?, ?)", ("audio_root", root_abs))
        cur.execute("INSERT OR REPLACE INTO audio_index_meta (key, value) VALUES (?, ?)", ("status", "building"))
        cur.execute("INSERT OR REPLACE INTO audio_index_meta (key, value) VALUES (?, ?)", ("started_at", datetime.now(dt_timezone.utc).isoformat()))
        conn.commit()

        root_path = Path(root_abs)
        if root_path.is_file():
            scan_items = [(str(root_path.parent), root_path.name, str(root_path), root_path.name)] if root_path.suffix.lower() in audio_exts else []
        else:
            scan_items = []
            for root, _, names in os.walk(root_abs):
                for nm in names:
                    suffix = os.path.splitext(nm)[1].lower()
                    if suffix not in audio_exts:
                        continue
                    full = os.path.abspath(os.path.join(root, nm))
                    try:
                        rel = os.path.relpath(full, root_abs).replace(os.sep, "/")
                    except Exception:
                        rel = nm
                    scan_items.append((root, nm, full, rel))

        for root, nm, full, rel in scan_items:
            suffix = os.path.splitext(nm)[1].lower()
            filename_lc = nm.lower()
            stem_lc = os.path.splitext(filename_lc)[0]
            path_lc = full.replace("\\", "/").lower()
            rel_lc = rel.replace("\\", "/").lower()
            parent_lc = os.path.dirname(rel_lc).replace("\\", "/").lower()
            batch.append((root_abs, nm, filename_lc, stem_lc, suffix, full, path_lc, rel_lc, parent_lc))
            count += 1
            if len(batch) >= batch_size:
                cur.executemany("INSERT INTO audio_files (audio_root, filename, filename_lc, stem_lc, suffix_lc, path, path_lc, rel_lc, parent_lc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
                conn.commit()
                batch.clear()
                if progress_callback:
                    progress_callback(count, root, time.monotonic() - started, False)

        if batch:
            cur.executemany("INSERT INTO audio_files (audio_root, filename, filename_lc, stem_lc, suffix_lc, path, path_lc, rel_lc, parent_lc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", batch)
            conn.commit()
            batch.clear()
            if progress_callback:
                progress_callback(count, root_abs, time.monotonic() - started, False)

        cur.execute("CREATE INDEX idx_audio_filename_lc ON audio_files(filename_lc)")
        cur.execute("CREATE INDEX idx_audio_stem_lc ON audio_files(stem_lc)")
        cur.execute("CREATE INDEX idx_audio_path_lc ON audio_files(path_lc)")
        cur.execute("CREATE INDEX idx_audio_rel_lc ON audio_files(rel_lc)")
        cur.execute("CREATE INDEX idx_audio_parent_lc ON audio_files(parent_lc)")
        now = datetime.now(dt_timezone.utc).isoformat()
        cur.execute("INSERT OR REPLACE INTO audio_index_meta (key, value) VALUES (?, ?)", ("file_count", str(count)))
        cur.execute("INSERT OR REPLACE INTO audio_index_meta (key, value) VALUES (?, ?)", ("created_at", now))
        cur.execute("INSERT OR REPLACE INTO audio_index_meta (key, value) VALUES (?, ?)", ("built_at", now))
        cur.execute("INSERT OR REPLACE INTO audio_index_meta (key, value) VALUES (?, ?)", ("status", "complete"))
        conn.commit()
    finally:
        conn.close()

    if index_db.exists():
        index_db.unlink()
    tmp_db.replace(index_db)
    elapsed = time.monotonic() - started
    if progress_callback:
        progress_callback(count, root_abs, elapsed, True)
    return {"ready": True, "file_count": int(count), "elapsed_s": float(elapsed), "index_db": str(index_db), "audio_root": root_abs}


def _audio_index_candidate_from_path(selected: Optional[Path]) -> Optional[Path]:
    if not selected:
        return None
    selected = Path(selected).expanduser()
    if selected.is_dir():
        for cand in (selected / "workspace" / "audio_index.sqlite", selected / "audio_index.sqlite"):
            if cand.exists():
                return cand
        return selected / "workspace" / "audio_index.sqlite"
    return selected


def _display_audio_index_status(status: Dict[str, Any], label: str = "Index") -> None:
    if status.get("ready"):
        st.success(f"{label} ready: {int(status.get('file_count', 0)):,} indexed audio files.")
        if status.get("audio_root"):
            st.caption(f"Indexed audio root: `{status.get('audio_root')}`")
        if status.get("index_db"):
            st.caption(f"Index file: `{status.get('index_db')}`")
    else:
        st.warning(f"{label} not ready: {status.get('reason', 'Unknown validation issue')}")
        if status.get("audio_root"):
            st.caption(f"Indexed audio root: `{status.get('audio_root')}`")


def _ensure_audio_index_ui(proj_path: Path, audio_root: Optional[Path], key_prefix: str, auto_build: bool = False, render_ui: bool = True) -> Dict[str, Any]:
    if not audio_root or not Path(audio_root).exists():
        return {"ready": False, "file_count": 0, "index_db": "", "reason": "Select a valid audio file or folder first."}

    index_db = _audio_index_db_path(proj_path)
    use_existing_key = f"{key_prefix}_use_existing_audio_index"
    existing_key = f"{key_prefix}_existing_audio_index_source"

    st.session_state.setdefault(use_existing_key, False)
    use_existing = bool(st.session_state.get(use_existing_key, False))
    current_status = _audio_index_status(index_db, audio_root, allow_root_mismatch=False, check_samples=False)

    if render_ui:
        with st.expander("Advanced options", expanded=False):
            st.checkbox(
                "Use an existing saved audio index",
                value=use_existing,
                key=use_existing_key,
                help="Leave this unticked for the normal import path. PAMalytics will use this project’s own index, building it automatically if needed.",
            )
            use_existing = bool(st.session_state.get(use_existing_key, False))
            if use_existing:
                selected = flexible_path_picker(
                    "Existing audio_index.sqlite",
                    existing_key,
                    allow_file=True,
                    allow_folder=False,
                    filetypes=[("SQLite", "*.sqlite *.db"), ("All files", "*.*")],
                    placeholder=str(PROJECTS_ROOT / "<project>" / "workspace" / "audio_index.sqlite"),
                    initial_dir=PROJECTS_ROOT,
                )
                cand = _audio_index_candidate_from_path(selected)
                if cand:
                    selected_status = _audio_index_status(cand, audio_root, allow_root_mismatch=False, check_samples=True)
                    _display_audio_index_status(selected_status, label="Selected index")
            else:
                st.caption("Normal users should leave this unticked.")

    use_existing = bool(st.session_state.get(use_existing_key, False))
    if use_existing:
        selected_raw = st.session_state.get(existing_key, "")
        cand = _audio_index_candidate_from_path(Path(str(selected_raw)) if selected_raw else None)
        if not cand:
            return {"ready": False, "file_count": 0, "index_db": "", "reason": "No existing audio index selected."}
        status = _audio_index_status(cand, audio_root, allow_root_mismatch=False, check_samples=True)
        if status.get("ready"):
            set_status(proj_path, "audio_resolver", "ready")
        return status

    status = _audio_index_status(index_db, audio_root, allow_root_mismatch=False, check_samples=False)
    if status.get("ready"):
        set_status(proj_path, "audio_resolver", "ready")
        return status

    should_build = bool(auto_build)
    if render_ui and not status.get("ready"):
        st.caption("A project audio index will be built from the selected audio location when ingestion runs.")

    if not should_build:
        status["index_db"] = str(index_db)
        return status

    status_box = st.empty()
    metric_box = st.empty()
    folder_box = st.empty()

    def _progress(n: int, folder: str, elapsed: float, done: bool):
        rate = (float(n) / elapsed) if elapsed > 0 else 0.0
        label = "Audio index complete" if done else "Indexing audio files"
        status_box.info(f"{label}: {int(n):,} files | elapsed {_format_elapsed(elapsed)} | {rate:,.0f} files/sec")
        metric_box.metric("Files indexed", f"{int(n):,}")
        folder_box.caption(f"Current folder: `{folder}`")

    try:
        result = _build_audio_index_sqlite(Path(audio_root), index_db, progress_callback=_progress)
        set_status(proj_path, "audio_resolver", "ready")
        st.success(f"Indexed {int(result.get('file_count', 0)):,} audio files in {_format_elapsed(float(result.get('elapsed_s', 0)))}.")
        return {"ready": True, "file_count": int(result.get("file_count", 0)), "index_db": str(index_db), "audio_root": result.get("audio_root", "")}
    except Exception as e:
        st.error(f"Audio indexing failed: {e}")
        return {"ready": False, "file_count": 0, "index_db": str(index_db), "reason": str(e)}

def _audio_index_query_paths(index_db: Path, raw_sql: str, params: Tuple[Any, ...]) -> List[str]:
    import sqlite3
    if not index_db or not Path(index_db).exists():
        return []
    try:
        with sqlite3.connect(str(index_db)) as conn:
            rows = conn.execute(raw_sql, params).fetchall()
        return list(dict.fromkeys([str(r[0]) for r in rows if r and _pa_clean_value(r[0])]))
    except Exception:
        return []


def _resolve_audio_values_sqlite(index_db: Path, value: str, source_file: str = "", results_root: Optional[Path] = None) -> List[str]:
    import re as _re
    raw = _pa_clean_value(value)
    if not raw or not index_db or not Path(index_db).exists():
        return []

    raw_rel_lc = make_file_key(raw).strip("/")
    raw_name_lc = Path(raw).name.lower()
    raw_stem_lc = _re.sub(r"\.[^.]+$", "", raw_name_lc)

    if _is_abs_like(raw):
        raw_path_lc = raw.replace("\\", "/").lower()
        m = _audio_index_query_paths(index_db, "SELECT path FROM audio_files WHERE path_lc = ?", (raw_path_lc,))
        if m:
            return m

    m = _audio_index_query_paths(index_db, "SELECT path FROM audio_files WHERE rel_lc = ?", (raw_rel_lc,))
    if m:
        return m

    if "/" in raw_rel_lc:
        m = _audio_index_query_paths(index_db, "SELECT path FROM audio_files WHERE rel_lc = ? OR rel_lc LIKE ?", (raw_rel_lc, "%/" + raw_rel_lc))
        if m:
            return m

    if source_file and results_root is not None:
        try:
            src_parent = Path(source_file).expanduser().resolve().parent
            try:
                folder = src_parent.relative_to(Path(results_root).expanduser().resolve()).as_posix()
            except Exception:
                folder = src_parent.name
            for cand in _pa_suffix_candidates(folder, raw_name_lc):
                m = _audio_index_query_paths(index_db, "SELECT path FROM audio_files WHERE rel_lc = ? OR rel_lc LIKE ?", (cand, "%/" + cand))
                if m:
                    return m
        except Exception:
            pass

    m = _audio_index_query_paths(index_db, "SELECT path FROM audio_files WHERE filename_lc = ?", (raw_name_lc,))
    if m:
        return m

    if raw_stem_lc:
        m = _audio_index_query_paths(index_db, "SELECT path FROM audio_files WHERE stem_lc = ?", (raw_stem_lc,))
        if m:
            return m

    return []


# Views
def view_login() -> None:
    hide_chrome(True, True)

    default_user = ""
    if AUTH_FILE.exists():
        try:
            prev = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
            default_user = prev.get("user", "")
        except Exception:
            pass

    # Centre the login card using columns
    left, centre, right = st.columns([2, 1, 2])
    with centre:
        # Vertical spacer to nudge the card down a bit
        st.markdown("<div style='height:10vh'></div>", unsafe_allow_html=True)

        # centred heading
        st.markdown(
            "<h2 style='text-align:center; margin-bottom:0.5rem;'>Login</h2>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p style='text-align:center; color:#9ca3af; margin-bottom:1.5rem;'>PAMalytics</p>",
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            user = st.text_input("Username", value=default_user, key="login_user")
            pin = st.text_input("PIN (optional)", type="password", key="login_pin")
            remember = st.checkbox("Remember me", value=True, key="login_remember")
            submit = st.form_submit_button("Sign in", width='stretch')

    if submit:
        if not user.strip():
            st.error("Please enter a username.")
        else:
            st.session_state.auth_user = user.strip()
            try:
                AUTH_FILE.write_text(
                    json.dumps(
                        {"remember": bool(remember), "user": st.session_state.auth_user}
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
            st.session_state.route = "hub"
            st.rerun()


def view_hub() -> None:
    """
    Project Hub:
    - Explain what PAMalytics does.
    - Let the user create a project (single use_case: external_results).
    - List recent projects with Launch / Edit / Delete.
    """
    hide_chrome(True, True)
    if not st.session_state.get("auth_user"):
        st.session_state.route = "login"
        st.rerun()

    st.title("Project Hub")
    _top_right_signout_button()

    with st.expander("Create a new project", expanded=True):
        st.markdown(
            "PAMalytics helps you **explore, review and validate classifier detections** "
            "from tools such as **BatDetect2** and **BirdNET**, and then summarise them "
            "in a consistent, project-based dashboard."
        )
        st.markdown(
            "- Start by creating a project for a single survey or study.\n"
            "- In the next step you will choose whether your detections come from **BatDetect2**, "
            "**BirdNET**, or a **custom / manual** CSV.\n"
        )

        name = st.text_input(
            "Project name",
            placeholder="e.g. Sabah 2024 – external results",
            key="proj_name",
        )

        if _btn("Create project", key="create_project_btn") and name.strip():
            folder = create_project(
                name=name.strip(),
                use_case="external_results",
                created_by=st.session_state.auth_user,
            )
            touch_last_opened(folder)
            st.session_state.current_project = str(folder)
            st.session_state.route = "overview"
            st.success(f"Created project: `{folder.name}`")
            st.rerun()

    st.subheader("Recent projects")

    projects = list_projects()
    if not projects:
        st.caption("No projects yet. Create one above.")
    else:
        for p in projects:
            data = load_project(p)
            norm_csv = project_path(p, "data_normalised") / "detections_normalised.csv"
            ready_for_launch = norm_csv.exists() and norm_csv.stat().st_size > 0

            cols = st.columns([4, 2, 2, 1, 1, 1])
            with cols[0]:
                st.markdown(f"**{data.get('name', '(unnamed)')}**  \n`{p.name}`")
            cols[1].write(f"Mode: `{data.get('use_case', 'external_results')}`")
            cols[2].write(f"Timezone: `{data.get('tz', 'UTC')}`")

            edit_key = f"edit_{p.name}"
            launch_key = f"launch_{p.name}"
            del_key = f"del_{p.name}"

            if cols[3].button("Edit", key=edit_key, width='stretch'):
                st.session_state.current_project = str(p)
                touch_last_opened(p)
                st.session_state.route = "overview"
                st.toast(f"Opened: {p.name}")
                st.rerun()

            if cols[4].button("Launch ▶", key=launch_key, width='stretch', disabled=not ready_for_launch):
                st.session_state.current_project = str(p)
                touch_last_opened(p)
                st.session_state.route = "dashboard"
                st.switch_page("pages/40_Dashboard.py")

            if cols[5].button("🗑️", key=del_key, help="Move project to Trash", width='stretch'):
                st.session_state[f"confirm_delete_{p.name}"] = True
                st.rerun()

            if st.session_state.get(f"confirm_delete_{p.name}"):
                st.warning(f"Move project `{p.name}` to Trash? This is reversible (stored in projects/.trash).")
                c1, c2 = st.columns([1, 1])
                if c1.button("Yes, move to Trash", key=f"confirm_yes_{p.name}"):
                    try:
                        new_loc = move_project_to_trash(p)
                        st.session_state.pop(f"confirm_delete_{p.name}", None)
                        st.success(f"Moved to Trash: `{new_loc.name}`")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not move to Trash: {e}")
                if c2.button("Cancel", key=f"confirm_no_{p.name}"):
                    st.session_state.pop(f"confirm_delete_{p.name}", None)
                    st.rerun()

    if st.session_state.get("current_project"):
        st.success(f"Active project: `{Path(st.session_state.current_project).name}`")


def _import_progress_cards(proj_path: Path) -> Dict[str, str]:
    data = load_project(proj_path)
    s = (data.get("status") or _default_status()).copy()
    norm_csv = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
    enriched = project_path(proj_path, "data_normalised") / "detections_enriched.csv"
    if norm_csv.exists():
        s["import_results"] = "ready"
    if enriched.exists():
        s["metadata_joins"] = "ready"
    return s


def view_overview() -> None:
    hide_chrome(True, True)
    if not st.session_state.get("auth_user"):
        st.session_state.route = "login"
        st.rerun()
    if not st.session_state.get("current_project"):
        st.session_state.route = "hub"
        st.rerun()

    proj_path = Path(st.session_state.current_project)
    ensure_paths_schema(proj_path)
    data = load_project(proj_path)

    st.title("Overview")
    c_back, _ = st.columns([1, 9])
    if c_back.button("⬅︎ Back to Project Hub", key="back_to_hub_from_overview"):
        st.session_state.route = "hub"
        st.rerun()
    st.caption(f"Project: **{data['name']}** • Mode: `{data.get('use_case', 'external_results')}` • Timezone: `{data.get('tz', 'UTC')}`")

    st.write("### Import phase (2 steps)")
    st.caption("Complete these in order: **1) Data mapping** → **2) Metadata mapping (optional)**.")
    s = _import_progress_cards(proj_path)

    norm_csv = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
    enrich_csv = project_path(proj_path, "data_normalised") / "detections_enriched.csv"
    audio_csv = project_path(proj_path, "workspace") / "audio_paths.csv"

    step1, step3 = st.columns(2)
    with step1:
        st.markdown("#### 1) Data mapping")
        st.markdown(chip(
            "ready" if s["import_results"] == "ready" else ("pending" if s["import_results"] == "pending" else "empty"),
            "ready" if s["import_results"] == "ready" else ("pending" if s["import_results"] == "pending" else "empty")
        ), unsafe_allow_html=True)
        if norm_csv.exists():
            st.caption(f"`{norm_csv.name}`")
        if st.button("Open Data mapping", key="open_step1"):
            st.session_state.route = "import"
            st.rerun()

    with step3:
        st.markdown("#### 2) Metadata mapping (optional)")
        state = s.get("metadata_joins", "empty")
        kind = "ready" if state == "ready" else ("pending" if state == "pending" else "empty")
        st.markdown(chip(kind, kind), unsafe_allow_html=True)
        if enrich_csv.exists():
            st.caption(f"`{enrich_csv.name}`")
        c3a, c3b = st.columns([1, 1])
        if c3a.button("Open Metadata mapping", key="open_step3"):
            st.session_state.route = "metadata"
            st.rerun()
        if c3b.button("Skip Metadata", key="skip_meta_btn"):
            set_status(proj_path, "metadata_joins", "skipped")
            st.success("Metadata step marked as skipped.")
            st.rerun()

    st.divider()

    stats = compute_import_stats(
        norm_csv=norm_csv,
        audio_csv=audio_csv if audio_csv.exists() else None,
        meta_csv=enrich_csv if enrich_csv.exists() else None,
        use_stem_fallback=st.session_state.get("use_stem_fallback", True),
    )
    total = int(stats.get("detections_rows", 0))
    with_audio = int(stats.get("detections_with_audio", 0))
    pct = (100.0 * with_audio / total) if total else 0.0

    m1, m2 = st.columns(2)
    m1.metric("Detections", f"{total:,}")
    m2.metric("Audio coverage", f"{pct:.1f}%", f"{with_audio:,} / {total:,}")

    st.divider()

    step1_ready = (s["import_results"] == "ready")
    if step1_ready:
        if s.get("metadata_joins") == "ready":
            st.success("Data mapping complete. Metadata mapping complete.")
        elif s.get("metadata_joins") == "skipped":
            st.info("Data mapping complete. Metadata mapping skipped.")
        else:
            st.info("Data mapping complete. You can add Metadata later (optional).")

        if st.button("Launch", key="launch_dashboard_from_overview"):
            st.session_state.route = "dashboard"
            st.switch_page("pages/40_Dashboard.py")
    else:
        st.info("Complete **Data mapping** to launch the PAMalytics dashboard.")


# Import results (Data mapping)
def _auto_guess(colnames: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in colnames}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def view_import_results() -> None:
    import pandas as pd
    from schema import drop_mapped_columns

    if not st.session_state.get("auth_user"):
        st.session_state.route = "login"
        st.rerun()
    if not st.session_state.get("current_project"):
        st.session_state.route = "hub"
        st.rerun()

    proj_path = Path(st.session_state.current_project)
    st.title("Import results - Data mapping")
    back_to_hub_bar("import_top")

    st.caption("Choose a classifier type to ingest detections, or use manual column mapping.")
    nav_row("Back to Overview", "overview", key_prefix="import_top")

    import_params = st.session_state.get("import_params", {})
    classifier_type = import_params.get("classifier_type", "manual")

    classifier_options = ["manual", "batdetect2", "birdnet"]
    if classifier_type not in classifier_options:
        classifier_type = "manual"

    classifier_type = st.radio(
        "Classifier type",
        options=classifier_options,
        index=classifier_options.index(classifier_type),
        format_func=lambda x: {
            "manual": "Manual (custom mapping)",
            "batdetect2": "BatDetect2 (per-clip CSVs)",
            "birdnet": "BirdNET (per-clip detections)",
        }.get(x, x),
        horizontal=True,
        key="classifier_type_radio",
    )
    st.session_state.import_params["classifier_type"] = classifier_type

    norm_csv = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
    ws_dir = project_path(proj_path, "workspace")
    audio_csv = ws_dir / "audio_paths.csv"

    # Shared preview of any existing saved normalised data (canonical)
    render_norm_preview(norm_csv, heading="Preview saved normalised data")
    st.divider()

    # PATH A: BATDETECT2 ADAPTER
    if classifier_type == "batdetect2":
        from adapters.batdetect2 import ingest_batdetect2

        bd2_csv_root = flexible_path_picker(
            "Classification results file or folder",
            "bd2_csv_root",
            allow_file=True,
            allow_folder=True,
            filetypes=[
                ("CSV", "*.csv"),
                ("All files", "*.*"),
            ],
            placeholder="/path/to/batdetect2_results.csv or /path/to/batdetect2_results_folder",
        )
        audio_base = flexible_path_picker(
            "Audio file or folder",
            "bd2_audio_base",
            allow_file=True,
            allow_folder=True,
            filetypes=[
                ("Audio", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.aif *.aiff"),
                ("All files", "*.*"),
            ],
            placeholder="/path/to/audio.wav or /path/to/audio_folder",
        )

        st.session_state.import_params["bd2_csv_root"] = str(bd2_csv_root) if bd2_csv_root else ""
        st.session_state.import_params["audio_base"] = str(audio_base) if audio_base else ""

        det_th = float(st.session_state.import_params.get("bd2_det_thresh", 0.5))
        cls_th = float(st.session_state.import_params.get("bd2_class_thresh", 0.2))
        te_fac = float(st.session_state.import_params.get("bd2_te_factor", 1.0))
        c1, c2, c3 = st.columns(3)
        det_th = c1.number_input("Detection threshold (det_prob ≥)", min_value=0.0, max_value=1.0, value=float(det_th), step=0.01, key="bd2_det_th")
        cls_th = c2.number_input("Class threshold (class_prob ≥)", min_value=0.0, max_value=1.0, value=float(cls_th), step=0.01, key="bd2_cls_th")
        te_fac = c3.number_input("Time expansion factor", min_value=0.1, max_value=100.0, value=float(te_fac), step=0.1, key="bd2_te_fac")

        with st.expander("Advanced mapping", expanded=False):
            st.caption("Remap BD2 → canonical columns.")
            prob_source = st.selectbox(
                "Map to `detection_probability`",
                options=["auto (prefer det_prob)", "det_prob", "class_prob"],
                index=0,
                key="bd2_prob_source",
            )
            keep_present_only = st.checkbox(
                "Keep present only (filter out 'absent')",
                value=bool(st.session_state.get("bd2_keep_present_only", False)),
                key="bd2_keep_present_only"
            )

        if audio_base and Path(audio_base).exists():
            _ensure_audio_index_ui(proj_path, audio_base, key_prefix="bd2", auto_build=False, render_ui=True)

        can_run = bd2_csv_root and bd2_csv_root.exists()
        run_btn = st.button("Ingest BatDetect2 results", disabled=not can_run, key="bd2_ingest_btn")
        if not can_run:
            st.info("Select a valid **Classification results file or folder** to enable ingestion.")

        if run_btn and can_run:
            audio_index_db = None
            if audio_base and Path(audio_base).exists():
                audio_index = _ensure_audio_index_ui(proj_path, audio_base, key_prefix="bd2", auto_build=True, render_ui=False)
                if not audio_index.get("ready"):
                    st.error(f"Audio index is not ready: {audio_index.get('reason', 'unknown validation issue')}")
                    return
                audio_index_db = Path(str(audio_index.get("index_db")))
            with st.spinner("Ingesting BatDetect2 CSVs and normalising…"):
                df_norm = ingest_batdetect2(
                    csv_root=bd2_csv_root,
                    audio_root=audio_base,
                    det_thresh=float(det_th),
                    class_thresh=float(cls_th),
                    te_factor_default=float(te_fac),
                    audio_index_db=audio_index_db,
                )

            src = st.session_state.get("bd2_prob_source")
            if src == "det_prob" and "det_prob" in df_norm.columns:
                df_norm["detection_probability"] = pd.to_numeric(df_norm["det_prob"], errors="coerce")
            elif src == "class_prob" and "class_prob" in df_norm.columns:
                df_norm["detection_probability"] = pd.to_numeric(df_norm["class_prob"], errors="coerce")

            if st.session_state.get("bd2_keep_present_only"):
                if "presence_label" in df_norm.columns:
                    df_norm = df_norm[df_norm["presence_label"].astype(str).str.lower() == "present"].copy()

            if df_norm.empty:
                st.warning("No rows were ingested from the selected BatDetect2 location.")
            else:
                norm_csv.parent.mkdir(parents=True, exist_ok=True)

                # Stage audio into the project so paths are portable.
                try:
                    audio_dest_root = project_path(proj_path, "data_raw") / "audio"
                    audio_root = audio_base if (audio_base and Path(audio_base).exists()) else None
                    if "file_path" in df_norm.columns:
                        df_norm["file_path_original"] = df_norm["file_path"].astype(str)
                        _uniq = df_norm["file_path_original"].fillna("").astype(str).unique()
                        _rel_map = {}
                        for _p0 in _uniq:
                            _p0s = _pa_clean_value(_p0)
                            if not _p0s:
                                _rel_map[_p0] = ""
                                continue
                            _src = resolve_input_audio_path(audio_root, _p0s)
                            if _src and _src.exists():
                                _rel = stage_audio_into_project(
                                    proj_path, _src, dest_root=audio_dest_root, audio_root=audio_root
                                )
                                _rel_map[_p0] = _rel if _rel else _p0s
                            else:
                                _rel_map[_p0] = _p0s
                        df_norm["file_path"] = df_norm["file_path_original"].astype(str).map(_rel_map).fillna("")
                except Exception:
                    pass

                df_norm = _pa_rebuild_file_keys_and_detection_ids(df_norm)
                df_norm.to_csv(norm_csv, index=False)

                if {"file_id", "file_path"} <= set(df_norm.columns):
                    cols = ["file_id", "file_path"]
                    if "file_key" in df_norm.columns:
                        cols.append("file_key")
                    if "file_path_original" in df_norm.columns:
                        cols.append("file_path_original")
                    mp = df_norm.loc[df_norm["file_path"].astype(str).str.strip().ne(""), cols].drop_duplicates()
                    if not mp.empty:
                        mp = mp.rename(columns={"file_id": "filename", "file_path": "path", "file_path_original": "original_path"})
                        ws_dir.mkdir(parents=True, exist_ok=True)
                        out_map = ws_dir / "audio_paths.csv"
                        mp.to_csv(out_map, index=False)
                        set_status(proj_path, "audio_resolver", "ready")
                        st.success(f"Saved audio mapping: `{out_map}`")

                set_status(proj_path, "import_results", "ready")
                st.session_state.import_params["bd2_det_thresh"] = float(det_th)
                st.session_state.import_params["bd2_class_thresh"] = float(cls_th)
                st.session_state.import_params["bd2_te_factor"] = float(te_fac)

                st.session_state["bd2_ingest_ready"] = True

                st.success(f"Normalised BD2 detections saved to: `{norm_csv}`")

        if st.session_state.get("bd2_ingest_ready") or norm_csv.exists():
            st.divider()

            render_audio_coverage(
                norm_csv=norm_csv,
                audio_csv=audio_csv,
                use_stem_fallback=st.session_state.get("use_stem_fallback", True),
            )

            render_norm_preview(norm_csv, heading="Preview mapped detections (BatDetect2)")

            cL, cR = st.columns([1, 1])
            with cL:
                if st.button("Launch PAMalytics dashboard ▶", key="go_dashboard_from_bd2"):
                    st.switch_page("pages/40_Dashboard.py")
            with cR:
                if st.button("Back to Overview ▶", key="go_overview_from_bd2"):
                    st.session_state.route = "overview"
                    st.rerun()

        return

    # PATH B: BIRDNET ADAPTER
    if classifier_type == "birdnet":
        from adapters.birdnet import ingest_birdnet

        bn_csv_root = flexible_path_picker(
            "BirdNET results file or folder",
            "bn_csv_root",
            allow_file=True,
            allow_folder=True,
            filetypes=[
                ("CSV", "*.csv"),
                ("All files", "*.*"),
            ],
            placeholder="/path/to/birdnet_results.csv or /path/to/birdnet_results_folder",
        )
        audio_base = flexible_path_picker(
            "Audio file or folder",
            "bn_audio_base",
            allow_file=True,
            allow_folder=True,
            filetypes=[
                ("Audio", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.aif *.aiff"),
                ("All files", "*.*"),
            ],
            placeholder="/path/to/audio.wav or /path/to/audio_folder",
        )

        st.session_state.import_params["bn_csv_root"] = str(bn_csv_root) if bn_csv_root else ""
        st.session_state.import_params["audio_base"] = str(audio_base) if audio_base else ""

        min_conf = float(st.session_state.import_params.get("bn_min_conf", 0.2))
        keep_present_only = bool(st.session_state.import_params.get("bn_keep_present_only", True))

        c1, c2 = st.columns(2)
        min_conf = c1.number_input(
            "Minimum confidence (BirdNET `confidence` ≥)",
            min_value=0.0, max_value=1.0,
            value=float(min_conf), step=0.01,
            key="bn_min_conf"
        )
        keep_present_only = c2.checkbox(
            "Keep present only (filter out low-confidence rows)",
            value=keep_present_only,
            key="bn_keep_present_only"
        )

        st.session_state.import_params["bn_min_conf"] = float(min_conf)
        st.session_state.import_params["bn_keep_present_only"] = bool(keep_present_only)

        if audio_base and Path(audio_base).exists():
            _ensure_audio_index_ui(proj_path, audio_base, key_prefix="bn", auto_build=False, render_ui=True)

        can_run = bn_csv_root and bn_csv_root.exists()
        run_btn = st.button("Ingest BirdNET results", disabled=not can_run, key="bn_ingest_btn")
        if not can_run:
            st.info("Select a valid **BirdNET results file or folder** to enable ingestion.")

        if run_btn and can_run:
            audio_index_db = None
            if audio_base and Path(audio_base).exists():
                audio_index = _ensure_audio_index_ui(proj_path, audio_base, key_prefix="bn", auto_build=True, render_ui=False)
                if not audio_index.get("ready"):
                    st.error(f"Audio index is not ready: {audio_index.get('reason', 'unknown validation issue')}")
                    return
                audio_index_db = Path(str(audio_index.get("index_db")))
            with st.spinner("Ingesting BirdNET results and normalising…"):
                df_norm = ingest_birdnet(
                    csv_root=bn_csv_root,
                    audio_root=audio_base,
                    min_conf=float(min_conf),
                    keep_only_present=bool(keep_present_only),
                    audio_index_db=audio_index_db,
                )

            if df_norm.empty:
                st.warning("No rows were ingested from the selected BirdNET location.")
            else:
                norm_csv.parent.mkdir(parents=True, exist_ok=True)

                # Stage audio into the project so paths are portable.
                try:
                    audio_dest_root = project_path(proj_path, "data_raw") / "audio"
                    audio_root = audio_base if (audio_base and Path(audio_base).exists()) else None
                    if "file_path" in df_norm.columns:
                        df_norm["file_path_original"] = df_norm["file_path"].astype(str)
                        _uniq = df_norm["file_path_original"].fillna("").astype(str).unique()
                        _rel_map = {}
                        for _p0 in _uniq:
                            _p0s = _pa_clean_value(_p0)
                            if not _p0s:
                                _rel_map[_p0] = ""
                                continue
                            _src = resolve_input_audio_path(audio_root, _p0s)
                            if _src and _src.exists():
                                _rel = stage_audio_into_project(
                                    proj_path, _src, dest_root=audio_dest_root, audio_root=audio_root
                                )
                                _rel_map[_p0] = _rel if _rel else _p0s
                            else:
                                _rel_map[_p0] = _p0s
                        df_norm["file_path"] = df_norm["file_path_original"].astype(str).map(_rel_map).fillna("")
                except Exception:
                    pass

                df_norm = _pa_rebuild_file_keys_and_detection_ids(df_norm)
                df_norm.to_csv(norm_csv, index=False)

                if {"file_id", "file_path"} <= set(df_norm.columns):
                    cols = ["file_id", "file_path"]
                    if "file_key" in df_norm.columns:
                        cols.append("file_key")
                    if "file_path_original" in df_norm.columns:
                        cols.append("file_path_original")
                    mp = df_norm.loc[
                        df_norm["file_path"].astype(str).str.strip().ne(""),
                        cols
                    ].drop_duplicates()

                    if not mp.empty:
                        mp = mp.rename(columns={"file_id": "filename", "file_path": "path", "file_path_original": "original_path"})
                        ws_dir.mkdir(parents=True, exist_ok=True)
                        out_map = ws_dir / "audio_paths.csv"
                        mp.to_csv(out_map, index=False)
                        set_status(proj_path, "audio_resolver", "ready")
                        st.success(f"Saved audio mapping: `{out_map}`")

                set_status(proj_path, "import_results", "ready")
                st.session_state["bn_ingest_ready"] = True
                st.success(f"Normalised BirdNET detections saved to: `{norm_csv}`")

        if st.session_state.get("bn_ingest_ready") or norm_csv.exists():
            st.divider()

            render_audio_coverage(
                norm_csv=norm_csv,
                audio_csv=audio_csv,
                use_stem_fallback=st.session_state.get("use_stem_fallback", True),
            )

            render_norm_preview(norm_csv, heading="Preview mapped detections (BirdNET)")

            cL, cR = st.columns([1, 1])
            with cL:
                if st.button("Launch PAMalytics dashboard ▶", key="go_dashboard_from_bn"):
                    st.switch_page("pages/40_Dashboard.py")
            with cR:
                if st.button("Back to Overview ▶", key="go_overview_from_bn"):
                    st.session_state.route = "overview"
                    st.rerun()

        return

    # PATH C: MANUAL MAPPING
    results_path = flexible_path_picker(
        "Classifier results file or folder",
        "manual_results_path",
        allow_file=True,
        allow_folder=True,
        filetypes=[
            ("Tabular", "*.csv *.tsv *.parquet"),
            ("All files", "*.*"),
        ],
        placeholder="/path/to/results.csv or /path/to/results_folder",
    )
    audio_base = flexible_path_picker(
        "Audio file or folder",
        "manual_audio_source",
        allow_file=True,
        allow_folder=True,
        filetypes=[
            ("Audio", "*.wav *.mp3 *.flac *.m4a *.aac *.ogg *.aif *.aiff"),
            ("All files", "*.*"),
        ],
        placeholder="/path/to/audio.wav or /path/to/audio_folder",
    )

    st.session_state.import_params["results_file"] = str(results_path) if results_path else ""
    if audio_base:
        st.session_state.import_params["audio_base"] = str(audio_base)

    norm_csv = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
    ws_dir = project_path(proj_path, "workspace")
    audio_csv = ws_dir / "audio_paths.csv"

    df = st.session_state.import_params.get("df")
    if results_path and results_path.exists():
        try:
            df, input_files = _read_result_inputs(results_path)
            if df is None or df.empty:
                st.error("No rows found in the selected results location.")
                return
            st.session_state.import_params["filename"] = results_path.name
            st.session_state.import_params["input_files"] = [str(p) for p in input_files]
            st.session_state.import_params["df"] = df
        except Exception as e:
            st.error(f"Could not read results: {e}")
            return

    if df is None and not norm_csv.exists():
        st.info("Select a **Classifier results file or folder** to begin.")
        return

    if df is not None:
        st.write("Preview (first 20 rows):")
        st.dataframe(df.head(20), width='stretch')
        input_files = st.session_state.import_params.get("input_files", [])
        if input_files:
            st.caption(f"Loaded {len(input_files):,} result file(s).")

    st.subheader("1) Link each detection to an audio file")

    if audio_base is None or not audio_base.exists():
        st.info("Pick a valid **Audio file or folder** to index and link audio files.")
        return

    cols = list(df.columns)

    def _auto_guess_av(colnames: List[str], candidates: List[str]) -> Optional[str]:
        lower = {c.lower(): c for c in colnames}
        for cand in candidates:
            if cand in lower:
                return lower[cand]
        return None

    audio_col_guess = _auto_guess_av(cols, [
        "file_path", "audio_path", "path", "filepath", "source_file",
        "filename", "file", "wav", "wav_path"
    ])
    current_audio_col = st.session_state.get("manual_audio_col", "—")
    if current_audio_col == "—" and audio_col_guess in cols:
        current_audio_col = audio_col_guess
    idx = (cols.index(current_audio_col) + 1) if (current_audio_col in cols) else 0
    audio_col = st.selectbox(
        "Which column in your results contains the recording file (filename or path)?",
        ["—"] + cols, index=idx, key="manual_audio_col"
    )
    if audio_col == "—":
        st.warning("Choose the audio column to proceed.")
        return

    audio_index = _ensure_audio_index_ui(proj_path, audio_base, key_prefix="manual", auto_build=True)
    if not audio_index.get("ready"):
        return
    if int(audio_index.get("file_count", 0)) <= 0:
        st.error("No supported audio files found in the selected audio location.")
        return
    wav_index = Path(str(audio_index.get("index_db")))

    df_link = df.copy()

    results_root = None
    try:
        results_root = results_path if results_path and Path(results_path).is_dir() else Path(results_path).parent
    except Exception:
        results_root = None

    vals = df_link[audio_col].astype(str).str.strip()
    sources = df_link["_ingest_source_file"].astype(str) if "_ingest_source_file" in df_link.columns else pd.Series([""] * len(df_link), index=df_link.index)
    cache = {}
    expanded_rows = []
    for idx_row, raw_value in vals.items():
        source_value = str(sources.loc[idx_row]) if idx_row in sources.index else ""
        cache_key = (str(raw_value), source_value)
        if cache_key not in cache:
            cache[cache_key] = _resolve_audio_values_sqlite(wav_index, str(raw_value), source_file=source_value, results_root=results_root)
        matches = cache[cache_key]
        base_row = df_link.loc[idx_row].copy()
        if matches:
            for match in matches:
                r = base_row.copy()
                r["file_path_original"] = match
                r["file_path"] = match
                expanded_rows.append(r)
        else:
            base_row["file_path_original"] = ""
            base_row["file_path"] = ""
            expanded_rows.append(base_row)

    df_link = pd.DataFrame(expanded_rows).reset_index(drop=True)
    df_link["file_key"] = df_link["file_path"].astype(str).map(make_file_key)

    matched_mask = df_link["file_path"].notna() & df_link["file_path"].astype(str).str.strip().ne("")
    total_rows = int(len(df_link))
    matched_rows = int(matched_mask.sum())
    pct = (100.0 * matched_rows / total_rows) if total_rows else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Detections", f"{total_rows:,}")
    c2.metric("Detections with audio", f"{matched_rows:,}")
    c3.metric("Audio coverage", f"{pct:.1f}%")

    preview_cols = []
    if audio_col in df_link.columns:
        preview_cols.append(audio_col)
    if "file_path" in df_link.columns and "file_path" not in preview_cols:
        preview_cols.append("file_path")
    if not preview_cols:
        preview_cols = ["file_path"]

    with st.expander("Linked preview (sample)", expanded=True):
        sample_matched = df_link.loc[matched_mask, preview_cols].head(30)
        if not sample_matched.empty:
            st.caption("Matched rows (first 30)")
            st.dataframe(sample_matched, width='stretch')
        else:
            st.caption("No matched rows in sample. (You can still see unmatched below.)")

    if matched_rows < total_rows:
        with st.expander("Unmatched rows (sample)"):
            sample_unmatched = df_link.loc[~matched_mask, preview_cols].head(30)
            st.dataframe(sample_unmatched, width='stretch')

    st.session_state["manual_df_linked"] = df_link

    proceed = st.checkbox(
        "Looks good — continue to column mapping",
        value=st.session_state.get("manual_audio_ok", False),
        key="manual_audio_ok"
    )
    if not proceed:
        return

    st.subheader("2) Map and normalise to the canonical schema")

    df_av = st.session_state["manual_df_linked"]
    st.write("Preview (first 20 rows from linked table):")
    st.dataframe(df_av.head(20), width='stretch')

    cols_av = list(df_av.columns)

    def _auto_guess_av(colnames: List[str], candidates: List[str]) -> Optional[str]:
        lower = {c.lower(): c for c in colnames}
        for cand in candidates:
            if cand in lower:
                return lower[cand]
        return None

    file_id_guess = _auto_guess_av(cols_av, ["file_id", "source_file", "filename", "file", "filepath", "path_in_results"])
    start_guess = _auto_guess_av(cols_av, ["detection_start_s", "start", "start_s", "start_time_s", "begin", "onset", "start_sec"])
    end_guess = _auto_guess_av(cols_av, ["detection_end_s", "end", "end_s", "end_time_s", "offset", "end_sec", "duration", "duration_s"])
    class_guess = _auto_guess_av(cols_av, ["species_name", "class", "species", "label", "prediction", "taxon"])
    score_guess = _auto_guess_av(cols_av, ["detection_probability", "score", "prob", "probability", "class_prob", "det_prob"])

    def pick(name_key: str, label: str, default: Optional[str]):
        current = st.session_state.import_params.get(name_key, "—")
        if current == "—" and (default in cols_av if cols_av else False):
            current = default
        idx = (cols_av.index(current) + 1) if (cols_av and current in cols_av) else 0
        choice = st.selectbox(label, ["—"] + cols_av, index=idx, key=f"select_{name_key}")
        st.session_state.import_params[name_key] = choice
        return choice

    file_id_col = pick("file_id", "Detector file id → `file_id`", file_id_guess)
    start_col = pick("start_s", "Start time (seconds) → `detection_start_s`", start_guess)
    end_col = pick("end_s", "End time or duration → `detection_end_s`", end_guess)

    st.markdown("**Presence label → `presence_label`**")
    label_mode = st.session_state.import_params.get("label_mode", "binary_presence_column")
    label_mode = st.radio(
        "",
        options=["binary_presence_column", "use_label_column"],
        index=0 if label_mode == "binary_presence_column" else 1,
        format_func=lambda x: "Binary presence column (0/1, true/false, yes/no…)" if x == "binary_presence_column" else "Use existing label column",
        horizontal=False,
        key="label_mode_radio"
    )
    st.session_state.import_params["label_mode"] = label_mode

    presence_col = st.session_state.import_params.get("presence_col", "—")
    label_col = st.session_state.import_params.get("label_col", "—")

    if label_mode == "binary_presence_column":
        guess_presence = _auto_guess_av(cols_av, ["presence_label", "present", "presence", "detected", "label", "class", "species"])
        if presence_col == "—" and guess_presence in cols_av:
            presence_col = guess_presence
        presence_col = st.selectbox("Presence column", ["—"] + cols_av,
                                    index=(cols_av.index(presence_col) + 1) if presence_col in cols_av else 0,
                                    key="presence_col_select")
        positive_tokens = st.text_input("Values that mean 'present' (comma-separated)", value=st.session_state.import_params.get("positive_tokens", "1,true,yes,present,y,t"), key="positive_tokens")
        positive_label_name = st.text_input("Canonical label for present detections", value=st.session_state.import_params.get("positive_label_name", "present"), key="positive_label_name")
        keep_only_present = st.checkbox("Keep only present rows (detections only)", value=bool(st.session_state.import_params.get("keep_only_present", True)), key="keep_only_present")
        st.session_state.import_params.update({
            "presence_col": presence_col,
            "positive_tokens": positive_tokens,
            "positive_label_name": positive_label_name,
            "keep_only_present": bool(keep_only_present),
        })
    else:
        label_col_guess = _auto_guess_av(cols_av, ["presence_label", "label", "species", "class", "prediction"])
        if label_col == "—" and label_col_guess in cols_av:
            label_col = label_col_guess
        label_col = st.selectbox("Label column", ["—"] + cols_av,
                                 index=(cols_av.index(label_col) + 1) if label_col in cols_av else 0,
                                 key="label_col_select")
        canonicalise_existing = st.checkbox("Canonicalise this label to present/absent", value=bool(st.session_state.import_params.get("canonicalise_existing", False)), key="canonicalise_existing")
        present_value_for_existing = st.text_input("Value that means 'present' (used only when canonicalising)", value=str(st.session_state.import_params.get("present_value_for_existing", "1")), key="present_value_for_existing")
        st.session_state.import_params.update({
            "label_col": label_col,
            "canonicalise_existing": bool(canonicalise_existing),
            "present_value_for_existing": present_value_for_existing,
        })

    species_col = pick("species_name", "Species / class → `species_name`", class_guess)
    prob_col = pick("score", "Probability / score → `detection_probability`", score_guess)

    missing = []
    if file_id_col == "—":
        missing.append("file_id")
    if start_col == "—":
        missing.append("detection_start_s")
    if end_col == "—":
        missing.append("detection_end_s")
    if label_mode == "binary_presence_column" and (presence_col in (None, "—")):
        missing.append("presence column")
    if label_mode == "use_label_column" and (label_col in (None, "—")):
        missing.append("label column")
    if species_col == "—":
        missing.append("species_name")
    if prob_col == "—":
        missing.append("detection_probability")
    if missing:
        st.warning("Please map required fields: " + ", ".join(missing))

    st.subheader("Options")
    convert_ms = st.checkbox("Times are in milliseconds (convert to seconds)", value=bool(st.session_state.import_params.get("convert_ms", False)), key="convert_ms")
    assume_utc = st.checkbox("Interpret datetimes as UTC when timezone is missing (only if you pass a datetime later)", value=bool(st.session_state.import_params.get("assume_utc", True)), key="assume_utc")
    te_factor_default = st.number_input("Time expansion factor (if your times are TE’d)", min_value=0.1, max_value=100.0, value=float(st.session_state.import_params.get("manual_te_factor", 1.0)), step=0.1, key="manual_te_factor")
    st.session_state.import_params.update({"convert_ms": bool(convert_ms), "assume_utc": bool(assume_utc)})

    st.markdown("### 3) Build preview (editable)")
    disabled = bool(missing) or (st.session_state.get("manual_df_linked") is None)
    if _btn("Build preview", key="build_preview_btn") and not disabled:
        norm, notes = _build_normalised_table(
            df=df_av, source_file_col=file_id_col, start_col=start_col, end_col=end_col,
            score_col=prob_col, ts_col=None, convert_ms=bool(convert_ms), assume_utc=bool(assume_utc),
            label_mode=label_mode, presence_col=presence_col, positive_tokens=st.session_state.import_params.get("positive_tokens", "1,true,yes,present,y,t"),
            positive_label_name=st.session_state.import_params.get("positive_label_name", "present"), keep_only_present=bool(st.session_state.import_params.get("keep_only_present", True)),
            label_col=label_col, canonicalise_existing=bool(st.session_state.import_params.get("canonicalise_existing", False)),
            present_value_for_existing=st.session_state.import_params.get("present_value_for_existing", "1"),
        )

        import numpy as _np

        if norm is None or norm.empty:
            st.warning("No rows after your label filter/mapping. Adjust your presence mapping or disable 'keep only present'.")
            return

        idx = norm.index
        cn = df_av.loc[idx].copy()

        def _from_norm(col_main, col_fallback=None, numeric=False):
            if col_main in norm.columns:
                s = norm[col_main].reindex(idx)
            elif col_fallback and col_fallback in norm.columns:
                s = norm[col_fallback].reindex(idx)
            else:
                return pd.Series([pd.NA] * len(idx), index=idx)
            if numeric:
                return pd.to_numeric(s, errors="coerce")
            return s

        cn["detection_start_s"] = _from_norm("detection_start_s", "start_s", numeric=True)
        cn["detection_end_s"] = _from_norm("detection_end_s", "end_s", numeric=True)

        pl = _from_norm("presence_label")
        if pl.isna().all() and "label" in norm.columns:
            pl = norm["label"].reindex(idx).astype(str).str.strip().str.lower()
        cn["presence_label"] = pl.astype(str).str.strip().str.lower()

        sp = _from_norm("species_name")
        if sp.isna().all():
            if species_col and species_col != "—" and species_col in df_av.columns:
                sp = df_av.loc[idx, species_col].astype(str)
            elif "species_name" in df_av.columns:
                sp = df_av.loc[idx, "species_name"].astype(str)
            else:
                sp = pd.Series([""] * len(idx), index=idx)
        cn["species_name"] = sp.astype(str)

        cn["detection_probability"] = _from_norm("detection_probability", "score", numeric=True)

        cn["file_id"] = df_av.loc[idx, file_id_col].astype(str)
        if "file_path_original" in df_av.columns:
            cn["file_path_original"] = df_av.loc[idx, "file_path_original"].astype(str)
            cn["file_path"] = df_av.loc[idx, "file_path"].astype(str)
        else:
            cn["file_path_original"] = df_av.loc[idx, "file_path"].astype(str)
            cn["file_path"] = df_av.loc[idx, "file_path"].astype(str)

        cn = _pa_rebuild_file_keys_and_detection_ids(cn)

        if callable(normalise_schema):
            try:
                cn = normalise_schema(cn, build_detection_id=False)
                cn = _pa_rebuild_file_keys_and_detection_ids(cn)
            except Exception as e:
                st.warning(f"Schema normalisation failed on preview: {e}")

        mapped_sources = set()
        for col in [file_id_col, start_col, end_col, species_col, prob_col]:
            if col and col != "—":
                mapped_sources.add(col)

        if label_mode == "binary_presence_column":
            if presence_col and presence_col != "—":
                mapped_sources.add(presence_col)
        else:
            if label_col and label_col != "—":
                mapped_sources.add(label_col)

        cn = drop_mapped_columns(cn, mapped_sources)

        st.session_state.import_preview_ready = True
        st.session_state.import_preview_df = cn.to_dict(orient="records")
        st.session_state.import_last_saved = None
        st.session_state.import_notes = notes
        st.rerun()

    if disabled:
        st.caption("Map all required fields above to enable the preview.")

    if st.session_state.get("import_preview_ready") and isinstance(st.session_state.get("import_preview_df"), list):
        norm = pd.DataFrame.from_records(st.session_state.import_preview_df)
        for n in st.session_state.get("import_notes", []):
            st.caption(n)
        if norm.empty:
            st.warning("No rows to display. Please check your present/absent values.")
            return

        st.subheader("4) Validate & edit the final mapped data (canonical)")
        edited = st.data_editor(norm, width='stretch', num_rows="dynamic", key="norm_editor")

        if _btn("Save normalised copy", key="save_norm_btn"):
            out_dir = project_path(proj_path, "data_normalised")
            out_dir.mkdir(parents=True, exist_ok=True)

            if callable(normalise_schema):
                try:
                    out_df = normalise_schema(edited.copy(), build_detection_id=False)
                except Exception as e:
                    st.warning(f"Schema normalisation failed on save (writing edited table as-is): {e}")
                    out_df = edited.copy()
            else:
                out_df = edited.copy()

            canonical_cols = [
                "file_id", "file_path", "file_key", "detection_id",
                "detection_start_s", "detection_end_s",
                "presence_label", "species_name", "detection_probability",
            ]

            mapped_sources = set()
            for col in [file_id_col, start_col, end_col, species_col, prob_col]:
                if col and col != "—":
                    mapped_sources.add(col)

            if label_mode == "binary_presence_column":
                if presence_col and presence_col != "—":
                    mapped_sources.add(presence_col)
            else:
                if label_col and label_col != "—":
                    mapped_sources.add(label_col)

            helper_legacy = {"source_file", "start_s", "end_s", "score", "label", "timestamp_utc"}
            mapped_sources |= helper_legacy
            mapped_sources -= set(canonical_cols)

            out_df = out_df[[c for c in out_df.columns if c not in mapped_sources]]

            for c in canonical_cols:
                if c not in out_df.columns:
                    out_df[c] = pd.NA
            other = [c for c in out_df.columns if c not in canonical_cols]
            out_df = out_df[canonical_cols + other]

            required = set(canonical_cols)
            miss = [c for c in required if c not in out_df.columns]
            if miss:
                st.error("Missing required columns: " + ", ".join(miss))
                st.stop()

            out_df["detection_start_s"] = pd.to_numeric(out_df["detection_start_s"], errors="coerce")
            out_df["detection_end_s"] = pd.to_numeric(out_df["detection_end_s"], errors="coerce")
            out_df["detection_probability"] = pd.to_numeric(out_df["detection_probability"], errors="coerce")

            # Stage audio into the project and store portable relative paths.
            try:
                audio_dest_root = project_path(proj_path, "data_raw") / "audio"

                audio_root = None
                _ab = st.session_state.import_params.get("audio_base") or ""
                try:
                    _abp = Path(str(_ab)) if str(_ab).strip() else None
                    if _abp and _abp.exists():
                        audio_root = _abp
                except Exception:
                    audio_root = None

                if "file_path_original" not in out_df.columns:
                    out_df["file_path_original"] = out_df["file_path"].astype(str)

                _orig = out_df["file_path_original"].astype(str).fillna("")
                _uniq = _orig.unique()
                _rel_map = {}
                for _p0 in _uniq:
                    _p0s = str(_p0).strip()
                    if not _p0s:
                        _rel_map[_p0] = ""
                        continue

                    # If a non-absolute path already exists inside the project, keep it as-is.
                    _cand_in_proj = resolve_project_path(proj_path, _p0s)
                    if (not _is_abs_like(_p0s)) and _cand_in_proj.exists():
                        try:
                            _rel_map[_p0] = _cand_in_proj.relative_to(proj_path.resolve()).as_posix()
                        except Exception:
                            _rel_map[_p0] = _p0s
                        continue

                    _src = resolve_input_audio_path(audio_root, _p0s)
                    if _src and _src.exists():
                        _rel = stage_audio_into_project(
                            proj_path, _src, dest_root=audio_dest_root, audio_root=audio_root
                        )
                        _rel_map[_p0] = _rel
                    else:
                        _rel_map[_p0] = ""

                out_df["file_path"] = _orig.map(_rel_map).fillna("")
                out_df = _pa_rebuild_file_keys_and_detection_ids(out_df)

                missing_stage = (_orig.str.strip().ne("")) & (out_df["file_path"].astype(str).str.strip().eq(""))
                if missing_stage.any():
                    st.error(
                        f"Could not stage {int(missing_stage.sum()):,} audio file(s) into the project. "
                        "Check the audio file or folder and try again."
                    )
                    st.stop()
            except Exception as _e:
                st.error(f"Audio staging failed: {_e}")
                st.stop()

            problems = []
            if out_df["file_id"].astype(str).str.strip().eq("").any():
                problems.append("Some rows have empty file_id.")
            if out_df["file_path"].astype(str).str.strip().eq("").any():
                problems.append("Some rows have empty file_path.")
            if out_df["detection_id"].astype(str).str.strip().eq("").any():
                problems.append("Some rows have empty detection_id.")
            if out_df["species_name"].astype(str).str.strip().eq("").any():
                problems.append("Some rows have empty species_name.")
            if out_df["detection_id"].duplicated().any():
                problems.append("detection_id must be unique; duplicates found.")
            bad_times = (
                out_df["detection_start_s"].isna() |
                out_df["detection_end_s"].isna() |
                (out_df["detection_start_s"] < 0) |
                (out_df["detection_end_s"] <= out_df["detection_start_s"])
            )
            if bad_times.any():
                problems.append("Invalid times: ensure start ≥ 0 and end > start for all rows.")
            bad_labels = ~out_df["presence_label"].astype(str).str.strip().isin(["present", "absent"])
            if bad_labels.any():
                problems.append("presence_label must be 'present' or 'absent' (lower-case).")
            bad_prob = (
                out_df["detection_probability"].isna() |
                (out_df["detection_probability"] < 0) |
                (out_df["detection_probability"] > 1)
            )
            if bad_prob.any():
                problems.append("detection_probability must be a real number in [0, 1] for all rows.")
            if problems:
                for p in problems:
                    st.error(p)
                st.stop()

            out_csv = out_dir / "detections_normalised.csv"
            out_df.to_csv(out_csv, index=False)

            try:
                ws_dir = project_path(proj_path, "workspace")
                ws_dir.mkdir(parents=True, exist_ok=True)
                cols = ["file_id", "file_path"]
                if "file_key" in out_df.columns:
                    cols.append("file_key")
                if "file_path_original" in out_df.columns:
                    cols.append("file_path_original")
                mp = out_df.loc[out_df["file_path"].astype(str).str.strip().ne(""), cols].drop_duplicates()
                if not mp.empty:
                    mp = mp.rename(columns={"file_id": "filename", "file_path": "path", "file_path_original": "original_path"})
                    mp.to_csv(ws_dir / "audio_paths.csv", index=False)
                    set_status(proj_path, "audio_resolver", "ready")
            except Exception:
                pass

            manifest = {
                "adapter": "manual",
                "mapping": {
                    "file_id": file_id_col,
                    "detection_start_s": start_col,
                    "detection_end_s": end_col,
                    "presence_label": (f"{presence_col}→present/absent" if label_mode == "binary_presence_column"
                                       else (f"{label_col} (canonicalised)" if st.session_state.import_params.get("canonicalise_existing") else label_col)),
                    "species_name": species_col,
                    "detection_probability": prob_col,
                    "file_path": "file_path",
                },
                "options": {
                    "convert_ms": bool(convert_ms),
                    "assume_utc": bool(assume_utc),
                    "label_mode": label_mode,
                    "positive_tokens": st.session_state.import_params.get("positive_tokens") if label_mode == "binary_presence_column" else None,
                    "keep_only_present": bool(st.session_state.import_params.get("keep_only_present")) if label_mode == "binary_presence_column" else None,
                    "canonicalise_existing": bool(st.session_state.import_params.get("canonicalise_existing")) if label_mode == "use_label_column" else None,
                    "present_value_for_existing": st.session_state.import_params.get("present_value_for_existing") if label_mode == "use_label_column" else None,
                    "te_factor_default": float(te_factor_default),
                },
                "input_file": st.session_state.import_params.get("filename"),
                "input_files": st.session_state.import_params.get("input_files"),
                "rows_out": int(len(out_df)),
                "created_at": datetime.now(dt_timezone.utc).isoformat(),
                "app_version": "pamalytics_studio 0.3.0",
            }
            ws_dir = project_path(proj_path, "workspace")
            ws_dir.mkdir(parents=True, exist_ok=True)
            (ws_dir / "ingest_mapping.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            st.session_state.import_last_saved = str(out_csv)
            set_status(proj_path, "import_results", "ready")
            st.session_state["manual_ingest_ready"] = True
            st.success(f"Saved normalised detections to: `{out_csv}`")

    ok_to_launch = False
    if norm_csv.exists():
        try:
            _chk = pd.read_csv(norm_csv, low_memory=False)
            need = {
                "file_id", "file_path", "detection_id",
                "detection_start_s", "detection_end_s",
                "presence_label", "species_name", "detection_probability",
            }
            if need.issubset(_chk.columns):
                _chk["detection_start_s"] = pd.to_numeric(_chk["detection_start_s"], errors="coerce")
                _chk["detection_end_s"] = pd.to_numeric(_chk["detection_end_s"], errors="coerce")
                _chk["detection_probability"] = pd.to_numeric(_chk["detection_probability"], errors="coerce")
                ok_to_launch = (
                    _chk["file_id"].astype(str).str.strip().ne("").all()
                    and _chk["file_path"].astype(str).str.strip().ne("").all()
                    and _chk["detection_id"].astype(str).str.strip().ne("").all()
                    and _chk["species_name"].astype(str).str.strip().ne("").all()
                    and (_chk["detection_probability"].between(0, 1)).all()
                    and (_chk["detection_end_s"] > _chk["detection_start_s"]).all()
                )
        except Exception:
            ok_to_launch = False

    have_norm_audio = norm_csv.exists() and audio_csv.exists()
    if have_norm_audio:
        st.divider()
        render_audio_coverage(
            norm_csv=norm_csv,
            audio_csv=audio_csv,
            use_stem_fallback=st.session_state.get("use_stem_fallback", True),
            heading="Audio coverage (manual mapping)",
        )
        render_norm_preview(norm_csv, heading="Preview mapped detections (manual)")

    if ok_to_launch:
        cL, cR = st.columns([1, 1])
        with cL:
            if st.button("Launch PAMalytics dashboard ▶", key="go_dashboard_from_manual"):
                st.switch_page("pages/40_Dashboard.py")
        with cR:
            if st.button("Back to Overview ▶", key="go_overview_from_manual"):
                st.session_state.route = "overview"
                st.rerun()
    else:
        st.info("Finalise ingestion steps prior to launching PAMalytics")


# Normalisation builder
def _build_normalised_table(
    df, source_file_col: str, start_col: str, end_col: str, score_col: Optional[str],
    ts_col: Optional[str], convert_ms: bool, assume_utc: bool, label_mode: str,
    presence_col: Optional[str], positive_tokens: str, positive_label_name: str,
    keep_only_present: bool, label_col: Optional[str], canonicalise_existing: bool,
    present_value_for_existing: str,
):
    import pandas as pd
    from datetime import datetime as _dt2

    def to_seconds(series):
        s = pd.to_numeric(series, errors="coerce")
        if convert_ms:
            s = s / 1000.0
        return s

    if label_mode == "binary_presence_column":
        tokens = {t.strip().lower() for t in positive_tokens.split(",") if t.strip() != ""}
        ser = df[presence_col].astype(str).str.strip().str.lower()
        present_mask = ser.isin(tokens)
        if keep_only_present:
            if not present_mask.any():
                empty = pd.DataFrame(columns=["source_file", "start_s", "end_s", "label", "score", "timestamp_utc"])
                return empty, ["No rows matched the chosen present value(s)."]
            base = df.loc[present_mask].copy()
        else:
            base = df.copy()
    else:
        base = df.copy()

    norm = pd.DataFrame()
    norm["source_file"] = base[source_file_col].astype(str)

    if end_col.lower() in {"duration", "duration_s"}:
        start_vals = to_seconds(base[start_col])
        dur_vals = to_seconds(base[end_col])
        norm["start_s"] = start_vals
        norm["end_s"] = start_vals + dur_vals
    else:
        norm["start_s"] = to_seconds(base[start_col])
        norm["end_s"] = to_seconds(base[end_col])

    if label_mode == "binary_presence_column":
        if keep_only_present:
            norm["label"] = positive_label_name or "present"
        else:
            ser_b = base[presence_col].astype(str).str.strip().str.lower()
            tokens_b = {t.strip().lower() for t in positive_tokens.split(",") if t.strip() != ""}
            present_mask_b = ser_b.isin(tokens_b)
            norm["label"] = (positive_label_name or "present")
            norm.loc[~present_mask_b, "label"] = "absent"
    else:
        lbl = base[label_col].astype(str)
        if canonicalise_existing:
            pv = str(present_value_for_existing).strip().lower()
            mask = lbl.str.strip().str.lower().eq(pv)
            lbl = lbl.mask(mask, "present")
            lbl = lbl.where(lbl == "present", "absent")
        norm["label"] = lbl

    norm["score"] = pd.to_numeric(base[score_col], errors="coerce") if (score_col and score_col != "—") else pd.NA

    notes = []
    if ts_col and ts_col != "—":
        raw = base[ts_col].astype(str).str.strip()
        ts = pd.to_datetime(raw, errors="coerce", utc=False)
        if ts.isna().mean() > 0.2:
            def parse_custom(x: str):
                x = x.strip()
                try:
                    if len(x) == 15 and x[8] == "_" and x[:8].isdigit() and x[9:].isdigit():
                        return _dt2.strptime(x, "%Y%m%d_%H%M%S")
                    if len(x) == 14 and x.isdigit():
                        return _dt2.strptime(x, "%Y%m%d%H%M%S")
                except Exception:
                    return None
                return None
            parsed = raw.apply(parse_custom)
            ts = ts.where(~ts.isna(), pd.to_datetime(parsed, errors="coerce", utc=False))
        try:
            tzinfo = ts.dt.tz
        except Exception:
            tzinfo = None
        if tzinfo is None:
            if assume_utc:
                try:
                    ts = ts.dt.tz_localize("UTC")
                    notes.append("Naïve datetimes interpreted as UTC (+00:00).")
                except Exception:
                    pass
            else:
                notes.append("Datetimes are timezone-naïve; consider enabling ‘Interpret as UTC’.")
        else:
            try:
                ts = ts.dt.tz_convert("UTC")
                notes.append("Datetimes converted to UTC (+00:00).")
            except Exception:
                pass
        norm["timestamp_utc"] = ts

    return norm, notes


# Metadata join (Metadata mapping)
def view_metadata() -> None:
    hide_chrome(True, True)
    import pandas as pd
    import os as _os
    if not st.session_state.get("auth_user"):
        st.session_state.route = "login"
        st.rerun()
    if not st.session_state.get("current_project"):
        st.session_state.route = "hub"
        st.rerun()

    proj_path = Path(st.session_state.current_project)
    st.title("Metadata mapping — Join metadata")
    st.caption("Upload a metadata table (e.g., site, lat, lon, recorder). Map a join key and save enriched detections.")
    nav_row("Back to Audio mapping", "locate_audio", "Back to Overview", "overview", key_prefix="meta_top")
    if st.columns([1, 1, 1])[2].button("Skip Metadata for now"):
        set_status(proj_path, "metadata_joins", "skipped")
        st.session_state.route = "overview"
        st.rerun()

    norm_csv = project_path(proj_path, "data_normalised") / "detections_normalised.csv"
    if not norm_csv.exists():
        st.error("No normalised detections found. Please complete Data mapping first.")
        return

    det = pd.read_csv(norm_csv)
    if det.empty:
        st.error("Detections table is empty.")
        return
    det["basename"] = det.get("source_file", det.get("file_id", "")).astype(str).apply(lambda p: _os.path.basename(p))
    det["stem"] = det["basename"].str.replace(r"\.[^.]+$", "", regex=True)
    det["recorder_id"] = det["basename"].apply(lambda n: n.split("_", 1)[0] if "_" in n else n)

    with st.expander("Preview derived columns", expanded=False):
        st.dataframe(det[["basename", "recorder_id"]].head(20), width='stretch')

    st.subheader("1) Upload metadata table")
    up = st.file_uploader("Upload metadata CSV / TSV / Parquet", type=["csv", "tsv", "parquet"], key="meta_up")
    if up is None:
        st.info("Select a metadata file to begin.")
        return

    try:
        if up.name.endswith(".parquet"):
            meta = pd.read_parquet(up)
        else:
            try:
                meta = pd.read_csv(up)
            except Exception:
                up.seek(0)
                meta = pd.read_csv(up, sep="\t")
    except Exception as e:
        st.error(f"Could not read metadata: {e}")
        return
    if meta.empty:
        st.error("Metadata file is empty.")
        return

    st.subheader("2) Choose join keys")
    det_key = st.selectbox("Detections key", options=["recorder_id", "basename", "stem", "source_file", "file_id"], index=0)
    meta_cols = list(meta.columns)
    meta_key = st.selectbox("Metadata key (column in uploaded table)", options=meta_cols)

    st.subheader("3) Preview join")
    preview_rows = min(200, len(det))
    merged = det.merge(meta, left_on=det_key, right_on=meta_key, how="left")
    st.dataframe(merged.head(preview_rows), width='stretch')
    join_rate = 100.0 * (1.0 - float(merged[meta_key].isna().mean()))
    st.info(f"Join coverage: **{len(merged) and join_rate:.1f}%** of detections joined with metadata (via `{det_key} == {meta_key}`).")

    if _btn("Save enriched detections"):
        out_csv = project_path(proj_path, "data_normalised") / "detections_enriched.csv"
        merged.to_csv(out_csv, index=False)
        set_status(proj_path, "metadata_joins", "ready")
        st.success(f"Saved enriched detections to: `{out_csv}`")
        nav_row("Back to Overview", "overview", "Launch dashboard", "dashboard", key_prefix="meta_after_save")


# Dashboard
def _load_renderer(module_stem: str, func_name: str):
    """
    Try, in order:
      1) import as a top-level module on sys.path
      2) load case-insensitive from scripts/
      3) load case-insensitive from scripts/pages/
    Return callable or None.
    """
    import importlib
    import importlib.util

    for cand in {module_stem, module_stem.capitalize()}:
        try:
            mod = importlib.import_module(cand)
            fn = getattr(mod, func_name, None)
            if callable(fn):
                return fn
        except ModuleNotFoundError:
            pass

    def _scan_and_load(folder: Path, stem: str):
        for p in folder.glob("*.py"):
            if p.stem.lower() == stem.lower() or stem.lower() in p.stem.lower():
                spec = importlib.util.spec_from_file_location(p.stem, p)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore
                    fn = getattr(mod, func_name, None)
                    if callable(fn):
                        return fn
        return None

    fn = _scan_and_load(SCRIPTS_DIR, module_stem)
    if fn:
        return fn
    pages_dir = SCRIPTS_DIR / "pages"
    if pages_dir.exists():
        fn = _scan_and_load(pages_dir, module_stem)
        if fn:
            return fn
    return None


def view_dashboard() -> None:
    hide_chrome(hide_sidebar=False, hide_header=True)

    import pandas as pd

    if not st.session_state.get("auth_user"):
        st.session_state.route = "login"
        st.rerun()
    if not st.session_state.get("current_project"):
        st.session_state.route = "hub"
        st.rerun()

    proj_path = Path(st.session_state.current_project)

    st.title("PAMalytics")

    page = st.session_state.get("pa_page", "Dashboard")

    df_det, notes = build_analysis_dataset(proj_path, use_stem_fallback=True)
    if df_det is None or df_det.empty:
        st.error("No matched detections with audio. Complete Import → Audio mapping → Metadata mapping first.")
        nav_row("Back to Overview", "overview", key_prefix="pa_err")
        return

    audio_csv = project_path(proj_path, "workspace") / "audio_paths.csv"
    try:
        audio_map = pd.read_csv(audio_csv)
    except Exception as e:
        st.error(f"Could not read audio map: {e}")
        return

    sources = ensure_detection_clips(proj_path, df_det, audio_map)

    with st.expander("Export", expanded=False):
        st.caption("Export everything currently available in the dashboard (all columns preserved).")
        exp_col1, exp_col2 = st.columns([1, 3])
        with exp_col1:
            if st.button("Export corrected CSV", key="export_corrected_csv"):
                try:
                    exports_dir = project_path(proj_path, "exports")
                    exports_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now(dt_timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    out_csv = exports_dir / f"pamalytics_export_{ts}.csv"
                    df_det.to_csv(out_csv, index=False)
                    st.success(f"Exported to: `{out_csv}`")
                except Exception as e:
                    st.error(f"Export failed: {e}")

    def _render(page_key: str, func_name: str):
        fn = _load_renderer(page_key, func_name)
        if not fn:
            st.error(f"Could not find a renderer for '{page_key}' in {SCRIPTS_DIR} or {SCRIPTS_DIR / 'pages'}.")
            return
        try:
            fn(df_det, sources)
        except TypeError as e:
            st.error(f"Dashboard entrypoint {func_name}() failed: {e}")
        except Exception as e:
            st.error(f"Error in {page_key}: {e}")

    if page == "Dashboard":
        _render("dashboard", "render_dashboard")
    elif page == "Validation":
        _render("validation", "render_validation")
    elif page == "Settings":
        _render("settings", "render_settings")
    elif page == "Occupancy":
        _render("occupancy", "render_occupancy")

    nav_row("Back to Overview", "overview", key_prefix="pa_bottom")


# Router
route = st.session_state.get("route", "login")
if route == "login":
    view_login()
elif route == "hub":
    view_hub()
elif route == "overview":
    view_overview()
elif route == "import":
    view_import_results()
elif route == "metadata":
    view_metadata()
elif route == "dashboard":
    st.switch_page("pages/40_Dashboard.py")
    st.stop()
else:
    st.session_state.route = "login"
    st.rerun()
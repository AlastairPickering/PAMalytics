# studio/core/pa_data.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import json, os
import pandas as pd

@dataclass(frozen=True)
class PADataset:
    detections: pd.DataFrame           # only rows with audio mapped
    audio_map: Optional[pd.DataFrame]  # from workspace/audio_paths.csv
    meta_present: bool                 # True if detections_enriched.csv used
    sources: Dict[str, str]            # file paths used

def _pp(project_dir: Path, *keys: str) -> Path:
    data = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    paths = data.get("paths", {})
    base = project_dir / paths.get(keys[0], keys[0])
    for k in keys[1:]:
        base = base / k
    return base.resolve()

def _first_existing(*cands: Path) -> Optional[Path]:
    for c in cands:
        if c and Path(c).exists(): return Path(c)
    return None

def load_pa_dataset(project_dir: Path) -> PADataset:
    project_dir = Path(project_dir)

    norm_csv = _pp(project_dir, "data_normalised") / "detections_normalised.csv"
    enr_csv  = _pp(project_dir, "data_normalised") / "detections_enriched.csv"
    audio_csv= _pp(project_dir, "workspace")       / "audio_paths.csv"

    det_path = _first_existing(enr_csv, norm_csv)
    if not det_path:
        raise FileNotFoundError("No detections file found. Complete the import steps first.")

    det = pd.read_csv(det_path)
    if det.empty:
        raise ValueError("Detections table is empty.")

    # Matching keys
    det["_basename"]   = det["source_file"].astype(str).apply(lambda p: os.path.basename(p).strip())
    det["_lower_name"] = det["_basename"].str.lower()
    det["_stem"]       = det["_lower_name"].str.replace(r"\.[^.]+$", "", regex=True)

    audio_df = None
    if audio_csv.exists():
        audio_df = pd.read_csv(audio_csv)
        if not audio_df.empty and "filename" in audio_df.columns:
            mp = audio_df.assign(_lower_name=audio_df["filename"].astype(str).str.strip().str.lower())
            mp["_stem"] = mp["_lower_name"].str.replace(r"\.[^.]+$", "", regex=True)

            # exact filename match
            det = det.merge(mp[["_lower_name","path"]], on="_lower_name", how="left")

            # unique-stem fallback for unresolved
            need = det["path"].isna()
            if need.any():
                stem_counts = mp["_stem"].value_counts()
                unique_stems = set(stem_counts[stem_counts == 1].index)
                if unique_stems:
                    stem_join = det.loc[need, ["_stem"]].merge(mp[["_stem","path"]], on="_stem", how="left")
                    det.loc[need, "path"] = stem_join["path"].values

    # Keep only rows that have audio
    if "path" in det.columns:
        det = det[det["path"].notna()].copy()

    # Convenience cols
    det["basename"]    = det["_basename"]
    det["recorder_id"] = det["basename"].apply(lambda n: n.split("_", 1)[0] if "_" in n else n)
    det.drop(columns=[c for c in ["_basename","_lower_name","_stem"] if c in det.columns], inplace=True)

    return PADataset(
        detections=det,
        audio_map=audio_df if audio_csv.exists() else None,
        meta_present=enr_csv.exists(),
        sources={"detections": str(det_path), "audio_map": str(audio_csv) if audio_csv.exists() else "", "project": str(project_dir)},
    )

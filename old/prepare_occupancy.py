# scripts/prepare_occupancy.py
from __future__ import annotations
import re
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# helpers
def _find_probability_column(df: pd.DataFrame) -> Optional[str]:
    direct = [c for c in df.columns if c.lower() in
              ("probability", "prob", "pred_prob", "pred_probability", "score", "p")]
    if direct:
        return direct[0]
    for c in df.columns:
        if "prob" in c.lower():
            return c
    return None

def _find_filename_like_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["filename", "audio_file", "file", "clip_id", "clip", "path"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == "object":
            return c
    return None

def _stem(s: str) -> str:
    return Path(str(s)).stem

def _parse_dt_full(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\D", "", regex=True)
    dt14 = pd.to_datetime(s.str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    mask = dt14.isna()
    if mask.any():
        dt8 = pd.to_datetime(s.str.slice(0, 8), format="%Y%m%d", errors="coerce")
        dt14[mask] = dt8[mask]
    return dt14

def _infer_dt_from_filename(stem: str) -> Optional[pd.Timestamp]:
    s = re.sub(r"[^0-9]", "", stem)
    if len(s) >= 14:
        return pd.to_datetime(s[:14], format="%Y%m%d%H%M%S", errors="coerce")
    if len(s) >= 8:
        return pd.to_datetime(s[:8], format="%Y%m%d", errors="coerce")
    return None

def _split_rec_and_dt(stem: str) -> Tuple[str, Optional[pd.Timestamp]]:
    parts = stem.split("_", 1)
    rec = parts[0] if parts else stem
    dt = _infer_dt_from_filename(stem)
    return rec, dt

def _write_matrix_csv(matrix: pd.DataFrame, out_path: Path, date_fmt: str = "%d/%m/%Y") -> None:
    """Reset index to 'site', format date columns nicely, and write CSV."""
    m = matrix.copy()
    m = m.reset_index().rename(columns={"index": "site"})
    new_cols = ["site"] + [c.strftime(date_fmt) if isinstance(c, pd.Timestamp) else str(c) for c in m.columns[1:]]
    m.columns = new_cols
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out_path, index=False)

# core transform
def build_occupancy_inputs(
    seg_csv: Path,
    out_long_csv: Path,
    out_matrix_csv: Path,
    tau: float = 0.5,
    file_seconds: int = 60,
    segment_seconds: int = 10,
    prob_source: str = "max",   
    out_prob_matrix_csv: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    From a segment-level CSV, produce:
      1) file-level summaries (internal),
      2) site×day long table with detect (0/1), effort, daily prob summaries,
      3) binary detection matrix (0/1),
      4) probability matrix (0..1) from daily max/mean probability.

    Returns (df_seg_used, df_long, matrix_bin, matrix_prob)
    """
    # load & normalize
    if seg_csv.suffix.lower() in (".xlsx", ".xls"):
        seg = pd.read_excel(seg_csv)
    else:
        seg = pd.read_csv(seg_csv, low_memory=False)
    if seg.empty:
        raise ValueError("Segment CSV is empty.")

    fname_col = _find_filename_like_column(seg)
    prob_col  = _find_probability_column(seg)
    if not fname_col or not prob_col:
        raise ValueError("Could not detect filename/probability columns in segment CSV.")

    seg = seg.copy()
    seg["filename_stem"] = seg[fname_col].astype(str).map(_stem).str.lower()
    seg[prob_col] = pd.to_numeric(seg[prob_col], errors="coerce")
    seg = seg.dropna(subset=[prob_col, "filename_stem"])

    # normalize 0..1 
    if seg[prob_col].max() > 1 and seg[prob_col].max() <= 100:
        seg[prob_col] = seg[prob_col] / 100.0
    seg[prob_col] = seg[prob_col].clip(0, 1)

    # timestamps
    if "date_time" in seg.columns:
        dt = _parse_dt_full(seg["date_time"])
    else:
        dt = seg["filename_stem"].map(lambda s: _infer_dt_from_filename(s))
    seg["dt"] = pd.to_datetime(dt, errors="coerce")
    seg["date"] = seg["dt"].dt.normalize()

    # recorder id
    if "recorder_id" in seg.columns:
        seg["recorder_id"] = seg["recorder_id"].astype(str)
    else:
        seg["recorder_id"] = seg["filename_stem"].map(lambda s: _split_rec_and_dt(s)[0])

    # file-level summaries
    file_sum = (
        seg.groupby(["recorder_id", "filename_stem", "date"], dropna=False)
           .agg(
               prob_max=(prob_col, "max"),
               prob_mean=(prob_col, "mean"),
               n_segments=(prob_col, "size"),
               n_pos=(prob_col, lambda x: int((x >= tau).sum())),
           )
           .reset_index()
    )
    file_sum["file_detect"] = (file_sum["prob_max"] >= tau).astype(int)
    file_sum["effort_min"] = (file_sum["n_segments"] * segment_seconds) / 60.0
    file_sum.loc[file_sum["effort_min"] <= 0, "effort_min"] = file_seconds / 60.0

    # site×day long table
    long = (
        file_sum.groupby(["recorder_id", "date"], dropna=False)
                .agg(
                    detect=("file_detect", lambda x: int((x >= 1).any())),
                    files_n=("filename_stem", "nunique"),
                    effort_min=("effort_min", "sum"),
                    prob_day_max=("prob_max", "max"),
                    prob_day_mean=("prob_mean", "mean"),
                )
                .reset_index()
                .rename(columns={"recorder_id": "site"})
    )
    long = long.sort_values(["site", "date"]).reset_index(drop=True)
    out_long_csv.parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_long_csv, index=False)

    # matrices
    matrix_bin = long.pivot(index="site", columns="date", values="detect").fillna(0).astype(int)
    _write_matrix_csv(matrix_bin, out_matrix_csv)

    score_col = "prob_day_max" if prob_source.lower().startswith("max") else "prob_day_mean"
    matrix_prob = long.pivot(index="site", columns="date", values=score_col)
    # prefer blanks for missing; if you want explicit 0 use .fillna(0)
    if out_prob_matrix_csv is None:
        out_prob_matrix_csv = out_matrix_csv.with_name("occupancy_probability_matrix.csv")
    _write_matrix_csv(matrix_prob, out_prob_matrix_csv)

    return seg, long, matrix_bin, matrix_prob


# CLI
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Prepare occupancy inputs from segment-level results.")
    ap.add_argument("--segments", required=True, type=Path, help="Path to segment-level CSV/XLSX")
    ap.add_argument("--out-long", required=True, type=Path, help="Output CSV for site×day long table")
    ap.add_argument("--out-matrix", required=True, type=Path, help="Output CSV for binary detection matrix")
    ap.add_argument("--tau", type=float, default=0.5, help="Probability threshold at file level")
    ap.add_argument("--file-seconds", type=int, default=60, help="Approx file duration (effort fallback)")
    ap.add_argument("--segment-seconds", type=int, default=10, help="Segment length used in classifier")
    ap.add_argument("--prob-source", choices=["max", "mean"], default="max",
                    help="Which daily summary to use for the probability matrix")
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    seg_path = args.segments
    build_occupancy_inputs(
        seg_path,
        args.out_long,
        args.out_matrix,
        tau=args.tau,
        file_seconds=args.file_seconds,
        segment_seconds=args.segment_seconds,
        prob_source=args.prob_source,
    )

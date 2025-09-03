# scripts/pages/1_Settings.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

st.set_page_config(page_title="Settings", layout="wide")

st.title("Settings")

# Session + paths
def _ss_get(key, default):
    return st.session_state.get(key, default)

# repo root ≈ two levels up from this file: scripts/pages/1_Settings.py -> repo/
try:
    APP_ROOT = Path(__file__).resolve().parents[2]
except Exception:
    APP_ROOT = Path.cwd()

RAW_AUDIO_DIR = Path(_ss_get("RAW_AUDIO_DIR", APP_ROOT))
RESULTS_DIR   = Path(_ss_get("RESULTS_DIR", APP_ROOT / "results"))

DEFAULT_AUDIO_DIR = Path(_ss_get("audio_base_dir", RAW_AUDIO_DIR / "processed" / "present"))
DEFAULT_FILENAME_LEVEL = Path(_ss_get("filename_level_path", RESULTS_DIR / "filename_level.csv"))

# Helpers
def _resolve_path(pth: str) -> Path:
    """Resolve a possibly relative path against sensible bases (CWD, repo root, RESULTS_DIR)."""
    p = Path(pth).expanduser()
    candidates = [
        p,
        Path.cwd() / p,
        APP_ROOT / p,
        RESULTS_DIR / p.name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # first attempt

def normalise_filename(s: pd.Series) -> pd.Series:
    """Return the stem (no extension)."""
    return s.astype(str).map(lambda x: Path(x).stem)

def _presence_from_columns(df_seg: pd.DataFrame, tau: float) -> np.ndarray:
    """
    Produce a boolean presence vector for segments using:
    - probability >= tau   (if 'probability' exists), else
    - prediction:
        * if looks continuous in [0,1], use >= tau,
        * else treat nonzero as present (>=1).
    """
    if "probability" in df_seg.columns:
        vals = pd.to_numeric(df_seg["probability"], errors="coerce")
        return (vals >= tau).to_numpy()

    if "prediction" in df_seg.columns:
        vals = pd.to_numeric(df_seg["prediction"], errors="coerce")
        finite = vals.dropna()
        if not finite.empty and finite.min() >= 0.0 and finite.max() <= 1.0 and finite.nunique() > 2:
            return (vals >= tau).to_numpy()
        return (vals >= 1).to_numpy()

    raise ValueError("Segment file must contain 'probability' or 'prediction'.")

def segments_to_filename_level(
    df_seg: pd.DataFrame,
    tau: float = 0.5,
    k: Optional[int] = None,
    n: Optional[int] = None,
    require_consecutive: bool = True,
) -> pd.DataFrame:
    """
    Convert segment-level results to filename-level.
      - Vanilla: any segment present (per tau) => 'present'
      - k-of-n consecutive: any sliding window of size n with >= k presences => 'present'
    Expected columns: 'filename', 'probability', 'prediction'.
    Optional: 'segment_idx' and/or 'start_time_s' for ordering; plus metadata columns are preserved (first row per file).
    """
    if "filename" not in df_seg.columns:
        raise ValueError("Segment file must contain a 'filename' column.")
    if not (("probability" in df_seg.columns) or ("prediction" in df_seg.columns)):
        raise ValueError("Need 'probability' or 'prediction' in segment file.")

    # Decide inside-file sort keys
    sort_cols = []
    if "segment_idx" in df_seg.columns:
        sort_cols.append("segment_idx")
    if "start_time_s" in df_seg.columns:
        sort_cols.append("start_time_s")
    if not sort_cols:
        sort_cols = ["filename"]  # stable group order

    out_rows = []
    for fname, g in df_seg.groupby("filename", sort=False):
        g = g.sort_values(sort_cols, kind="mergesort")

        pres = _presence_from_columns(g, tau=tau).astype(int)

        # Decide presence
        present = False
        if k and n and (n > 0) and (k > 0) and (k <= n) and len(pres) > 0:
            if require_consecutive:
                if len(pres) >= n:
                    win = np.convolve(pres, np.ones(n, dtype=int), mode="valid")
                    present = bool((win >= k).any())
                else:
                    present = pres.sum() >= k
            else:
                present = pres.sum() >= k
        else:
            present = bool(pres.any())

        # Carry optional metadata (first row)
        meta_cols = [c for c in [
            "recorder_id", "date_time", "utm_x", "utm_y",
            "station_name", "Elevation", "location_situation"
        ] if c in g.columns]
        meta = {c: g[c].iloc[0] for c in meta_cols}

        out_rows.append({
            "filename": fname,
            "FinalLabel": "present" if present else "absent",
            **meta
        })

    df_out = pd.DataFrame(out_rows)
    df_out["filename_stem"] = normalise_filename(df_out["filename"])
    return df_out

def _detect_tau_from_probs_and_preds(df: pd.DataFrame):
    """
    Return (tau_star, info_dict) if we can infer a sensible threshold; else (None, info).
    Requires both 'probability' and 'prediction' columns.
    - Exact case: max(prob | y=0) < min(prob | y=1). Any tau in (max0, min1] works; we return the midpoint.
    - Otherwise: choose tau among unique probs that minimises disagreement with y.
    """
    if "probability" not in df.columns or "prediction" not in df.columns:
        return None, {"reason": "Need both 'probability' and 'prediction'."}

    probs = pd.to_numeric(df["probability"], errors="coerce")
    preds = pd.to_numeric(df["prediction"], errors="coerce").round().astype("Int64")

    mask = probs.notna() & preds.notna()
    if not mask.any():
        return None, {"reason": "No valid overlapping values."}

    p = probs[mask].to_numpy()
    y = preds[mask].astype(int).to_numpy()

    if (y == 1).any() and (y == 0).any():
        max0 = p[y == 0].max(initial=-np.inf)
        min1 = p[y == 1].min(initial=np.inf)
    else:
        max0, min1 = -np.inf, np.inf

    # Exact separability
    if max0 < min1:
        tau_star = float((max0 + min1) / 2.0)
        return tau_star, {"exact": True, "range": (float(max0), float(min1)), "error_rate": 0.0}

    # Otherwise, pick tau that minimises disagreement
    uniq = np.unique(p[~np.isnan(p)])
    best_tau, best_err = None, 1.0
    for t in uniq:
        yhat = (p >= t).astype(int)
        err = float((yhat != y).mean())
        if err < best_err:
            best_err, best_tau = err, float(t)

    if best_tau is None:
        return None, {"reason": "No unique probabilities to test."}
    return best_tau, {"exact": False, "error_rate": best_err}

# Audio folder
st.subheader("Audio folder")

col_a1, col_a2 = st.columns([1.6, 0.4])
with col_a1:
    audio_dir_str = st.text_input(
        "Path to audio clips folder",
        value=str(DEFAULT_AUDIO_DIR),
        help="Default is 'processed/present'."
    )
with col_a2:
    if Path(audio_dir_str).exists():
        st.success("Folder found.")
    else:
        st.warning("Folder does not exist (check path).")

if st.button("Save audio folder"):
    st.session_state["audio_base_dir"] = audio_dir_str
    st.success(f"Audio folder set to: {audio_dir_str}")

st.markdown("---")

# Data source
st.subheader("Data source")

src_mode = st.radio(
    "Choose what the dashboard should use",
    ["Use filename-level CSV/XLSX", "Build from segment-level file"],
    index=0,
    horizontal=True,
)

# A) Use an existing filename-level file
if src_mode == "Use filename-level CSV/XLSX":
    col_f1, col_f2 = st.columns([1.6, 0.4])
    with col_f1:
        fname_csv = st.text_input(
            "Path to filename-level CSV/XLSX",
            value=str(DEFAULT_FILENAME_LEVEL),
            help="Must contain at least 'filename' and 'FinalLabel' columns."
        )
    with col_f2:
        p = _resolve_path(fname_csv)
        st.caption(f"Resolved path: `{p}`")
        st.write("✅ Found" if p.exists() else "⚠️ Not found")

    if p.exists():
        # Peek to see if UserLabel exists (don’t mutate unless the user opts in)
        try:
            probe = pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)
        except Exception as e:
            probe = None
            st.warning(f"Could not preview file: {e}")

        missing_ul = (probe is not None) and ("UserLabel" not in probe.columns)
        if missing_ul:
            st.info("This file has no 'UserLabel' column. You can add an empty one now (recommended).")
            if st.checkbox("Add empty 'UserLabel' column to this file now", value=False):
                try:
                    probe["UserLabel"] = ""
                    # Save back preserving type
                    if p.suffix.lower() in (".xlsx", ".xls"):
                        try:
                            with pd.ExcelWriter(p, engine="openpyxl", mode="w") as w:
                                probe.to_excel(w, sheet_name="filename_level", index=False)
                        except Exception:
                            probe.to_excel(p, index=False)
                    else:
                        probe.to_csv(p, index=False)
                    st.success("Added 'UserLabel' column.")
                except Exception as e:
                    st.error(f"Failed to add column: {e}")

    use_now = st.button("Use this file in dashboard")
    if use_now:
        p = _resolve_path(fname_csv)
        if not p.exists():
            st.error(f"File not found: {p}")
        else:
            st.session_state["filename_level_path"] = str(p)
            st.success(f"Dashboard will use: {p}")

# B) Build from a segment-level file
else:
    st.markdown("**Segment-level input** (e.g. `merged_classification_results.csv` or `.xlsx`)")

    col_s1, col_s2 = st.columns([1.6, 0.4])
    with col_s1:
        seg_path_str = st.text_input(
            "Path to segment-level CSV/XLSX",
            value=str(RESULTS_DIR / "merged_classification_results.csv"),
        )
    with col_s2:
        seg_p_preview = _resolve_path(seg_path_str)
        st.caption(f"Resolved path: `{seg_p_preview}`")
        st.write("✅ Found" if seg_p_preview.exists() else "⚠️ Not found")

    # Threshold and k-of-n controls
    st.markdown("**Segment decision**")
    c0, c1, c2, c3, c4 = st.columns([0.7, 0.55, 0.5, 0.5, 0.9])

    tau_default = float(st.session_state.get("tau_guess", 0.50))
    with c0:
        tau = st.number_input("Threshold (tau)", min_value=0.0, max_value=1.0, value=tau_default, step=0.01,
                              help="Probability cutoff for a segment to be 'present'.")
    with c1:
        if st.button("Auto-detect τ from file", use_container_width=True):
            seg_p = _resolve_path(seg_path_str)
            if not seg_p.exists():
                st.error(f"Segment-level file not found: {seg_p}")
            else:
                try:
                    df_probe = pd.read_excel(seg_p) if seg_p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(seg_p)
                    tau_star, info = _detect_tau_from_probs_and_preds(df_probe)
                    if tau_star is None:
                        st.warning(f"Could not infer τ. {info.get('reason','')}")
                    else:
                        st.session_state['tau_guess'] = round(float(tau_star), 3)
                        msg = f"Suggested τ ≈ {st.session_state['tau_guess']:.3f}"
                        if info.get("exact"):
                            lo, hi = info["range"]
                            msg += f" (exact separation: {lo:.3f}–{hi:.3f})"
                        else:
                            msg += f" (min. disagreement ≈ {info.get('error_rate', 0.0):.2%})"
                        st.success(msg)
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to auto-detect τ: {e}")

    with c2:
        enable_kofn = st.checkbox("Enable k-of-n", value=False)
    with c3:
        n = st.number_input("n (window size)", min_value=1, max_value=24, value=3, step=1, disabled=not enable_kofn)
    with c4:
        k = st.number_input("k (min presences)", min_value=1, max_value=24, value=2, step=1, disabled=not enable_kofn)

    st.caption("Tip: if training used τ ≈ 0.35 with 2-in-3, try τ in 0.30–0.40 here to match totals.")

    # Output target
    out_default = RESULTS_DIR / "filename_level_converted.csv"
    out_path_str = st.text_input("Output filename-level CSV path", value=str(out_default))

    if st.button("Convert and save"):
        seg_p = _resolve_path(seg_path_str)
        if not seg_p.exists():
            st.error(f"Segment-level file not found: {seg_p}")
        else:
            try:
                # Load segment-level file
                if seg_p.suffix.lower() in (".xlsx", ".xls"):
                    df_seg = pd.read_excel(seg_p)
                else:
                    df_seg = pd.read_csv(seg_p)

                # Convert to filename-level
                df_out = segments_to_filename_level(
                    df_seg.copy(),
                    tau=float(tau),
                    k=int(k) if enable_kofn else None,
                    n=int(n) if enable_kofn else None,
                    require_consecutive=True,
                )

                # Ensure editable override column exists for ANY output
                if "UserLabel" not in df_out.columns:
                    df_out["UserLabel"] = ""

                # Save
                out_p = _resolve_path(out_path_str)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                if out_p.suffix.lower() in (".xlsx", ".xls"):
                    try:
                        with pd.ExcelWriter(out_p, engine="openpyxl", mode="w") as w:
                            df_out.to_excel(w, sheet_name="filename_level", index=False)
                    except Exception:
                        df_out.to_excel(out_p, index=False)
                else:
                    df_out.to_csv(out_p, index=False)

            except Exception as e:
                st.error(f"Failed to convert/save: {e}")
            else:
                st.success(f"Saved filename-level file to {out_p}")
                st.write("Preview (first 20 rows):")
                st.dataframe(df_out.head(20), use_container_width=True)

                # Offer to activate
                if st.button("Use this new file in the dashboard"):
                    st.session_state["filename_level_path"] = str(out_p)
                    st.success("Active filename-level file updated.")

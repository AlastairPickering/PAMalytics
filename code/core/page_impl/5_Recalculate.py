# scripts/pages/5_Recalculate.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import streamlit as st

# helpers

def _num(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def _ensure_canonical(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    if "basename" not in df.columns:
        src = df.get("file_id", df.get("source_file", ""))
        df["basename"] = src.astype(str).map(lambda p: Path(p).name)
    if "filename_stem" not in df.columns:
        df["filename_stem"] = df["basename"].astype(str).map(lambda s: Path(s).stem.lower())

    if "species_name" not in df.columns:
        if "class" in df.columns:
            df["species_name"] = df["class"].astype(str)
        else:
            df["species_name"] = ""

    if "detection_probability" not in df.columns:
        def _best_prob(row: pd.Series) -> float:
            for c in ("probability", "prob", "score", "class_prob", "det_prob"):
                if c in row and pd.notna(row[c]):
                    try:
                        v = float(row[c])
                        if np.isfinite(v):
                            return v
                    except Exception:
                        pass
            return np.nan
        df["detection_probability"] = df.apply(_best_prob, axis=1)
    df["detection_probability"] = pd.to_numeric(df["detection_probability"], errors="coerce")

    if "detection_start_s" not in df.columns and "start_s" in df.columns:
        df["detection_start_s"] = pd.to_numeric(df["start_s"], errors="coerce")
    if "detection_end_s" not in df.columns and "end_s" in df.columns:
        df["detection_end_s"] = pd.to_numeric(df["end_s"], errors="coerce")

    return df

def _presence_k_of_n(
    times_s: np.ndarray,
    present_mask: np.ndarray,
    *,
    k: Optional[int],
    window_s: Optional[float],
    require_consecutive: bool,
) -> bool:
    idx = np.where(present_mask)[0]
    if k is None:
        return bool(idx.size > 0)
    if idx.size == 0:
        return False

    if require_consecutive:
        longest, cur = 1, 1
        for i in range(1, idx.size):
            if idx[i] == idx[i - 1] + 1:
                cur += 1
                longest = max(longest, cur)
            else:
                cur = 1
        if longest >= k:
            return True
        if window_s and window_s > 0:
            t = times_s[idx]
            left = 0
            for right in range(t.size):
                while t[right] - t[left] > window_s:
                    left += 1
                if (right - left + 1) >= k:
                    return True
        return False
    else:
        if (window_s is None) or window_s <= 0:
            return bool(idx.size >= k)
        t = times_s[idx]
        left = 0
        for right in range(t.size):
            while t[right] - t[left] > window_s:
                left += 1
            if (right - left + 1) >= k:
                return True
        return False

def _summarise_effect(df_clip: pd.DataFrame) -> Dict[str, float]:
    n = len(df_clip)
    m = int(pd.to_numeric(df_clip["present_decision"], errors="coerce").fillna(0).sum())
    return {"clips": n, "present_clips": m, "present_pct": (100.0 * m / n) if n else 0.0}

# page entry

def render_recalculate(df: pd.DataFrame, sources: Dict) -> None:
    st.header("Recalculate presence")

    if df is None or df.empty:
        st.info("No detections loaded for this project.")
        return

    proj_root = Path(sources.get("project") or sources.get("project_root") or ".")
    df_det = _ensure_canonical(df)

    # Controls
    top1, top2, top3, top4, top5 = st.columns([1.0, 1.0, 1.0, 1.0, 2.0])
    with top1:
        tau = st.number_input("τ (probability ≥)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)
    with top2:
        enable_k = st.checkbox("Enable k-of-n window", value=False)
    with top3:
        k = st.number_input("k", min_value=1, max_value=50, value=2, step=1, disabled=not enable_k)
    with top4:
        window_s = st.number_input("Window (s)", min_value=0.1, max_value=120.0, value=3.0, step=0.1, disabled=not enable_k)
    with top5:
        require_consec = st.checkbox("Require consecutive above-τ", value=True, disabled=not enable_k)

    work = df_det.copy()
    work = work.dropna(subset=["filename_stem"])
    work["prob_f"] = pd.to_numeric(work["detection_probability"], errors="coerce").fillna(-np.inf)
    work["t_mid"] = np.where(
        work["detection_end_s"].notna() & work["detection_start_s"].notna(),
        (pd.to_numeric(work["detection_start_s"], errors="coerce") +
         pd.to_numeric(work["detection_end_s"], errors="coerce")) / 2.0,
        pd.to_numeric(work.get("detection_start_s"), errors="coerce")
    )

    groups = work.groupby(["filename_stem", "species_name"], dropna=False)
    rows_out: List[Dict] = []
    for (stem, sp), g in groups:
        g = g.sort_values("t_mid")
        t = pd.to_numeric(g["t_mid"], errors="coerce").fillna(0.0).to_numpy()
        above = (g["prob_f"].to_numpy() >= float(tau))
        present = _presence_k_of_n(
            t, above,
            k=int(k) if enable_k else None,
            window_s=float(window_s) if enable_k else None,
            require_consecutive=bool(require_consec and enable_k),
        )
        filename = g["basename"].iloc[0] if "basename" in g.columns else (g["file_id"].iloc[0] if "file_id" in g.columns else f"{stem}.wav")
        rec = g["recorder_id"].iloc[0] if "recorder_id" in g.columns else None

        rows_out.append({
            "filename": str(filename),
            "filename_stem": str(stem),
            "species_name": str(sp) if pd.notna(sp) else "",
            "FinalLabel": "present" if present else "absent",
            "present_decision": int(bool(present)),  # <-- added
            "tau": float(tau),
            "k": int(k) if enable_k else None,
            "window_s": float(window_s) if enable_k else None,
            "require_consecutive": bool(require_consec and enable_k),
            **({"recorder_id": rec} if rec is not None else {})
        })

    fn_level = pd.DataFrame(rows_out)
    if fn_level.empty:
        st.warning("No output rows produced.")
        return

    summary = (
        fn_level.groupby("species_name", dropna=False)
        .apply(_summarise_effect)
        .apply(pd.Series)
        .reset_index()
        .sort_values(["present_pct", "species_name"], ascending=[False, True])
    )

    s1, s2 = st.columns([1.2, 1.8])
    with s1:
        st.subheader("Overall")
        total = len(fn_level)
        present = int(pd.to_numeric(fn_level["present_decision"], errors="coerce").fillna(0).sum())
        pct = (100.0 * present / total) if total else 0.0
        st.metric("Rows", f"{total:,}")
        st.metric("Present", f"{present:,} ({pct:.1f}%)")
    with s2:
        st.subheader("By species")
        st.dataframe(summary, width='stretch')

    st.subheader("Preview (first 25)")
    st.dataframe(fn_level.head(25), width='stretch')

    out_dir = proj_root / "data_normalised"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / "filename_level_recalc.csv"

    left, right = st.columns([1, 2])
    with left:
        if st.button("Save filename-level CSV"):
            try:
                fn_level.to_csv(out_p, index=False)
                st.success(f"Saved: {out_p}")
            except Exception as e:
                st.error(f"Failed to save: {e}")
    with right:
        if out_p.exists() and st.button("Use this for the dashboard"):
            st.session_state["filename_level_path"] = str(out_p)
            st.success("Active filename-level file set for the dashboard.")

# Backwards-compat export
def render_settings(df, sources):
    return render_recalculate(df, sources)

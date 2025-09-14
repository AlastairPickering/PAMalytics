# scripts/pages/4_Occupancy.py
from pathlib import Path
import warnings
from typing import List
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from config import RESULTS_DIR

# ---- Page setup ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Occupancy")
st.title("Occupancy")
st.caption(
    "Build site×day detection histories and daily probabilities from segment-level results, "
    "then fit a single-season Bayesian occupancy model (site-level ψ, global detection p)."
)

# Let Altair render large data tables
alt.data_transformers.disable_max_rows()

# ---- Controls (top row) ----------------------------------------------------
DEFAULT_SEG = RESULTS_DIR / "merged_classification_results.csv"

top1, top2, top3, top4, top5 = st.columns([2.2, 1.2, 1.1, 1.1, 1.8])
with top1:
    seg_path_str = st.text_input("Segment-level results", str(DEFAULT_SEG))
with top2:
    tau = st.slider("τ (file detection)", 0.0, 1.0, 0.50, 0.01)
with top3:
    segment_seconds = st.number_input("Segment (s)", min_value=1, value=10, step=1)
with top4:
    file_seconds = st.number_input("File (s)", min_value=10, value=60, step=10)
with top5:
    prob_method = st.selectbox(
        "Daily probability uses",
        ["Daily max", "Cumulative across files (1 − Π(1 − p))"],
        index=0,
    )

# Modelling controls (keep near top)
m1, m2, m3, m4, m5 = st.columns([1.0, 1.0, 1.0, 1.0, 1.2])
with m1:
    draws = st.number_input("MCMC draws", min_value=200, max_value=5000, value=1000, step=100)
with m2:
    tune = st.number_input("MCMC tune", min_value=200, max_value=5000, value=1000, step=100)
with m3:
    target_accept = st.slider("Target accept", 0.70, 0.99, 0.90, 0.01)
with m4:
    chains = st.number_input("Chains", min_value=2, max_value=8, value=2, step=1)
with m5:
    fit_now = st.button("Fit occupancy model")

# ---- Helpers ---------------------------------------------------------------
def _read_segments(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Segment file not found: {path}")
        st.stop()
    if path.suffix.lower() in (".xlsx", ".xls"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_excel(path)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_csv(path, low_memory=False)
    return df

def _coerce_dt(s: pd.Series) -> pd.Series:
    """
    Parse date from 'date_time'-like strings:
      - strip non-digits
      - try %Y%m%d%H%M%S else %Y%m%d
    Return normalized (midnight) timestamps for stable day grouping.
    """
    ss = s.astype(str).str.replace(r"\D", "", regex=True)
    dt14 = pd.to_datetime(ss.str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    miss = dt14.isna()
    if miss.any():
        dt8 = pd.to_datetime(ss.str.slice(0, 8), format="%Y%m%d", errors="coerce")
        dt14[miss] = dt8[miss]
    return dt14.dt.normalize()

def _ensure_required_cols(df: pd.DataFrame, cols: List[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns in {label}: {missing}")
        st.stop()

def _day_prob(np_probs: np.ndarray, method: str) -> float:
    """Aggregate file probabilities into a single daily probability."""
    if np_probs.size == 0:
        return np.nan
    if method.startswith("Daily max"):
        return float(np_probs.max())
    # cumulative: 1 - ∏(1 - p) across files
    return float(1.0 - np.prod(1.0 - np.clip(np_probs, 0.0, 1.0)))

# ---- Build site×day long table from segment-level results ------------------
seg_path = Path(seg_path_str)
df_seg = _read_segments(seg_path)

# Expected columns from your pipeline
_req = ["filename", "recorder_id", "segment_idx", "probability", "date_time"]
_ensure_required_cols(df_seg, _req, "segment results")

df2 = df_seg.copy()
df2["filename"] = df2["filename"].astype(str)
df2["recorder_id"] = df2["recorder_id"].astype(str)
df2["probability"] = pd.to_numeric(df2["probability"], errors="coerce")
df2["segment_idx"] = pd.to_numeric(df2["segment_idx"], errors="coerce").fillna(0).astype(int)
df2["date"] = _coerce_dt(df2["date_time"])
df2 = df2[df2["date"].notna()].copy()

# File-level probability = max segment prob within the file
file_probs = (
    df2.groupby(["recorder_id", "filename", "date"], as_index=False)["probability"]
       .max()
       .rename(columns={"probability": "file_prob"})
)
file_probs["file_prob"] = pd.to_numeric(file_probs["file_prob"], errors="coerce").clip(0, 1)

# Binary file detection via τ
file_probs["file_detect"] = (file_probs["file_prob"] >= float(tau)).astype(int)

# Day-level aggregation (site×day)
day_list = []
for (site, day), g in file_probs.groupby(["recorder_id", "date"]):
    p_files = g["file_prob"].dropna().to_numpy()
    prob_day = _day_prob(p_files, prob_method)
    detect = int(g["file_detect"].max()) if len(g) else 0
    files_n = int(len(g))
    effort_min = files_n * float(file_seconds) / 60.0
    day_list.append(
        {"site": str(site), "date": pd.to_datetime(day), "prob_day": prob_day,
         "detect": detect, "files_n": files_n, "effort_min": effort_min}
    )
df_long = pd.DataFrame(day_list)

# Drop the erroneous “site” that is actually a date string
df_long = df_long[df_long["site"].astype(str) != "20250407"].reset_index(drop=True)

# Pretty dates for display; show dd/mm/YYYY (no 00:00:00)
df_long["date_str"] = pd.to_datetime(df_long["date"]).dt.strftime("%d/%m/%Y")

# ---- Headline metrics ------------------------------------------------------
n_sites = df_long["site"].nunique() if not df_long.empty else 0
n_days = df_long["date"].nunique() if not df_long.empty else 0
naive_occ = df_long.groupby("site")["detect"].max().mean() if n_sites else 0.0

h1, h2, h3, h4 = st.columns(4)
h1.metric("Sites", f"{n_sites}")
h2.metric("Days", f"{n_days}")
h3.metric("Naïve occupancy", f"{naive_occ:.2f}")
h4.metric("Mean effort (min/day)", f"{df_long['effort_min'].mean():.1f}" if not df_long.empty else "0.0")

# ---- Bayesian occupancy model (KING PLOT FIRST) ----------------------------
st.subheader("Site-level occupancy (ψ)")

if fit_now:
    try:
        import pymc as pm
        import arviz as az

        src = df_long.copy()
        for c in ("site", "date", "detect", "effort_min"):
            if c not in src.columns:
                st.error(f"Missing required column for modelling: {c}")
                st.stop()

        src["site"] = src["site"].astype(str)
        src["date"] = pd.to_datetime(src["date"], errors="coerce").dt.normalize()
        src["detect"] = pd.to_numeric(src["detect"], errors="coerce").fillna(0).astype(int)
        src["effort_min"] = pd.to_numeric(src["effort_min"], errors="coerce").fillna(0.0)
        src = src[src["date"].notna()].copy()

        # Only keep site×day cells with any effort
        obs = src[(src["effort_min"] > 0.0)].copy()
        if obs.empty:
            st.warning("No surveyed (effort>0) site×day cells to model.")
            st.stop()

        site_order_m = sorted(obs["site"].unique().tolist())
        date_order_m = sorted(obs["date"].unique().tolist())

        det_mat = (
            obs.pivot_table(index="site", columns="date", values="detect", aggfunc="max")
            .reindex(index=site_order_m, columns=date_order_m)
        ).fillna(0).astype(int)

        # Flatten observed detections by site×day where effort>0
        y_obs_list, site_idx_list = [], []
        site_to_idx = {s: i for i, s in enumerate(site_order_m)}
        for s in site_order_m:
            row = det_mat.loc[s]
            # All entries present are observed (since we filtered by effort>0)
            y_obs_list.extend(row.astype(int).tolist())
            site_idx_list.extend([site_to_idx[s]] * len(row))

        y_obs = np.asarray(y_obs_list, dtype="int8")
        site_idx = np.asarray(site_idx_list, dtype="int32")

        n_sites_m = len(site_order_m)

        with pm.Model() as occ_model:
            # Site random effects on logit(ψ)
            psi_logit = pm.Normal("psi_logit", mu=0.0, sigma=1.5, shape=n_sites_m)
            psi = pm.Deterministic("psi", pm.math.sigmoid(psi_logit))
            z = pm.Bernoulli("z", p=psi, shape=n_sites_m)

            # Global detection probability
            p_det = pm.Beta("p_det", alpha=1, beta=1)

            # Observed detection
            p_obs = pm.Deterministic("p_obs", z[site_idx] * p_det)
            y = pm.Bernoulli("y", p=p_obs, observed=y_obs)

            idata = pm.sample(
                draws=int(draws),
                tune=int(tune),
                chains=int(chains),
                cores=min(int(chains), max(cpu_count(), 1)),
                target_accept=float(target_accept),
                random_seed=42,
                progressbar=True,
            )

        # Summaries
        psi_post = idata.posterior["psi"].stack(draws=("chain", "draw")).values  # (site, draws)
        psi_mean = psi_post.mean(axis=1)
        psi_lo = np.quantile(psi_post, 0.025, axis=1)
        psi_hi = np.quantile(psi_post, 0.975, axis=1)

        pdet_post = idata.posterior["p_det"].stack(draws=("chain", "draw")).values
        pdet_mean = float(pdet_post.mean())
        pdet_lo = float(np.quantile(pdet_post, 0.025))
        pdet_hi = float(np.quantile(pdet_post, 0.975))

        # Effort days = number of surveyed days per site
        effort_days = (
            obs.groupby("site")["date"].nunique()
            .reindex(site_order_m)
            .fillna(0)
            .astype(int)
            .tolist()
        )

        site_summary = pd.DataFrame(
            {
                "site": site_order_m,
                "psi_mean": psi_mean,
                "psi_lo": psi_lo,
                "psi_hi": psi_hi,
                "effort_days": effort_days,
            }
        ).sort_values("psi_mean", ascending=False)

        # King plot: error bars + bubbles
        err = alt.Chart(site_summary).mark_rule().encode(
            x=alt.X("site:N", sort=site_summary["site"].tolist(),
                    axis=alt.Axis(title="Site", labelAngle=-45)),
            y=alt.Y("psi_lo:Q", title="Occupancy (ψ)", scale=alt.Scale(domain=[0, 1])),
            y2="psi_hi:Q",
            tooltip=[
                alt.Tooltip("site:N", title="Site"),
                alt.Tooltip("psi_mean:Q", title="ψ (mean)", format=".2f"),
                alt.Tooltip("psi_lo:Q", title="2.5%", format=".2f"),
                alt.Tooltip("psi_hi:Q", title="97.5%", format=".2f"),
                alt.Tooltip("effort_days:Q", title="Effort days"),
            ],
        )

        pts = alt.Chart(site_summary).mark_circle(opacity=0.9).encode(
            x=alt.X("site:N", sort=site_summary["site"].tolist(),
                    axis=alt.Axis(title="Site", labelAngle=-45)),
            y=alt.Y("psi_mean:Q", title="Occupancy (ψ)", scale=alt.Scale(domain=[0, 1])),
            size=alt.Size("effort_days:Q", legend=alt.Legend(title="Effort days")),
            color=alt.Color("psi_mean:Q", scale=alt.Scale(scheme="viridis"), legend=None),
        )

        st.altair_chart((err + pts).properties(height=460), use_container_width=True)
        st.caption(
            f"Global detection probability p̂: **{pdet_mean:.2f}** "
            f"(95% CI {pdet_lo:.2f}–{pdet_hi:.2f})"
        )

    except Exception as e:
        st.error("Modelling failed:")
        st.exception(e)

# ---- Discrete detection heatmaps ------------------------------------------
st.subheader("Detection heatmaps")

viz = df_long.copy()
if viz.empty:
    st.info("No data to display.")
else:
    viz["site_label"] = viz["site"].astype(str)
    viz["site_key"] = viz["site_label"].str.strip().str.lower()
    viz["date_key"] = pd.to_datetime(viz["date"], errors="coerce").dt.normalize()
    viz["date_str"] = viz["date_key"].dt.strftime("%d/%m/%Y")

    viz["detect"] = pd.to_numeric(viz.get("detect", 0), errors="coerce").fillna(0).astype(int)
    viz["prob_day"] = pd.to_numeric(viz.get("prob_day", np.nan), errors="coerce")
    viz["files_n"] = pd.to_numeric(viz.get("files_n", 0), errors="coerce").fillna(0).astype(int)
    viz["effort_min"] = pd.to_numeric(viz.get("effort_min", 0), errors="coerce").fillna(0.0)

    site_order = sorted(viz["site_label"].dropna().unique().tolist())
    all_dates = pd.to_datetime(sorted(viz["date_key"].dropna().unique()))
    date_order = [d.strftime("%d/%m/%Y") for d in all_dates]

    # Render-friendly canvas size
    CELL = 34
    width = max(CELL * max(1, len(date_order)), 300)
    height = max(CELL * max(1, len(site_order)), 300)

    # Binary heatmap (single cell per site×day)
    chart_bin = (
        alt.Chart(viz)
        .transform_aggregate(
            detect="max(detect)",
            files_n="sum(files_n)",
            effort_min="sum(effort_min)",
            groupby=["site_key", "site_label", "date_str"],
        )
        .transform_calculate(
            state="(datum.files_n > 0 || datum.effort_min > 0) ? "
                  "(datum.detect == 1 ? 'present' : 'absent') : 'no_effort'"
        )
        .mark_rect(stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("date_str:N", title="Date", sort=date_order,
                    axis=alt.Axis(labelAngle=-45, ticks=False, labelFontSize=12, titleFontSize=12)),
            y=alt.Y("site_label:N", title="Site", sort=site_order,
                    axis=alt.Axis(labelFontSize=12, titleFontSize=12)),
            color=alt.Color(
                "state:N",
                scale=alt.Scale(
                    domain=["present", "absent", "no_effort"],
                    range=["#1a73e8", "#e0e0e0", "#ffffff"],
                ),
                legend=alt.Legend(title="State"),
            ),
            tooltip=[
                alt.Tooltip("site_label:N", title="Site"),
                alt.Tooltip("date_str:N", title="Date"),
                alt.Tooltip("detect:Q", title="Detect"),
                alt.Tooltip("files_n:Q", title="# Files"),
                alt.Tooltip("effort_min:Q", title="Effort (min)", format=".1f"),
            ],
        )
        .properties(width=width, height=height)
    )
    st.markdown("**Binary presence/absence (white = no effort)**")
    st.altair_chart(chart_bin, use_container_width=True)

    # Probability heatmap (pre-aggregate to avoid Vega transform quirks)
    prob_df = (
        viz.groupby(["site_key", "site_label", "date_str"], as_index=False)
           .agg(prob_day=("prob_day", "max"),
                files_n=("files_n", "sum"),
                effort_min=("effort_min", "sum"))
    )
    prob_df = prob_df[(prob_df["files_n"] > 0) | (prob_df["effort_min"] > 0)].copy()
    prob_df["prob_day"] = pd.to_numeric(prob_df["prob_day"], errors="coerce").fillna(0.0)
    prob_df["site_label"] = prob_df["site_label"].astype(str)
    prob_df["date_str"] = prob_df["date_str"].astype(str)

    if prob_df.empty:
        st.info("No surveyed days with probability values to display.")
    else:
        # Sort orders derived from probability table itself
        try:
            date_order_prob = sorted(
                prob_df["date_str"].dropna().unique().tolist(),
                key=lambda s: pd.to_datetime(s, format="%d/%m/%Y"),
            )
        except Exception:
            date_order_prob = sorted(prob_df["date_str"].dropna().unique().tolist())
        site_order_prob = sorted(prob_df["site_label"].dropna().unique().tolist())

        chart_prob_base = (
            alt.Chart(prob_df)
            .mark_rect(stroke="white", strokeWidth=1, opacity=1)
            .encode(
                x=alt.X("date_str:N", title="Date", sort=date_order_prob,
                        axis=alt.Axis(labelAngle=-45, ticks=False, labelFontSize=12, titleFontSize=12)),
                y=alt.Y("site_label:N", title="Site", sort=site_order_prob,
                        axis=alt.Axis(labelFontSize=12, titleFontSize=12)),
                color=alt.Color(
                    "prob_day:Q",
                    scale=alt.Scale(domain=[0, 1], scheme="viridis"),
                    legend=alt.Legend(title="Daily probability", format=".2f"),
                ),
                tooltip=[
                    alt.Tooltip("site_label:N", title="Site"),
                    alt.Tooltip("date_str:N", title="Date"),
                    alt.Tooltip("prob_day:Q", title="Daily prob", format=".2f"),
                    alt.Tooltip("files_n:Q", title="# Files"),
                    alt.Tooltip("effort_min:Q", title="Effort (min)", format=".1f"),
                ],
            )
            .properties(width=width, height=height)
        )

        label_df = prob_df.copy()
        label_df["pct"] = np.round(label_df["prob_day"] * 100).astype("Int64")
        label_df["label_color"] = np.where(label_df["prob_day"] >= 0.5, "white", "black")

        chart_prob_labels = (
            alt.Chart(label_df)
            .mark_text(fontSize=11, fontWeight="bold")
            .encode(
                x=alt.X("date_str:N", sort=date_order_prob),
                y=alt.Y("site_label:N", sort=site_order_prob),
                text=alt.Text("pct:Q"),
                color=alt.Color("label_color:N", scale=None),
            )
            .properties(width=width, height=height)
        )

        st.markdown("**Probability heatmap (surveyed days only)**")
        st.altair_chart((chart_prob_base + chart_prob_labels), use_container_width=True)

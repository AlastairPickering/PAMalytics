# classify.py
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from config import EMBEDDINGS_DIR, RESULTS_DIR

# Quiet noisy warnings when loading a model saved with a different sklearn minor version
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Load production classifier bundle (pipeline + threshold)
BUNDLE_PATH = Path(
    "~/gdrive_mount/pileated_gibbon_production/models/logreg_beats_pipeline_v5.joblib"
).expanduser()

with warnings.catch_warnings():
    warnings.simplefilter("ignore", InconsistentVersionWarning)
    bundle = joblib.load(BUNDLE_PATH)

clf_pipe = bundle["pipeline"]
THRESH   = float(bundle["threshold"])
print(f"Loaded production bundle.\n  Model: {clf_pipe}\n  Decision threshold τ = {THRESH:.3f}")

def predict_proba_one(emb: np.ndarray) -> float:
    """Return P(present) for a single 1D embedding array."""
    return float(clf_pipe.predict_proba(emb.reshape(1, -1))[:, 1])

def classify_all_embeddings():
    records = []

    # Expect filenames like "rec123_20250101T000000_seg3_emb.npy"
    emb_paths = sorted(EMBEDDINGS_DIR.glob("*_seg*_emb.npy"))
    if not emb_paths:
        print(f"No segment embeddings found in {EMBEDDINGS_DIR}")
        return

    for emb_path in emb_paths:
        stem = emb_path.stem  # e.g., "rec123_20250101T000000_seg3_emb"
        try:
            filepart, segpart, _ = stem.rsplit("_", 2)
            seg_idx = int(segpart.replace("seg", ""))
        except ValueError:
            # Skip anything that doesn't match the expected pattern
            continue

        emb  = np.load(emb_path)
        prob = predict_proba_one(emb)
        pred = int(prob >= THRESH)

        records.append({
            "filename":       filepart,
            "segment_idx":    seg_idx,
            "embedding_file": str(emb_path),
            "prediction":     pred,         # decision at τ
            "probability":    prob
        })

    # Segment-level results
    df_seg = pd.DataFrame.from_records(records)
    df_seg = df_seg.sort_values(["filename", "segment_idx"]).reset_index(drop=True)

    out_seg = RESULTS_DIR / "classification_results_segments.csv"
    df_seg.to_csv(out_seg, index=False)
    print(f"Segment classification → {out_seg}")

    # File-level OR: any segment predicted positive ⇒ file positive
    df_file = (
        df_seg.groupby("filename", as_index=False)
              .agg(
                  n_segments=("segment_idx", "count"),
                  n_positive=("prediction", "sum"),
                  mean_prob=("probability", "mean"),
              )
    )
    df_file["decision"] = (df_file["n_positive"] > 0).astype(int)

    out_file = RESULTS_DIR / "classification_results_files.csv"
    df_file.to_csv(out_file, index=False)
    print(f"File-level summary → {out_file}")

if __name__ == "__main__":
    classify_all_embeddings()

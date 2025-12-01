# pipeline.py
import os
import sys
import time
import math
import shutil
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import argparse
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from sklearn.exceptions import InconsistentVersionWarning
from urllib.request import urlretrieve

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r"PySoundFile failed\. Trying audioread instead\.")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*librosa\.core\.audio\.__audioread_load.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*Audioread support is deprecated.*")

from preprocessing import (
    adaptive_bandpass_filter,
    noise_reduction,
    normalize_amplitude,
    dynamic_range_compression,
    load_beats_model,
    extract_embedding_from_array
)
from config import (
    RAW_AUDIO_DIR as CFG_RAW_AUDIO_DIR,
    RESULTS_DIR as CFG_RESULTS_DIR,
    SEGMENT_SECONDS,
    TARGET_LENGTH,
    DEVICE,
    BEATS_URL,
    BEATS_CKPT as CFG_BEATS_CKPT,   # from config (models/BEATs_iter3_plus_AS2M.pt)
)

def _looks_like_lfs_pointer(p: Path) -> bool:
    try:
        head = p.read_bytes()[:120]
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False

def _looks_like_html(p: Path) -> bool:
    try:
        head = p.read_bytes()[:128].lower()
        return head.startswith(b"<") or b"<html" in head
    except Exception:
        return False

def ensure_asset(local_path: Path, url: str, min_bytes: int = 50_000_000) -> Path:
    """
    Ensure `local_path` is a real checkpoint. If it's missing, tiny, an LFS pointer,
    or an HTML page, (re)download from `url`.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)

    def _bad_file(p: Path) -> bool:
        if not p.exists():
            return True
        try:
            sz = p.stat().st_size
        except Exception:
            return True
        # too small, LFS pointer, or HTML error page
        return (sz < min_bytes) or _looks_like_lfs_pointer(p) or _looks_like_html(p)

    if _bad_file(local_path):
        # remove junk then download the real thing
        try:
            if local_path.exists():
                local_path.unlink()
        except Exception:
            pass
        print(f"[INFO] Downloading {url} -> {local_path}")
        urlretrieve(url, local_path)
        print(f"[INFO] Downloaded {local_path} ({local_path.stat().st_size} bytes)")

        # validate again; bail out if still bad
        if _bad_file(local_path):
            raise RuntimeError(
                f"Downloaded file at {local_path} is not a valid checkpoint "
                f"(pointer/HTML/too small). Check your release asset URL."
            )
    return local_path

# Repo roots & sensible defaults
SCRIPTS_DIR = Path(__file__).resolve().parent               # .../scripts
REPO_ROOT   = SCRIPTS_DIR.parent                            # repo root
DEFAULT_BEATS_DIR  = (REPO_ROOT / "models" / "unilm" / "beats").resolve()

# Try to get bundle path from config; else fall back to repo/models
try:
    from config import MODEL_BUNDLE_PATH as CFG_BUNDLE
except Exception:
    CFG_BUNDLE = (REPO_ROOT / "models" / "logreg_beats_pipeline_v5.joblib").resolve()

# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pileated gibbon pipeline")
    p.add_argument("--audio-dir",     type=str, default=str(CFG_RAW_AUDIO_DIR), help="Input audio directory")
    p.add_argument("--results-dir",   type=str, default=str(CFG_RESULTS_DIR),   help="Results directory")
    p.add_argument("--status-file",   type=str, default="",                     help="Optional status JSON path")
    p.add_argument("--log-file",      type=str, default="",                     help="(Accepted, ignored here)")
    p.add_argument("--beats-dir",     type=str, default=str(DEFAULT_BEATS_DIR), help="Path to BEATs source dir")
    p.add_argument("--beats-ckpt",    type=str, default=str(CFG_BEATS_CKPT),    help="Path to BEATs checkpoint .pt")
    p.add_argument("--model-bundle",  type=str, default=str(CFG_BUNDLE),        help="Classifier bundle (.joblib)")
    p.add_argument("--metadata-xlsx", type=str, default=str(REPO_ROOT / "metadata.xlsx"),
                   help="Optional metadata Excel to merge on 'recorder_id'")
    return p.parse_args()

args = parse_args()

RAW_AUDIO_DIR = Path(args.audio_dir).expanduser().resolve()
RESULTS_DIR   = Path(args.results_dir).expanduser().resolve()
STATUS_FILE   = Path(args.status_file).expanduser().resolve() if args.status_file else None
BEATS_DIR     = Path(args.beats_dir).expanduser().resolve()
BEATS_CKPT    = Path(args.beats_ckpt).expanduser().resolve()
BUNDLE_PATH   = Path(args.model_bundle).expanduser().resolve()
METADATA_XLSX = Path(args.metadata_xlsx).expanduser().resolve()

# Dirs
PROCESSED_DIR = RAW_AUDIO_DIR / "processed"
PRESENT_DIR   = PROCESSED_DIR / "present"
ABSENT_DIR    = PROCESSED_DIR / "absent"
for d in (PROCESSED_DIR, PRESENT_DIR, ABSENT_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# BEATs sources: prefer vendored paths in repo; fall back to arg
BEATS_DIR_CANDIDATES = [
    REPO_ROOT / "models" / "unilm" / "beats",  # canonical
    REPO_ROOT / "models" / "unlim" / "beats",  # tolerate earlier name
    BEATS_DIR,                                 # user-provided
]
for _cand in BEATS_DIR_CANDIDATES:
    if _cand.exists():
        BEATS_DIR = _cand.resolve()
        break
else:
    raise RuntimeError("BEATs sources not found. Expected at models/unilm/beats/ (or models/unlim/beats/).")

sys.path.append(str(BEATS_DIR))

# Ensure weights exist locally (downloads once from Release URLs)
ensure_asset(BEATS_CKPT, BEATS_URL)

# Load BEATs
beats_model = load_beats_model(BEATS_CKPT, BEATS_DIR, DEVICE)

# Load classifier bundle
with warnings.catch_warnings():
    warnings.simplefilter("ignore", InconsistentVersionWarning)
    bundle = joblib.load(BUNDLE_PATH)

classifier_pipeline = bundle["pipeline"]
DECISION_THRESHOLD  = float(bundle["threshold"])

# Constants
FILE_BATCH_SIZE      = 100   # max audio files per cycle
INFERENCE_BATCH_SIZE = 20    # segments per model batch
POLL_INTERVAL        = 10    # seconds between cycles

# Core per-file logic
def preprocess_and_segment(audio_file: Path) -> List[Tuple[int, float, float, np.ndarray, int]]:
    import librosa
    y, sr = librosa.load(str(audio_file), sr=None)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    y = adaptive_bandpass_filter(y, sr)
    y = noise_reduction(y, sr)
    y = normalize_amplitude(y)
    y = dynamic_range_compression(y)

    seg_samples = SEGMENT_SECONDS * sr
    n_segs = math.ceil(len(y) / seg_samples)
    segments: List[Tuple[int, float, float, np.ndarray, int]] = []
    for idx in range(n_segs):
        start = idx * seg_samples
        end = min(start + seg_samples, len(y))
        segments.append((idx, start / sr, end / sr, y[start:end], sr))
    return segments

def classify_proba(X: np.ndarray) -> np.ndarray:
    return classifier_pipeline.predict_proba(X)[:, 1]

def process_one_file(audio_file: Path) -> List[dict]:
    """
    Returns a list of segment-level records for this file.
    Moves the file to present/absent based on OR rule at τ.
    """
    try:
        processed_at = datetime.now().replace(microsecond=0).isoformat()
        stem = audio_file.stem
        recorder_id, date_time = stem.split("_", 1) if "_" in stem else (stem, "")

        segments = preprocess_and_segment(audio_file)
        seg_idxs, t0s, t1s, probs = [], [], [], []

        # Batch embed + classify
        for i in range(0, len(segments), INFERENCE_BATCH_SIZE):
            batch = segments[i:i + INFERENCE_BATCH_SIZE]
            idxs, bt0, bt1, chunks, srs = zip(*batch)

            emb_batch = []
            kept_idx, kept_t0, kept_t1 = [], [], []
            for idx, t0, t1, chunk, sr in zip(idxs, bt0, bt1, chunks, srs):
                try:
                    emb = extract_embedding_from_array(chunk, sr, TARGET_LENGTH, beats_model, DEVICE)
                    emb_batch.append(emb)
                    kept_idx.append(idx)
                    kept_t0.append(t0)
                    kept_t1.append(t1)
                except Exception as e:
                    print(f"[WARN] Embedding failed for segment in {audio_file}: {e}")

            if not emb_batch:
                continue

            X = np.vstack(emb_batch)
            try:
                p = classify_proba(X)
            except Exception as e:
                print(f"[WARN] Classification failed for {audio_file}: {e}")
                continue

            seg_idxs.extend(kept_idx)
            t0s.extend(kept_t0)
            t1s.extend(kept_t1)
            probs.extend(p.tolist())

        if not probs:
            print(f"[WARN] No usable segments for {audio_file}; leaving file in place.")
            return []

        # Sort by segment index
        ord_idx = np.argsort(seg_idxs)
        seg_idxs = np.array(seg_idxs)[ord_idx]
        t0s      = np.array(t0s)[ord_idx]
        t1s      = np.array(t1s)[ord_idx]
        probs    = np.array(probs)[ord_idx]

        pred = (probs >= DECISION_THRESHOLD).astype(int)
        any_positive = bool(pred.any())

        records: List[dict] = []
        for idx, t0, t1, pr, prob in zip(seg_idxs, t0s, t1s, pred, probs):
            records.append({
                "filename":       stem,
                "recorder_id":    recorder_id,
                "date_time":      date_time,
                "processed_at":   processed_at,
                "segment_idx":    int(idx),
                "start_time_s":   round(float(t0), 2),
                "end_time_s":     round(float(t1), 2),
                "embedding_file": "",
                "prediction":     int(pr),
                "probability":    float(prob),
                "audio_file":     str(audio_file)
            })

        # Move file based on OR rule
        dest = PRESENT_DIR if any_positive else ABSENT_DIR
        try:
            shutil.move(str(audio_file), dest / audio_file.name)
        except Exception as e:
            print(f"[WARN] Could not move {audio_file} to {dest}: {e}")

        return records

    except Exception as e:
        print(f"[WARN] Skipping file {audio_file} due to error: {e}")
        return []

# Post-processing
def merge_metadata() -> None:
    if not METADATA_XLSX.exists():
        print(f"Metadata missing at {METADATA_XLSX}, skipping.")
        return
    try:
        df_res  = pd.read_csv(RESULTS_DIR / "classification_results.csv")
        df_meta = pd.read_excel(METADATA_XLSX)
        df_merged = pd.merge(df_res, df_meta, on="recorder_id", how="left")
        out = RESULTS_DIR / "merged_classification_results.csv"
        df_merged.to_csv(out, index=False)
        print(f"Merged results: {out}")
    except Exception as e:
        print(f"[WARN] merge_metadata failed: {e}")

def list_input_files() -> List[Path]:
    try:
        files = [p for p in RAW_AUDIO_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
        files.sort()
        return files[:FILE_BATCH_SIZE]
    except Exception as e:
        print(f"[WARN] Could not list input files: {e}")
        return []

def append_results(all_records: List[dict]) -> None:
    idx_path = RESULTS_DIR / "classification_results.csv"
    cols = [
        "filename","recorder_id","date_time","processed_at","segment_idx",
        "start_time_s","end_time_s","embedding_file",
        "prediction","probability","audio_file"
    ]
    if not all_records:
        print("No new records this cycle.")
        return
    try:
        df_new = pd.DataFrame(all_records, columns=cols)
        if idx_path.exists():
            try:
                df_exist = pd.read_csv(idx_path)
                df_comb = pd.concat([df_exist, df_new], ignore_index=True)
            except Exception:
                df_comb = df_new
        else:
            df_comb = df_new
        df_comb.to_csv(idx_path, index=False)
        print(f"Master results written to {idx_path}")
    except Exception as e:
        print(f"[WARN] Failed to write master CSV {idx_path}: {e}")

# Cycle
def process_audio_files_cycle() -> None:
    files = list_input_files()
    num_files = len(files)
    print(f"Processing {num_files} files (max {FILE_BATCH_SIZE}) this cycle.")
    if num_files == 0:
        return

    started = datetime.now().replace(microsecond=0).isoformat()
    write_status(state="running", progress=0.0, done=0, total=num_files, started=started, message="Starting")

    all_records: List[dict] = []
    done = 0

    # Thread pool to avoid macOS spawn/pickling headaches
    max_workers = min(8, (os.cpu_count() or 4) * 2)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one_file, f): f for f in files}
        for fut in tqdm(as_completed(futures), total=num_files, desc="Files processed"):
            fpath = futures[fut]
            try:
                recs = fut.result() or []
                all_records.extend(recs)
            except Exception as e:
                print(f"[WARN] Worker failed on {fpath}: {e}")
            done += 1
            write_status(
                state="running",
                progress=done / max(1, num_files),
                done=done,
                total=num_files,
                current=str(fpath),
                started=started,
                message=f"Processed {done}/{num_files}"
            )

    append_results(all_records)
    merge_metadata()
    write_status(state="idle", progress=1.0, done=done, total=num_files, started=started, message="Cycle complete")

# Main
if __name__ == "__main__":
    print(f"[INFO] Using RAW_AUDIO_DIR={RAW_AUDIO_DIR}")
    print(f"[INFO] Writing results to {RESULTS_DIR}")
    print(f"[INFO] BEATs dir  : {BEATS_DIR}")
    print(f"[INFO] BEATs ckpt : {BEATS_CKPT}")
    print(f"[INFO] Bundle     : {BUNDLE_PATH}")
    if STATUS_FILE:
        print(f"[INFO] Status file: {STATUS_FILE}")

    while True:
        print("\nStarting processing cycle…")
        try:
            process_audio_files_cycle()
        except Exception as e:
            print(f"[WARN] Cycle failed unexpectedly but loop will continue: {e}")
        print(f"Cycle complete. Sleeping {POLL_INTERVAL}s…")
        try:
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("Interrupted by user.")
            break

# config.py (Python 3.9 compatible)
from pathlib import Path
from typing import Union
import torch
import librosa

# Repo-relative roots
SCRIPTS_DIR = Path(__file__).resolve().parent          # .../scripts
REPO_ROOT   = SCRIPTS_DIR.parent                       # repo root

# Default directories 
RAW_AUDIO_DIR  = (REPO_ROOT / "audio").resolve()
EMBEDDINGS_DIR = (REPO_ROOT / "embeddings").resolve()
RESULTS_DIR    = (REPO_ROOT / "results").resolve()
MODELS_DIR     = (REPO_ROOT / "models").resolve()

# Release asset URLs
BEATS_URL     = "https://github.com/AlastairPickering/pileated_gibbon_classifier/releases/download/v.1.0.0/BEATs_iter3_plus_AS2M.pt"

# Where to cache them locally
REPO_ROOT  = Path(__file__).resolve().parent.parent
MODELS_DIR = (REPO_ROOT / "models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BEATS_CKPT     = MODELS_DIR / "BEATs_iter3_plus_AS2M.pt"

# Ensure output folders exist
for p in (EMBEDDINGS_DIR, RESULTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Small helpers
def in_results(*parts: Union[str, Path]) -> Path:
    return (RESULTS_DIR.joinpath(*map(Path, parts))).resolve()

def in_models(*parts: Union[str, Path]) -> Path:
    return (MODELS_DIR.joinpath(*map(Path, parts))).resolve()

def in_audio(*parts: Union[str, Path]) -> Path:
    return (RAW_AUDIO_DIR.joinpath(*map(Path, parts))).resolve()

# Segment configuration
SEGMENT_SECONDS = 10  # fixed 10s chunks

# Detect sample rate from first .wav in RAW_AUDIO_DIR (fallback = 48 kHz)
try:
    wavs = [p for p in RAW_AUDIO_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
except Exception:
    wavs = []

if wavs:
    try:
        _sr = librosa.get_samplerate(str(wavs[0]))
    except Exception:
        _sr = 48_000
else:
    _sr = 48_000

TARGET_LENGTH = int(SEGMENT_SECONDS * _sr)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model bundle (pipeline + threshold)
MODEL_BUNDLE_PATH = in_models("logreg_beats_pipeline_v5.joblib")

# config.py
from pathlib import Path
import torch
import librosa

# DIRECTORIES
RAW_AUDIO_DIR  = Path("~/gdrive_mount/pileated_gibbon_production/audio").expanduser()
EMBEDDINGS_DIR = Path("~/gdrive_mount/pileated_gibbon_production/embeddings").expanduser()
RESULTS_DIR    = Path("~/gdrive_mount/pileated_gibbon_production/results").expanduser()

# Ensure output folders exist
for p in (EMBEDDINGS_DIR, RESULTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# SEGMENT CONFIGURATION
SEGMENT_SECONDS = 10  # fixed 10s chunks

# Detect sample rate from first .wav in RAW_AUDIO_DIR (fallback = 48 kHz)
_audio_files = [p for p in RAW_AUDIO_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".wav"]
if _audio_files:
    try:
        _sr = librosa.get_samplerate(str(_audio_files[0]))
    except Exception:
        _sr = 48_000
else:
    _sr = 48_000

TARGET_LENGTH = int(SEGMENT_SECONDS * _sr)

# DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL BUNDLE (pipeline + threshold)
# Both pipeline.py and classify.py should load threshold from this bundle.
MODEL_BUNDLE_PATH = Path(
    "~/gdrive_mount/pileated_gibbon_production/models/logreg_beats_pipeline_v5.joblib"
).expanduser()

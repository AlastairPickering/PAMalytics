# preprocessing.py
import librosa
import numpy as np
import scipy.signal as signal
import noisereduce as nr
import torch
from pathlib import Path

# Frequency band (Hz) for the band-pass filter
LOW_FREQ = 500
HIGH_FREQ = 2000


def adaptive_bandpass_filter(audio, sr, lowcut=LOW_FREQ, highcut=HIGH_FREQ, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, audio)


def noise_reduction(audio, sr):
    try:
        return nr.reduce_noise(y=audio, sr=sr)
    except AssertionError as e:
        # FFT window size invalid — skip denoising for this clip
        print(f"Warning: skipping noise_reduction due to: {e}")
        return audio


def normalize_amplitude(audio):
    return audio / (np.max(np.abs(audio)) + 1e-7)


def dynamic_range_compression(audio):
    return np.tanh(audio)


def pad_or_trim_waveform(waveform, target_length):
    """
    Pad (with zeros) or trim a 1D waveform to target_length samples.
    """
    current_length = waveform.shape[-1]
    if current_length < target_length:
        pad_length = target_length - current_length
        waveform = np.pad(waveform, (0, pad_length), mode="constant")
    elif current_length > target_length:
        waveform = waveform[:target_length]
    return waveform


def load_and_preprocess(file_path: Path, target_length: int):
    """
    Load audio from file_path, mono it, apply band-pass, noise reduction,
    normalization, gentle compression, then pad/trim to target_length.
    Returns (tensor[1, T], sample_rate).
    """
    # Load audio using librosa; returns mono waveform and sample rate
    waveform, sr = librosa.load(str(file_path), sr=None)
    if waveform.ndim > 1:  # stereo → mono
        waveform = np.mean(waveform, axis=0)

    waveform = adaptive_bandpass_filter(waveform, sr)
    waveform = noise_reduction(waveform, sr)
    waveform = normalize_amplitude(waveform)
    waveform = dynamic_range_compression(waveform)
    waveform = pad_or_trim_waveform(waveform, target_length)

    # Return tensor of shape (1, T) and sr
    return torch.tensor(waveform).float().unsqueeze(0), sr


# BEATs feature extraction

def load_beats_model(model_path: Path, beats_dir: Path, device):
    """
    Load BEATs model given a checkpoint and the BEATs source directory.
    """
    import sys
    sys.path.append(str(beats_dir))
    from BEATs import BEATs, BEATsConfig  # type: ignore

    checkpoint = torch.load(str(model_path), map_location=device)
    cfg = BEATsConfig(checkpoint["cfg"])
    beats_model = BEATs(cfg)
    beats_model.load_state_dict(checkpoint["model"])
    beats_model.eval()
    beats_model.to(device)
    return beats_model


def extract_embedding(file_path: Path, target_length: int, beats_model, device):
    """
    End-to-end: load & preprocess a file, run BEATs, mean-pool time to 1D embedding.
    """
    waveform, sr = load_and_preprocess(file_path, target_length)
    waveform = waveform.to(device)
    with torch.no_grad():
        features = beats_model.extract_features(waveform)[0]
    embedding = features.mean(dim=1).cpu().numpy().squeeze()
    return embedding


def extract_embedding_from_array(
    waveform: np.ndarray,
    sr: int,
    target_samples: int,
    beats_model,
    device,
):
    """
    Pad/trim an in-memory waveform to target_samples, then run BEATs and mean-pool
    to get a numpy embedding (1D).
    """
    wav = pad_or_trim_waveform(waveform, target_samples)
    tensor = torch.tensor(wav).float().unsqueeze(0).to(device)
    with torch.no_grad():
        feats = beats_model.extract_features(tensor)[0]
    return feats.mean(dim=1).cpu().numpy().squeeze()

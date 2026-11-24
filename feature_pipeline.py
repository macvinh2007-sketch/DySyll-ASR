"""One compatible feature pipeline for both training and inference.

16kHz mono; log-mel (80); 4 prosody dimensions:
relative log-F0, P(voiced), centered log-RMS, centered spectral-tilt proxy.
Spectral tilt is only a phonation PROXY; replace after phonetic validation.
"""
from __future__ import annotations
import numpy as np

SR = 16_000
HOP = 160      # 10 ms
NFFT = 400     # 25 ms
N_MELS = 80


def _match_length(arr, n, fill=0.):
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == n:
        return arr
    if not len(arr):
        return np.full(n, fill, dtype=np.float32)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arr)), arr).astype(np.float32)


def load_audio(path: str, trim_db: float = 28., sr: int = SR):
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) < sr * .12:
        raise ValueError("Audio shorter than 120 ms")
    y, _ = librosa.effects.trim(y, top_db=trim_db, frame_length=512, hop_length=HOP)
    if len(y) < sr * .12:
        raise ValueError("Audio too short after trimming")
    return np.asarray(y, dtype=np.float32)


def compute_features(y: np.ndarray):
    import librosa
    y = np.asarray(y, dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError("Non-finite waveform")
    # Same spectral settings MUST be used at train/eval/inference.
    mel_power = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=NFFT,
                       hop_length=HOP, win_length=NFFT, n_mels=N_MELS,
                       power=2.0, center=True)
    mel_db = librosa.power_to_db(mel_power, ref=1.0).T.astype(np.float32)
    mean = mel_db.mean(axis=0, keepdims=True)
    std = np.maximum(mel_db.std(axis=0, keepdims=True), 1.0)
    mel = np.clip((mel_db - mean) / std, -5., 5.).astype(np.float32)
    T = len(mel)

    try:
        f0, voiced, vprob = librosa.pyin(y, sr=SR, fmin=65, fmax=550,
                           frame_length=1024, hop_length=HOP, center=True)
        f0 = _match_length(np.nan_to_num(f0, nan=0.), T)
        vprob = _match_length(np.nan_to_num(vprob, nan=0.), T)
        voiced = _match_length(np.asarray(voiced, dtype=np.float32), T)
    except (ValueError, FloatingPointError):
        f0 = np.zeros(T, dtype=np.float32)
        vprob = np.zeros(T, dtype=np.float32)
        voiced = np.zeros(T, dtype=np.float32)
    valid = (f0 > 0) & (voiced > .5)
    logf = np.zeros(T, dtype=np.float32)
    if valid.any():
        base = np.median(np.log(np.maximum(f0[valid], 1)))
        logf[valid] = np.clip((np.log(f0[valid]) - base) / np.log(2), -1.5, 1.5)
    vprob = np.clip(vprob, 0.0, 1.0)
    rms = _match_length(librosa.feature.rms(y=y, frame_length=NFFT,
                          hop_length=HOP, center=True)[0], T)
    logrms = np.log(rms + 1e-6)
    logrms = np.clip((logrms - np.median(logrms)) / 3.0, -3.0, 3.0)

    # Log-spectral tilt proxy: slope of log magnitude vs log frequency (not CPP/HNR).
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP, win_length=NFFT, center=True))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
    inds = (freqs >= 200) & (freqs <= 4000)
    xx = np.log(freqs[inds] + 1.0)
    xx = (xx - xx.mean()) / max(xx.std(), 1e-6)
    yy = np.log(S[inds] + 1e-4)
    tilt = _match_length((xx[:, None] * yy).mean(axis=0), T)
    tilt = np.clip((tilt - np.median(tilt)) / 3.0, -3.0, 3.0)
    prosody = np.stack([logf, vprob, logrms, tilt], axis=1).astype(np.float32)
    if mel.shape[1] != 80 or prosody.shape != (T, 4):
        raise AssertionError((mel.shape, prosody.shape))
    if not np.isfinite(mel).all() or not np.isfinite(prosody).all():
        raise ValueError("Non-finite features")
    return mel, prosody


def load_compute(path: str, trim_db=28.):
    y=load_audio(path, trim_db=trim_db)
    mel, prosody = compute_features(y)
    return mel, prosody, len(y) / SR

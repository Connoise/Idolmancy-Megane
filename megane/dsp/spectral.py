"""Minimal spectral analysis: STFT and dB-scaled spectrograms (NumPy only).

Used by the GUI's spectrogram view in Phase 2; Phase 4 builds the full
image<->spectrogram resynthesis path on top of this.
"""
from __future__ import annotations

import numpy as np

from ..core import backend


def stft(x, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Short-time Fourier transform of a 1-D signal.

    Returns a complex array of shape ``(frames, n_fft // 2 + 1)``. The signal
    is zero-padded so even short inputs yield at least one frame. Runs on the
    CPU (analysis/visualization path, not the synthesis hot loop).
    """
    sig = backend.to_cpu(x).astype(np.float64).reshape(-1)
    if len(sig) < n_fft:
        sig = np.pad(sig, (0, n_fft - len(sig)))
    window = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(sig, n_fft)[::hop]
    return np.fft.rfft(frames * window, axis=1)


def magnitude_db(spec: np.ndarray, floor_db: float = -90.0) -> np.ndarray:
    """Magnitude of a complex spectrum in dBFS-ish scale, clipped at a floor."""
    mag = np.abs(spec)
    ref = mag.max()
    if ref <= 0:
        return np.full(mag.shape, floor_db)
    db = 20.0 * np.log10(np.maximum(mag / ref, 1e-12))
    return np.maximum(db, floor_db)


def spectrogram_db(
    x, sample_rate: float, n_fft: int = 1024, hop: int = 256, floor_db: float = -90.0
):
    """dB spectrogram plus axis info.

    Returns ``(mag_db, duration_s, max_freq_hz)`` where ``mag_db`` has shape
    ``(frames, bins)`` -- time along axis 0, frequency along axis 1.
    """
    spec = stft(x, n_fft=n_fft, hop=hop)
    mag = magnitude_db(spec, floor_db=floor_db)
    n = backend.to_cpu(x).reshape(-1).shape[0]
    duration = n / float(sample_rate) if sample_rate else 0.0
    return mag, duration, sample_rate / 2.0

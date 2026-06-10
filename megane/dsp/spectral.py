"""Spectral analysis & resynthesis: STFT, ISTFT, Griffin-Lim (NumPy).

The analysis half feeds the GUI's spectrogram view (Phase 2); the synthesis
half (ISTFT + Griffin-Lim phase estimation) is the Phase 4 image->audio
spectral path. This module is deliberately CPU/NumPy: FFT resynthesis is a
bake-style operation, and ``np.add.at`` overlap-add has no portable CuPy
equivalent. The GPU-accelerated spectral path is the additive oscillator bank
in the spectral *node*, which runs through the backend abstraction.
"""
from __future__ import annotations

import numpy as np

from ..core import backend


def stft(x, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Short-time Fourier transform of a 1-D signal.

    Returns a complex array of shape ``(frames, n_fft // 2 + 1)``. The signal
    is zero-padded so even short inputs yield at least one frame.
    """
    sig = backend.to_cpu(x).astype(np.float64).reshape(-1)
    if len(sig) < n_fft:
        sig = np.pad(sig, (0, n_fft - len(sig)))
    window = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(sig, n_fft)[::hop]
    return np.fft.rfft(frames * window, axis=1)


def istft(spec: np.ndarray, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Inverse STFT via windowed overlap-add (synthesis window = Hann).

    Inverse of :func:`stft` up to edge effects: the overlap-add is normalized
    by the summed squared window, the standard Griffin-Lim-compatible form.
    """
    frames = np.fft.irfft(np.asarray(spec), n=n_fft, axis=1)
    window = np.hanning(n_fft)
    frames = frames * window
    n_frames = frames.shape[0]
    out_len = n_fft + hop * (n_frames - 1)
    out = np.zeros(out_len)
    wsum = np.zeros(out_len)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    np.add.at(out, idx, frames)
    # values must be materialized to idx's shape: np.add.at with a broadcast
    # (stride-0) values array yields corrupted sums on some NumPy versions
    np.add.at(wsum, idx, np.repeat((window**2)[None, :], n_frames, axis=0))
    return out / np.maximum(wsum, 1e-8)


def griffin_lim(mag: np.ndarray, n_fft: int = 1024, hop: int = 256,
                iterations: int = 32, seed: int = 0) -> np.ndarray:
    """Estimate a signal whose STFT magnitude matches ``mag`` (frames, bins).

    Classic Griffin-Lim: start from seeded random phase, alternately enforce
    the target magnitude and STFT-consistency. Deterministic for a given seed,
    keeping renders reproducible. ``iterations=0`` returns the random-phase
    reconstruction (noisier, much faster).
    """
    mag = np.asarray(mag, dtype=np.float64)
    rng = np.random.default_rng(seed)
    angles = np.exp(2j * np.pi * rng.random(mag.shape))
    for _ in range(max(0, int(iterations))):
        x = istft(mag * angles, n_fft, hop)
        spec = stft(x, n_fft, hop)[: mag.shape[0]]
        angles[: spec.shape[0]] = np.exp(1j * np.angle(spec))
    return istft(mag * angles, n_fft, hop)


def resize_axis0(arr: np.ndarray, new_len: int) -> np.ndarray:
    """Linearly resample a 2-D array along axis 0 to ``new_len`` rows."""
    arr = np.asarray(arr, dtype=np.float64)
    old_len = arr.shape[0]
    if old_len == new_len:
        return arr
    if old_len == 1:
        return np.repeat(arr, new_len, axis=0)
    pos = np.linspace(0, old_len - 1, new_len)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, old_len - 1)
    frac = (pos - i0)[:, None]
    return arr[i0] * (1.0 - frac) + arr[i1] * frac


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

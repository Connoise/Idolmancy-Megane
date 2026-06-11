"""Audio export and (best-effort) preview playback.

WAV writing prefers ``soundfile`` (libsndfile) when present -- it supports
float and 24-bit subtypes. If it is unavailable we fall back to the Python
standard-library ``wave`` module (16-bit PCM), so a render never hard-fails for
lack of an optional dependency.

Playback uses ``sounddevice`` when available; on a headless machine it simply
reports that no audio device is present.
"""
from __future__ import annotations

import os
import wave

import numpy as np

from ..core import backend

try:  # optional, preferred
    import soundfile as _sf
except Exception:  # noqa: BLE001
    _sf = None

try:  # optional, only for live preview
    import sounddevice as _sd
except Exception:  # noqa: BLE001
    _sd = None


def _to_interleaved(data) -> np.ndarray:
    """Return a float32 array shaped (num_frames, num_channels) in [-1, 1]."""
    arr = backend.to_cpu(data)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = np.ascontiguousarray(arr.T, dtype=np.float32)  # (frames, channels)
    return np.clip(arr, -1.0, 1.0)


def write_wav(path: str, data, sample_rate: float, subtype: str = "PCM_16") -> str:
    """Write a mono/stereo buffer to ``path``. ``data`` is (C, N) or (N,).

    With ``soundfile`` installed the container format follows the file
    extension (``.wav``, ``.flac``, ...). FLAC does not support FLOAT; an
    incompatible subtype falls back to PCM_24 rather than failing the render.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames = _to_interleaved(data)
    sr = int(round(sample_rate))

    if _sf is not None:
        if path.lower().endswith(".flac") and subtype == "FLOAT":
            subtype = "PCM_24"
        _sf.write(path, frames, sr, subtype=subtype)
        return path

    if not path.lower().endswith(".wav"):
        raise RuntimeError(
            f"writing {os.path.splitext(path)[1]!r} requires the 'soundfile' "
            "package; only .wav is available via the stdlib fallback")
    # Fallback: 16-bit PCM via the standard library.
    pcm = (frames * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(frames.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def play(data, sample_rate: float) -> bool:
    """Preview a buffer. Returns False if no playback backend/device is available."""
    if _sd is None:
        return False
    try:  # pragma: no cover - depends on host audio hardware
        _sd.play(_to_interleaved(data), int(round(sample_rate)))
        _sd.wait()
        return True
    except Exception:  # noqa: BLE001
        return False

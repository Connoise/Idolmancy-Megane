"""Synthesis helpers: turn a sequence of values into an audio waveform.

The core technique is *sample-and-hold + continuous-phase oscillation*:

1. Each control value maps to a frequency and is held for ``step_samples``.
2. We integrate instantaneous frequency into phase with a running sum, so the
   oscillator phase is continuous across steps -- no clicks at boundaries.

Everything is vectorized through the active backend (NumPy or CuPy), so this
same code path will run on the GPU once CuPy is enabled.
"""
from __future__ import annotations

import numpy as np

from ..core import backend


def normalize(values, mode: str = "data_range"):
    """Normalize a value array to ``[0, 1]``.

    * ``data_range`` -- stretch the observed [min, max] across [0, 1]
      (best for *hearing* variation; relative).
    * ``unit``       -- assume values are already ~[0, 1] and just clip
      (faithful/absolute when upstream already normalized by dtype).
    """
    xp = backend.xp()
    v = xp.asarray(values, dtype=backend.float_dtype())
    if mode == "unit":
        return xp.clip(v, 0.0, 1.0)
    if mode == "data_range":
        v_min = v.min()
        v_max = v.max()
        span = v_max - v_min
        if float(backend.to_cpu(span)) <= 0.0:
            return xp.full_like(v, 0.5)
        return (v - v_min) / span
    raise ValueError(f"unknown normalize mode {mode!r}")


def sample_and_hold(freqs, step_samples: int):
    """Repeat each frequency ``step_samples`` times -> per-sample frequency."""
    xp = backend.xp()
    f = xp.asarray(freqs, dtype=backend.float_dtype())
    return xp.repeat(f, max(1, int(step_samples)))


def oscillate(freq_per_sample, sample_rate: float, amplitude: float = 0.8):
    """Render a continuous-phase sine from a per-sample frequency array."""
    xp = backend.xp()
    f = xp.asarray(freq_per_sample, dtype=backend.float_dtype())
    # Phase is the running integral of 2*pi*f/sr; cumulative sum keeps it
    # continuous so consecutive frequency steps never click.
    phase = xp.cumsum(2.0 * np.pi * f / float(sample_rate))
    return (amplitude * xp.sin(phase)).astype(backend.float_dtype())


def apply_fades(audio, sample_rate: float, fade_ms: float = 5.0):
    """Apply short linear fade in/out to remove start/stop transients."""
    xp = backend.xp()
    a = xp.asarray(audio, dtype=backend.float_dtype())
    n = a.shape[-1]
    fade = int(sample_rate * fade_ms / 1000.0)
    fade = min(fade, n // 2)
    if fade <= 0:
        return a
    ramp = xp.linspace(0.0, 1.0, fade, dtype=backend.float_dtype())
    a = a.copy()
    a[..., :fade] *= ramp
    a[..., -fade:] *= ramp[::-1]
    return a


def render_value_sequence(
    values,
    sample_rate: float,
    step_samples: int,
    *,
    amplitude: float = 0.8,
    fade_ms: float = 5.0,
):
    """Convenience: values(as frequencies) -> faded audio buffer (1-D)."""
    fps = sample_and_hold(values, step_samples)
    audio = oscillate(fps, sample_rate, amplitude)
    return apply_fades(audio, sample_rate, fade_ms)

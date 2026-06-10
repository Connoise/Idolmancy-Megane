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

    Statistics are computed in float64 regardless of the working precision so
    that fp16 mode cannot overflow on large raw values; only the *result* is
    cast to the working dtype.
    """
    xp = backend.xp()
    v = xp.asarray(values, dtype=xp.float64)
    if mode == "unit":
        return xp.clip(v, 0.0, 1.0).astype(backend.float_dtype())
    if mode == "data_range":
        v_min = v.min()
        v_max = v.max()
        span = v_max - v_min
        if float(backend.to_cpu(span)) <= 0.0:
            return xp.full(v.shape, 0.5, dtype=backend.float_dtype())
        return ((v - v_min) / span).astype(backend.float_dtype())
    raise ValueError(f"unknown normalize mode {mode!r}")


def sample_and_hold(freqs, step_samples: int):
    """Repeat each frequency ``step_samples`` times -> per-sample frequency."""
    xp = backend.xp()
    f = xp.asarray(freqs, dtype=backend.float_dtype())
    return xp.repeat(f, max(1, int(step_samples)))


def resample_nearest(values, n: int):
    """Resample a 1-D array to ``n`` samples by nearest-index lookup.

    The standard way Phase-3 nodes align a modulation input (amp, shape,
    velocity...) of arbitrary length to a target step/sample count.
    """
    xp = backend.xp()
    v = xp.asarray(values)
    if v.shape[-1] == n:
        return v
    if v.shape[-1] == 1:
        return xp.repeat(v, n)
    idx = xp.rint(xp.linspace(0, v.shape[-1] - 1, n)).astype(int)
    return v[idx]


def phase_accumulate(freq_per_sample, sample_rate: float):
    """Integrate per-sample frequency into wrapped phase (float64 radians).

    Accumulation stays in float64 no matter the working precision: an
    fp16/fp32 running sum loses pitch accuracy within a fraction of a second.
    """
    xp = backend.xp()
    f = xp.asarray(freq_per_sample)
    phase = xp.cumsum(2.0 * np.pi * f.astype(xp.float64) / float(sample_rate))
    return xp.mod(phase, 2.0 * np.pi)


def waveform_from_phase(phase, kind: str = "sine"):
    """Evaluate a (naive, unfiltered) waveform at the given phase (radians).

    Saw/square/triangle are intentionally naive -- aliasing and all. Faithful
    raw translation is the project's point; band-limited oscillators can come
    later as an option.
    """
    xp = backend.xp()
    if kind == "sine":
        return xp.sin(phase)
    p = xp.mod(phase / (2.0 * np.pi), 1.0)
    if kind == "saw":
        return 2.0 * p - 1.0
    if kind == "square":
        return xp.where(p < 0.5, 1.0, -1.0)
    if kind == "triangle":
        return 1.0 - 4.0 * xp.abs(p - 0.5)
    raise ValueError(f"unknown waveform {kind!r}")


def oscillate(freq_per_sample, sample_rate: float, amplitude: float = 0.8):
    """Render a continuous-phase sine from a per-sample frequency array."""
    phase = phase_accumulate(freq_per_sample, sample_rate)
    xp = backend.xp()
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

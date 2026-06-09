"""Pitch mapping: turn normalized values in ``[0, 1]`` into frequencies (Hz).

Implements the per-method pitch toggle from the spec:

* ``continuous`` -- map the value range onto a frequency range (log or linear).
  Faithful, no quantization.
* ``scale``      -- snap to the nearest degree of a musical scale/tuning.
* ``note_set``   -- pick from an explicit list of MIDI note numbers.

All functions are vectorized and accept either scalars or NumPy arrays.
"""
from __future__ import annotations

import numpy as np

A4_HZ = 440.0
A4_MIDI = 69

# Scale degrees as semitone offsets within one octave.
SCALES: dict[str, list[int]] = {
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],  # natural minor
    "pentatonic": [0, 2, 4, 7, 9],
    "whole_tone": [0, 2, 4, 6, 8, 10],
}


def note_to_freq(midi) -> Any:  # type: ignore[valid-type]
    """Convert MIDI note number(s) to frequency in Hz (12-TET, A4=440)."""
    midi = np.asarray(midi, dtype=np.float64)
    return A4_HZ * 2.0 ** ((midi - A4_MIDI) / 12.0)


def map_continuous(u, f_min: float, f_max: float, curve: str = "log"):
    """Map ``u in [0, 1]`` to ``[f_min, f_max]``.

    ``log`` spacing is perceptually even (equal ratios -> equal pitch steps) and
    is the default; ``linear`` maps the raw value range directly onto Hz.
    """
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    if curve == "linear":
        return f_min + u * (f_max - f_min)
    if curve == "log":
        f_min = max(f_min, 1e-6)
        return f_min * (f_max / f_min) ** u
    raise ValueError(f"unknown curve {curve!r} (use 'log' or 'linear')")


def scale_midi_notes(root_midi: int, scale_name: str, octaves: int) -> np.ndarray:
    """Build the ascending MIDI notes of a scale spanning ``octaves`` octaves."""
    if scale_name not in SCALES:
        raise ValueError(f"unknown scale {scale_name!r}; choose from {list(SCALES)}")
    offsets = SCALES[scale_name]
    notes = [root_midi + 12 * o + s for o in range(max(1, octaves)) for s in offsets]
    return np.asarray(notes, dtype=np.float64)


def map_scale(u, root_midi: int = 48, scale_name: str = "major", octaves: int = 4):
    """Map ``u in [0, 1]`` to a frequency on the chosen scale (nearest degree)."""
    notes = scale_midi_notes(root_midi, scale_name, octaves)
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    idx = np.rint(u * (len(notes) - 1)).astype(int)
    return note_to_freq(notes[idx])


def map_note_set(u, notes: list[int]):
    """Map ``u in [0, 1]`` to a frequency chosen from an explicit MIDI note set."""
    if not notes:
        raise ValueError("note_set mapping requires at least one note")
    arr = np.asarray(notes, dtype=np.float64)
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)
    idx = np.rint(u * (len(arr) - 1)).astype(int)
    return note_to_freq(arr[idx])


def map_values(
    u,
    mode: str,
    *,
    f_min: float = 110.0,
    f_max: float = 1760.0,
    curve: str = "log",
    root_midi: int = 48,
    scale_name: str = "major",
    octaves: int = 4,
    notes: list[int] | None = None,
):
    """Dispatch to the requested pitch mode and return frequencies in Hz."""
    if mode == "continuous":
        return map_continuous(u, f_min, f_max, curve)
    if mode == "scale":
        return map_scale(u, root_midi, scale_name, octaves)
    if mode == "note_set":
        return map_note_set(u, notes or [60])
    raise ValueError(f"unknown pitch mode {mode!r}")

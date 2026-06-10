"""Core data types that flow between nodes.

These mirror the TouchDesigner mental model the project is built around:

======== ============ ==================================================
Megane    TD analogy   Carries
======== ============ ==================================================
Image     TOP          2-D multi-channel raster
Channel   CHOP         1-D numeric stream(s) + sample rate -- the
                       universal "math data": control values, envelopes,
                       BPM, *and* audio (audio == a Channel at audio rate)
MIDIData  (events)     discrete note events with timing
Table     DAT          structured statistics / metadata
======== ============ ==================================================

Unifying audio and control data as one ``Channel`` type is what makes
node->node modulation and sample-rate handling fall out naturally: audio is
just a high-rate channel, a BPM input is just a 1-sample (constant) channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import backend


# Port data-type identifiers used for wiring/validation.
IMAGE = "image"
CHANNEL = "channel"
MIDI = "midi"
TABLE = "table"
ANY = "any"


@dataclass
class Image:
    """A 2-D raster. ``data`` is ``(H, W)`` or ``(H, W, C)``.

    Pixel values are floating point. When loaded with normalization they are in
    ``[0, 1]``; otherwise they retain the source scale. ``color_space`` is
    informational for now (defaults to sRGB, as decoded by Pillow).
    """

    data: Any
    color_space: str = "sRGB"
    source_path: str | None = None

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def channels(self) -> int:
        return int(self.data.shape[2]) if self.data.ndim == 3 else 1

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Image({self.height}x{self.width}x{self.channels}, {self.color_space})"


def _as_2d(arr) -> Any:
    """Normalize stream data to shape ``(num_streams, num_samples)``."""
    a = backend.xp().asarray(arr)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim != 2:
        raise ValueError(f"Channel data must be 1-D or 2-D, got shape {a.shape}")
    return a


@dataclass
class Channel:
    """One or more numeric streams sharing a sample rate.

    ``data`` is stored as ``(num_streams, num_samples)``. A mono audio buffer is
    ``(1, N)``; stereo is ``(2, N)``. A constant control value is ``(1, 1)``.

    The ``sample_rate`` of a control channel is informational -- consumer nodes
    (e.g. an oscillator) decide final timing. For audio channels it is the
    literal sample rate used on export/playback.
    """

    data: Any
    sample_rate: float = 48000.0

    def __post_init__(self) -> None:
        self.data = _as_2d(self.data)

    @classmethod
    def mono(cls, samples, sample_rate: float = 48000.0) -> "Channel":
        return cls(_as_2d(samples), sample_rate)

    @classmethod
    def constant(cls, value: float) -> "Channel":
        return cls(np.full((1, 1), float(value)), sample_rate=1.0)

    @property
    def num_streams(self) -> int:
        return int(self.data.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        return self.num_samples / float(self.sample_rate) if self.sample_rate else 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Channel(streams={self.num_streams}, samples={self.num_samples}, "
            f"sr={self.sample_rate:g})"
        )


@dataclass
class MIDINote:
    """A single note event. Times are in seconds."""

    pitch: int  # MIDI note number 0..127
    velocity: int = 96  # 0..127
    start: float = 0.0
    duration: float = 0.5
    track: int = 0


@dataclass(repr=False)
class MIDIData:
    """A collection of note events (produced by translator nodes like to_midi)."""

    notes: list[MIDINote] = field(default_factory=list)
    tempo_bpm: float = 120.0

    @property
    def duration(self) -> float:
        return max((n.start + n.duration for n in self.notes), default=0.0)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"MIDIData(notes={len(self.notes)}, tempo={self.tempo_bpm:g} bpm, "
                f"duration={self.duration:.2f}s)")


@dataclass
class Table:
    """Structured statistics / metadata. A thin, flexible dict wrapper."""

    rows: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.rows[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.rows[key] = value

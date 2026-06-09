"""Oscillator node: Channel (control values) -> Channel (mono audio).

This is the Phase 1 translator + synth. It normalizes the incoming value
sequence, maps each value to a frequency via the per-method pitch toggle
(continuous / scale / note_set), then renders a continuous-phase tone using
sample-and-hold timing.

Timing is set either by a fixed total duration or by BPM (one step per beat
subdivision), satisfying the spec's "spatial data -> time via a BPM input".
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.node import Node, Param, Port, register
from ..core import types
from ..core.types import Channel
from ..dsp import pitch, synth


@register
class OscillatorNode(Node):
    type_name = "oscillator"
    inputs = [Port("values", types.CHANNEL)]
    outputs = [Port("audio", types.CHANNEL)]
    params = [
        # value -> [0,1] normalization
        Param("input_range", "data_range", choices=["data_range", "unit"],
              help="'data_range' stretches observed min/max; 'unit' assumes ~[0,1]."),
        # pitch mapping
        Param("pitch_mode", "continuous", choices=["continuous", "scale", "note_set"],
              help="Per-method pitch toggle."),
        Param("f_min", 110.0, help="continuous: low frequency (Hz)."),
        Param("f_max", 1760.0, help="continuous: high frequency (Hz)."),
        Param("curve", "log", choices=["log", "linear"],
              help="continuous: frequency spacing."),
        Param("root_midi", 48, help="scale: root MIDI note."),
        Param("scale", "major", choices=list(pitch.SCALES.keys()),
              help="scale: scale name."),
        Param("octaves", 4, help="scale: number of octaves spanned."),
        Param("notes", [60, 62, 64, 67, 69], help="note_set: explicit MIDI notes."),
        # timing
        Param("duration_mode", "total_seconds", choices=["total_seconds", "bpm"],
              help="How step length is derived."),
        Param("total_seconds", 4.0, help="total_seconds mode: output length."),
        Param("bpm", 120.0, help="bpm mode: beats per minute."),
        Param("steps_per_beat", 1.0, help="bpm mode: subdivisions per beat."),
        # output
        Param("amplitude", 0.8, help="Peak amplitude (0..1)."),
        Param("sample_rate", 48000.0, help="Output audio sample rate (Hz)."),
    ]

    def _frequencies(self, u) -> np.ndarray:
        return pitch.map_values(
            u,
            self.values["pitch_mode"],
            f_min=float(self.values["f_min"]),
            f_max=float(self.values["f_max"]),
            curve=self.values["curve"],
            root_midi=int(self.values["root_midi"]),
            scale_name=self.values["scale"],
            octaves=int(self.values["octaves"]),
            notes=list(self.values["notes"]),
        )

    def _step_samples(self, n_values: int, sr: float) -> int:
        if self.values["duration_mode"] == "bpm":
            beat = 60.0 / float(self.values["bpm"])
            step_dur = beat / float(self.values["steps_per_beat"])
        else:
            step_dur = float(self.values["total_seconds"]) / max(1, n_values)
        return max(1, int(round(step_dur * sr)))

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ch: Channel = inputs["values"]
        values = ch.data[0]  # first stream
        sr = float(self.values["sample_rate"])

        u = synth.normalize(values, self.values["input_range"])
        freqs = self._frequencies(u)
        step_samples = self._step_samples(len(np.atleast_1d(freqs)), sr)
        audio = synth.render_value_sequence(
            freqs, sr, step_samples, amplitude=float(self.values["amplitude"])
        )
        return {"audio": Channel.mono(audio, sample_rate=sr)}

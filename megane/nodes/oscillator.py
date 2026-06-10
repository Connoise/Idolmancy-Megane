"""Oscillator node: Channel (control values) -> Channel (mono audio).

The core translator + synth. It normalizes the incoming value sequence, maps
each value to a frequency via the per-method pitch toggle (continuous / scale /
note_set), and renders a continuous-phase tone with sample-and-hold timing.

Phase 3 adds waveform selection and three optional modulation inputs, which is
what makes the color pipeline expressive: drive ``values`` from hue, ``amp``
from brightness (dynamics), and ``shape`` from saturation (timbre). ``bpm`` can
likewise be a constant param or fed from another node.

* ``amp``   -- per-sample amplitude multiplier (resampled to the output).
* ``shape`` -- per-sample morph in [0,1] from a pure sine to ``waveform``.
* ``bpm``   -- overrides the bpm param in bpm timing mode.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import backend, types
from ..core.node import Node, Param, Port, register
from ..core.types import Channel
from ..dsp import pitch, synth


@register
class OscillatorNode(Node):
    type_name = "oscillator"
    inputs = [Port("values", types.CHANNEL), Port("amp", types.CHANNEL),
              Port("shape", types.CHANNEL), Port("bpm", types.CHANNEL)]
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
        # tone
        Param("waveform", "sine", choices=["sine", "saw", "square", "triangle"],
              help="Base waveform (shape input morphs sine->this)."),
        # timing
        Param("duration_mode", "total_seconds", choices=["total_seconds", "bpm"],
              help="How step length is derived."),
        Param("total_seconds", 4.0, help="total_seconds mode: output length."),
        Param("bpm", 120.0, help="bpm mode: beats per minute (bpm input overrides)."),
        Param("steps_per_beat", 1.0, help="bpm mode: subdivisions per beat."),
        # output
        Param("amplitude", 0.8, help="Peak amplitude (0..1; amp input multiplies)."),
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

    def _step_samples(self, n_values: int, sr: float, bpm: float) -> int:
        if self.values["duration_mode"] == "bpm":
            step_dur = (60.0 / bpm) / float(self.values["steps_per_beat"])
        else:
            step_dur = float(self.values["total_seconds"]) / max(1, n_values)
        return max(1, int(round(step_dur * sr)))

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ch: Channel = inputs["values"]
        xp = backend.xp()
        sr = float(self.values["sample_rate"])

        # pitch mapping runs on CPU (tiny, per-step); synthesis runs on backend
        u = backend.to_cpu(synth.normalize(ch.data[0], self.values["input_range"]))
        freqs = np.atleast_1d(self._frequencies(u))

        bpm = float(self.values["bpm"])
        if "bpm" in inputs:
            bpm = abs(float(backend.to_cpu(inputs["bpm"].data[0])[0])) or bpm
        step_samples = self._step_samples(len(freqs), sr, bpm)

        fps = synth.sample_and_hold(freqs, step_samples)
        n = int(fps.shape[-1])
        phase = synth.phase_accumulate(fps, sr)

        wave = synth.waveform_from_phase(phase, self.values["waveform"])
        if "shape" in inputs:
            s = xp.clip(synth.resample_nearest(inputs["shape"].data[0], n), 0.0, 1.0)
            wave = (1.0 - s) * synth.waveform_from_phase(phase, "sine") + s * wave

        audio = float(self.values["amplitude"]) * wave
        if "amp" in inputs:
            audio = audio * xp.clip(synth.resample_nearest(inputs["amp"].data[0], n),
                                    0.0, None)
        audio = synth.apply_fades(audio.astype(backend.float_dtype()), sr)
        return {"audio": Channel.mono(audio, sample_rate=sr)}

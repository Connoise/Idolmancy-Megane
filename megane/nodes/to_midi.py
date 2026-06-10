"""Channel -> MIDI translator: turn any value stream into note events.

This is the central translator of the spec's "function nodes output math data,
translator nodes convert it" design: anything that emits a Channel (raster
scan, color scan, expression...) becomes MIDI here.

Pitch quantization is inherent to MIDI, so the per-method toggle offers
``chromatic`` (nearest semitone in a note range), ``scale``, and ``note_set``.
``merge_repeats`` chooses faithful (every step retriggers) vs musical
(consecutive equal pitches merge into one held note).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import backend, types
from ..core.node import Node, Param, Port, register
from ..core.types import Channel, MIDIData, MIDINote
from ..dsp import pitch, synth


@register
class ToMidiNode(Node):
    type_name = "to_midi"
    inputs = [Port("values", types.CHANNEL), Port("velocity", types.CHANNEL),
              Port("bpm", types.CHANNEL)]
    outputs = [Port("midi", types.MIDI)]
    params = [
        Param("input_range", "data_range", choices=["data_range", "unit"],
              help="'data_range' stretches observed min/max; 'unit' assumes ~[0,1]."),
        Param("pitch_mode", "chromatic", choices=["chromatic", "scale", "note_set"],
              help="How values quantize to MIDI notes."),
        Param("low_note", 36, help="chromatic: lowest MIDI note."),
        Param("high_note", 84, help="chromatic: highest MIDI note."),
        Param("root_midi", 48, help="scale: root MIDI note."),
        Param("scale", "major", choices=list(pitch.SCALES.keys()),
              help="scale: scale name."),
        Param("octaves", 4, help="scale: octaves spanned."),
        Param("notes", [60, 62, 64, 67, 69], help="note_set: explicit MIDI notes."),
        Param("duration_mode", "bpm", choices=["bpm", "total_seconds"],
              help="Step timing source."),
        Param("bpm", 120.0, help="Tempo (bpm input overrides)."),
        Param("steps_per_beat", 1.0, help="bpm mode: subdivisions per beat."),
        Param("total_seconds", 4.0, help="total_seconds mode: full length."),
        Param("gate", 0.9, help="Note length as a fraction of the step (0..1]."),
        Param("velocity", 96, help="Constant velocity (velocity input overrides)."),
        Param("merge_repeats", False, choices=[True, False],
              help="Merge consecutive equal pitches into one held note."),
        Param("track", 0, help="Track number stamped on the notes."),
    ]

    def _pitches(self, u: np.ndarray) -> np.ndarray:
        mode = self.values["pitch_mode"]
        if mode == "chromatic":
            lo, hi = int(self.values["low_note"]), int(self.values["high_note"])
            lo, hi = min(lo, hi), max(lo, hi)
            return np.rint(lo + u * (hi - lo)).astype(int)
        if mode == "scale":
            notes = pitch.scale_midi_notes(int(self.values["root_midi"]),
                                           self.values["scale"],
                                           int(self.values["octaves"]))
        else:
            notes = np.asarray(self.values["notes"] or [60], dtype=np.float64)
        idx = np.rint(u * (len(notes) - 1)).astype(int)
        return notes[idx].astype(int)

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ch: Channel = inputs["values"]
        u = backend.to_cpu(synth.normalize(ch.data[0], self.values["input_range"]))
        u = np.atleast_1d(np.clip(u, 0.0, 1.0))
        n = len(u)

        bpm = float(self.values["bpm"])
        if "bpm" in inputs:
            bpm = abs(float(backend.to_cpu(inputs["bpm"].data[0])[0])) or bpm
        if self.values["duration_mode"] == "bpm":
            step = 60.0 / bpm / float(self.values["steps_per_beat"])
        else:
            step = float(self.values["total_seconds"]) / n

        if "velocity" in inputs:
            v01 = backend.to_cpu(synth.resample_nearest(inputs["velocity"].data[0], n))
            vels = np.rint(1 + np.clip(v01, 0.0, 1.0) * 126).astype(int)
        else:
            vels = np.full(n, int(self.values["velocity"]))

        pitches = self._pitches(u)
        gate = float(np.clip(self.values["gate"], 0.05, 1.0))
        track = int(self.values["track"])

        notes: list[MIDINote] = []
        i = 0
        while i < n:
            j = i + 1
            if self.values["merge_repeats"]:
                while j < n and pitches[j] == pitches[i]:
                    j += 1
            length = (j - i) * step
            notes.append(MIDINote(pitch=int(pitches[i]), velocity=int(vels[i]),
                                  start=i * step, duration=length * gate,
                                  track=track))
            i = j

        return {"midi": MIDIData(notes=notes, tempo_bpm=bpm)}

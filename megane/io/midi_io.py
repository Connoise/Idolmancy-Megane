"""MIDI file export via mido.

Notes carry absolute times in seconds; we convert to ticks using the
``MIDIData.tempo_bpm`` so a DAW importing the file at that tempo reproduces
the intended timing exactly. Notes are grouped into tracks by their ``track``
attribute (one mido track per distinct value -- the seam for the future
"image splitting -> multi-part composition" feature).
"""
from __future__ import annotations

import os

import mido

from ..core.types import MIDIData

TICKS_PER_BEAT = 480


def write_midi(midi: MIDIData, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bpm = float(midi.tempo_bpm) if midi.tempo_bpm > 0 else 120.0
    ticks_per_second = TICKS_PER_BEAT * bpm / 60.0

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track_ids = sorted({n.track for n in midi.notes}) or [0]

    for i, track_id in enumerate(track_ids):
        track = mido.MidiTrack()
        mid.tracks.append(track)
        if i == 0:
            track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))

        channel = track_id % 16
        events: list[tuple[int, int, mido.Message]] = []  # (tick, order, msg)
        for n in midi.notes:
            if n.track != track_id:
                continue
            pitch = int(max(0, min(127, n.pitch)))
            vel = int(max(1, min(127, n.velocity)))
            on_tick = int(round(n.start * ticks_per_second))
            off_tick = int(round((n.start + max(n.duration, 1e-3)) * ticks_per_second))
            # note_off sorts before note_on at the same tick so repeated
            # pitches re-trigger instead of cancelling the new note
            events.append((on_tick, 1, mido.Message("note_on", note=pitch,
                                                    velocity=vel, channel=channel)))
            events.append((off_tick, 0, mido.Message("note_off", note=pitch,
                                                     velocity=0, channel=channel)))
        events.sort(key=lambda e: (e[0], e[1]))

        now = 0
        for tick, _order, msg in events:
            msg.time = tick - now
            now = tick
            track.append(msg)

    mid.save(path)
    return path

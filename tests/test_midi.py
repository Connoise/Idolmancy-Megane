"""Tests for the Channel->MIDI translator and .mid export."""
import numpy as np
import mido
import pytest

from megane.core.types import Channel, MIDIData, MIDINote
from megane.io import midi_io
from megane.nodes.midi_output import MidiOutputNode
from megane.nodes.to_midi import ToMidiNode


def test_to_midi_note_count_and_range():
    ch = Channel.mono(np.linspace(0, 1, 12, dtype=np.float32), sample_rate=12.0)
    out = ToMidiNode(pitch_mode="chromatic", low_note=48, high_note=72,
                     duration_mode="bpm", bpm=120.0).cook({"values": ch})
    midi = out["midi"]
    assert isinstance(midi, MIDIData)
    assert len(midi.notes) == 12
    assert all(48 <= n.pitch <= 72 for n in midi.notes)
    # ascending input -> ascending pitch, first beat at t=0, step=0.5s
    assert midi.notes[0].pitch == 48 and midi.notes[-1].pitch == 72
    assert midi.notes[0].start == pytest.approx(0.0)
    assert midi.notes[1].start == pytest.approx(0.5)


def test_to_midi_merge_repeats():
    vals = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    ch = Channel.mono(vals, sample_rate=4.0)
    merged = ToMidiNode(pitch_mode="note_set", notes=[60, 72],
                        merge_repeats=True).cook({"values": ch})["midi"]
    # three equal lows collapse to one held note + one high = 2 notes
    assert len(merged.notes) == 2
    assert merged.notes[0].duration == pytest.approx(3 * merged.notes[1].duration, rel=0.2)


def test_to_midi_velocity_input():
    ch = Channel.mono(np.linspace(0, 1, 5, dtype=np.float32), sample_rate=5.0)
    vel = Channel.mono(np.array([0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32), 5.0)
    midi = ToMidiNode().cook({"values": ch, "velocity": vel})["midi"]
    vels = [n.velocity for n in midi.notes]
    assert vels[0] < vels[-1] and max(vels) <= 127 and min(vels) >= 1


def test_to_midi_bpm_input_overrides():
    ch = Channel.mono(np.linspace(0, 1, 4, dtype=np.float32), sample_rate=4.0)
    bpm = Channel.constant(60.0)  # 1 beat = 1 s
    midi = ToMidiNode(duration_mode="bpm", steps_per_beat=1.0).cook(
        {"values": ch, "bpm": bpm})["midi"]
    assert midi.tempo_bpm == pytest.approx(60.0)
    assert midi.notes[1].start == pytest.approx(1.0)


def test_midi_file_roundtrip(tmp_path):
    notes = [MIDINote(pitch=60, velocity=100, start=0.0, duration=0.5),
             MIDINote(pitch=64, velocity=80, start=0.5, duration=0.5, track=1)]
    path = tmp_path / "x.mid"
    midi_io.write_midi(MIDIData(notes=notes, tempo_bpm=120.0), str(path))
    assert path.exists()

    mid = mido.MidiFile(str(path))
    assert mid.ticks_per_beat == midi_io.TICKS_PER_BEAT
    # two tracks (one per distinct track id)
    assert len(mid.tracks) == 2
    note_ons = [m for tr in mid.tracks for m in tr if m.type == "note_on" and m.velocity > 0]
    assert {m.note for m in note_ons} == {60, 64}


def test_midi_output_node_writes(tmp_path):
    ch = Channel.mono(np.linspace(0, 1, 6, dtype=np.float32), sample_rate=6.0)
    midi = ToMidiNode().cook({"values": ch})["midi"]
    out = MidiOutputNode(filename="t.mid", directory=str(tmp_path))
    result = out.cook({"midi": midi})
    assert result["midi"] is midi  # pass-through
    assert (tmp_path / "t.mid").exists()
    assert out.last_path == str(tmp_path / "t.mid")

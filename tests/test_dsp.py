"""Tests for pitch mapping and synthesis."""
import numpy as np
import pytest

from megane.dsp import pitch, synth


def test_a4_is_440():
    assert pitch.note_to_freq(69) == pytest.approx(440.0)
    assert pitch.note_to_freq(60) == pytest.approx(261.6256, rel=1e-4)  # middle C


def test_continuous_endpoints_and_monotonic():
    u = np.linspace(0, 1, 50)
    f = pitch.map_continuous(u, 100.0, 1000.0, "log")
    assert f[0] == pytest.approx(100.0)
    assert f[-1] == pytest.approx(1000.0)
    assert np.all(np.diff(f) > 0)  # strictly increasing


def test_scale_within_range():
    u = np.linspace(0, 1, 20)
    f = pitch.map_scale(u, root_midi=48, scale_name="major", octaves=4)
    notes = pitch.scale_midi_notes(48, "major", 4)
    assert f.min() >= pitch.note_to_freq(notes.min()) - 1e-6
    assert f.max() <= pitch.note_to_freq(notes.max()) + 1e-6


def test_normalize_modes():
    vals = np.array([10.0, 20.0, 30.0])
    dr = synth.normalize(vals, "data_range")
    assert float(dr.min()) == pytest.approx(0.0)
    assert float(dr.max()) == pytest.approx(1.0)
    # constant input must not divide by zero
    flat = synth.normalize(np.array([5.0, 5.0, 5.0]), "data_range")
    assert np.all(np.asarray(flat) == 0.5)


def test_oscillate_bounds_and_length():
    freqs = np.array([220.0, 440.0, 880.0])
    audio = synth.render_value_sequence(freqs, 48000.0, step_samples=4800, amplitude=0.8)
    assert audio.shape[-1] == 3 * 4800
    assert float(np.abs(np.asarray(audio)).max()) <= 0.8 + 1e-6
    assert not np.any(np.isnan(np.asarray(audio)))

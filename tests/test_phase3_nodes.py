"""Tests for wavetable, statistics, control/math nodes, and oscillator mods."""
import numpy as np
import pytest

from megane.core.graph import Graph
from megane.core.types import Channel, Image
from megane.nodes.control import (ConstantNode, ExpressionNode, RangeMapNode,
                                   ResampleNode)
from megane.nodes.oscillator import OscillatorNode
from megane.nodes.statistics import StatisticsNode
from megane.nodes.wavetable import WavetableNode


def _gradient(width=64, height=16):
    return Image(data=np.tile(np.linspace(0, 1, width, dtype=np.float32), (height, 1)))


# -- wavetable ----------------------------------------------------------------
def test_wavetable_row_output_bounds():
    img = _gradient()
    out = WavetableNode(source="row", frequency=220.0, duration=0.5,
                        sample_rate=22050.0).cook({"image": img})["audio"]
    a = np.asarray(out.data)
    assert out.num_samples == int(0.5 * 22050)
    assert np.all(np.isfinite(a)) and np.abs(a).max() <= 0.8 + 1e-6


def test_wavetable_histogram_source():
    img = _gradient()
    out = WavetableNode(source="histogram", bins=128, duration=0.25).cook({"image": img})
    assert np.all(np.isfinite(np.asarray(out["audio"].data)))


def test_wavetable_freq_and_amp_inputs():
    img = _gradient()
    n_hint = 24000
    freq = Channel.mono(np.full(n_hint, 440.0, dtype=np.float32), 48000.0)
    amp = Channel.mono(np.linspace(0, 1, 100, dtype=np.float32), 100.0)
    out = WavetableNode(duration=0.5, sample_rate=48000.0).cook(
        {"image": img, "freq": freq, "amp": amp})["audio"]
    a = np.asarray(out.data[0])
    # amp ramp from 0 -> start quiet, end louder
    assert np.abs(a[:200]).max() < np.abs(a[-200:]).max()


# -- statistics ---------------------------------------------------------------
def test_statistics_outputs():
    img = _gradient(width=100)
    outs = StatisticsNode(scale=240.0).cook({"image": img})
    assert set(outs) >= {"table", "mean", "std", "min", "max"}
    t = outs["table"]
    assert t["mean"] == pytest.approx(np.linspace(0, 1, 100).mean(), abs=1e-3)
    # scaled constant channel: mean*240 (a usable BPM)
    assert float(np.asarray(outs["mean"].data[0])[0]) == pytest.approx(t["mean"] * 240.0)


# -- control / math -----------------------------------------------------------
def test_constant_node():
    out = ConstantNode(value=120.0).cook({})["value"]
    assert out.num_samples == 1 and float(np.asarray(out.data[0])[0]) == 120.0


def test_expression_node_math():
    a = Channel.mono(np.array([0.0, 0.5, 1.0], dtype=np.float32), 3.0)
    b = Channel.mono(np.array([1.0, 1.0, 1.0], dtype=np.float32), 3.0)
    out = ExpressionNode(expr="a * 2 + b").cook({"a": a, "b": b})["out"]
    assert np.allclose(np.asarray(out.data[0]), [1.0, 2.0, 3.0])


def test_expression_node_resamples_mismatched_lengths():
    a = Channel.mono(np.linspace(0, 1, 10, dtype=np.float32), 10.0)
    b = Channel.mono(np.array([2.0], dtype=np.float32), 1.0)
    out = ExpressionNode(expr="a * b").cook({"a": a, "b": b})["out"]
    assert out.num_samples == 10


def test_expression_node_no_builtins():
    a = Channel.mono(np.array([1.0], dtype=np.float32), 1.0)
    with pytest.raises(Exception):
        ExpressionNode(expr="__import__('os').system('echo hi')").cook({"a": a})


def test_range_map_linear_and_geometric():
    ch = Channel.mono(np.array([0.0, 0.5, 1.0], dtype=np.float32), 3.0)
    lin = RangeMapNode(in_mode="fixed", in_min=0, in_max=1, out_min=100,
                       out_max=200).cook({"values": ch})["values"]
    assert np.allclose(np.asarray(lin.data[0]), [100, 150, 200])
    geo = RangeMapNode(in_mode="fixed", in_min=0, in_max=1, curve="geometric",
                       out_min=110, out_max=440).cook({"values": ch})["values"]
    assert np.asarray(geo.data[0])[1] == pytest.approx(220.0, rel=1e-4)  # octave midpoint


def test_resample_interpolate_changes_length():
    ch = Channel.mono(np.sin(np.linspace(0, 6, 100)).astype(np.float32), 100.0)
    out = ResampleNode(mode="interpolate", target_rate=50.0).cook({"values": ch})["values"]
    assert out.sample_rate == 50.0 and out.num_samples == 50


def test_resample_reinterpret_keeps_samples():
    ch = Channel.mono(np.arange(10, dtype=np.float32), 100.0)
    out = ResampleNode(mode="reinterpret", target_rate=200.0).cook({"values": ch})["values"]
    assert out.num_samples == 10 and out.sample_rate == 200.0


# -- oscillator modulation (Phase 3 additions) --------------------------------
def test_oscillator_waveform_and_backward_compat():
    ch = Channel.mono(np.linspace(0, 1, 16, dtype=np.float32), 16.0)
    saw = OscillatorNode(waveform="saw", total_seconds=0.5,
                         sample_rate=22050.0).cook({"values": ch})["audio"]
    assert np.all(np.isfinite(np.asarray(saw.data)))
    # default still sine, ~total_seconds long, amplitude-bounded
    sine = OscillatorNode(total_seconds=1.0, amplitude=0.5).cook({"values": ch})["audio"]
    assert abs(sine.num_samples - 48000) < 16 * 50
    assert np.abs(np.asarray(sine.data)).max() <= 0.5 + 1e-6


def test_oscillator_amp_modulation():
    ch = Channel.mono(np.linspace(0, 1, 8, dtype=np.float32), 8.0)
    amp = Channel.mono(np.linspace(0, 1, 8, dtype=np.float32), 8.0)
    out = OscillatorNode(total_seconds=1.0).cook({"values": ch, "amp": amp})["audio"]
    a = np.asarray(out.data[0])
    assert np.abs(a[:1000]).max() < np.abs(a[-1000:]).max()  # swells in


def test_oscillator_bpm_input(tmp_path):
    ch = Channel.mono(np.linspace(0, 1, 8, dtype=np.float32), 8.0)
    bpm = Channel.constant(60.0)
    out = OscillatorNode(duration_mode="bpm", steps_per_beat=1.0,
                         sample_rate=8000.0).cook({"values": ch, "bpm": bpm})["audio"]
    # 8 steps * 1 s/step at 8 kHz
    assert out.num_samples == 8 * 8000


# -- end-to-end color -> midi graph -------------------------------------------
def test_color_to_midi_graph(tmp_path):
    import megane  # noqa: F401 (register)
    from PIL import Image as PILImage
    from megane.nodes.color_scan import ColorScanNode  # noqa: F401
    from megane.nodes.image_input import ImageInputNode
    from megane.nodes.midi_output import MidiOutputNode
    from megane.nodes.to_midi import ToMidiNode

    # hue sweep image
    w = 64
    hue = np.linspace(0, 1, w)
    import colorsys
    rgb = np.array([[colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in hue]] * 8)
    p = tmp_path / "hue.png"
    PILImage.fromarray((rgb * 255).astype(np.uint8)).save(p)

    g = Graph()
    img = g.add(ImageInputNode(path=str(p)))
    cs = g.add(ColorScanNode(space="hsv"))
    tm = g.add(ToMidiNode(pitch_mode="chromatic", input_range="unit"))
    mo = g.add(MidiOutputNode(filename="hue.mid", directory=str(tmp_path)))
    g.connect(img.id, "image", cs.id, "image")
    g.connect(cs.id, "c1", tm.id, "values")
    g.connect(tm.id, "midi", mo.id, "midi")
    g.cook(mo.id)
    assert (tmp_path / "hue.mid").exists()

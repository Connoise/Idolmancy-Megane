"""Tests for the Phase 5 breadth nodes: stereo, harmonic, gradient,
metadata, split."""
import numpy as np
import pytest
from PIL import Image as PILImage

from megane.core.types import Channel, Image
from megane.nodes.gradient import GradientNode
from megane.nodes.harmonic import HarmonicNode
from megane.nodes.metadata import MetadataNode
from megane.nodes.split import SplitNode
from megane.nodes.stereo import MixNode, PanNode, StereoMergeNode, StereoWidthNode


def _tone(n=1000, sr=8000.0):
    t = np.arange(n) / sr
    return Channel.mono(np.sin(2 * np.pi * 440 * t).astype(np.float32), sr)


# -- stereo -------------------------------------------------------------------
def test_pan_hard_left_and_center():
    ch = _tone()
    left = PanNode(pan=-1.0).cook({"audio": ch})["audio"]
    assert left.num_streams == 2
    l, r = np.asarray(left.data)
    assert np.abs(r).max() < 1e-6 and np.abs(l).max() > 0.5
    center = PanNode(pan=0.0).cook({"audio": ch})["audio"]
    l, r = np.asarray(center.data)
    assert np.allclose(l, r)  # equal power center
    assert np.abs(l).max() == pytest.approx(np.sqrt(0.5), rel=1e-2)


def test_pan_input_channel():
    ch = _tone(n=2000)
    sweep = Channel.mono(np.linspace(-1, 1, 50, dtype=np.float32), 50.0)
    out = PanNode().cook({"audio": ch, "pan": sweep})["audio"]
    l, r = np.asarray(out.data)
    # starts left-heavy, ends right-heavy
    assert np.abs(l[:200]).max() > np.abs(r[:200]).max()
    assert np.abs(r[-200:]).max() > np.abs(l[-200:]).max()


def test_stereo_merge_and_width():
    a, b = _tone(), _tone()
    st = StereoMergeNode().cook({"left": a, "right": b})["audio"]
    assert st.num_streams == 2
    mono = StereoWidthNode(width=0.0).cook({"audio": st})["audio"]
    l, r = np.asarray(mono.data)
    assert np.allclose(l, r)  # width 0 collapses to mid


def test_mix_aligns_rates_and_lengths():
    a = Channel.mono(np.ones(100, dtype=np.float32), 100.0)   # 1 s
    b = Channel.mono(np.ones(100, dtype=np.float32) * 0.5, 200.0)  # 0.5 s
    out = MixNode(gain_a=1.0, gain_b=2.0).cook({"a": a, "b": b})["audio"]
    assert out.sample_rate == 200.0
    d = np.asarray(out.data[0])
    assert len(d) == 200
    assert d[0] == pytest.approx(2.0)   # 1*1 + 0.5*2 in the overlap
    assert d[-1] == pytest.approx(1.0)  # b ended; only a remains


# -- harmonic -------------------------------------------------------------------
def _single_row_image(h, w, row):
    arr = np.zeros((h, w), dtype=np.float32)
    arr[row, :] = 1.0
    return Image(data=arr)


def _peak_freq(audio, sr):
    spec = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64)))
    return float(np.argmax(spec)) * sr / len(audio)


def test_harmonic_fundamental_and_third():
    sr = 16000.0
    # bottom row (after flip) = fundamental
    img = _single_row_image(16, 8, row=15)
    audio = HarmonicNode(harmonics=16, f0=200.0, total_seconds=0.5,
                         sample_rate=sr).cook({"image": img})["audio"]
    assert abs(_peak_freq(audio.data[0], sr) - 200.0) < 8.0
    # row 13 of 16 from top -> band index 2 -> 3rd harmonic = 600 Hz
    img3 = _single_row_image(16, 8, row=13)
    audio3 = HarmonicNode(harmonics=16, f0=200.0, total_seconds=0.5,
                          sample_rate=sr).cook({"image": img3})["audio"]
    assert abs(_peak_freq(audio3.data[0], sr) - 600.0) < 8.0


def test_harmonic_f0_input_and_nyquist_guard():
    img = Image(data=np.ones((8, 8), dtype=np.float32))
    f0 = Channel.constant(3000.0)  # harmonics 2+ exceed Nyquist at 8 kHz
    audio = HarmonicNode(harmonics=8, total_seconds=0.2,
                         sample_rate=8000.0).cook({"image": img, "f0": f0})["audio"]
    assert np.all(np.isfinite(np.asarray(audio.data)))


# -- gradient -------------------------------------------------------------------
def test_gradient_directions():
    # horizontal luminance ramp: dx > 0, dy == 0, direction = 0 turns
    img = Image(data=np.tile(np.linspace(0, 1, 32, dtype=np.float32), (16, 1)))
    outs = GradientNode().cook({"image": img})
    assert set(outs) == {"dx", "dy", "magnitude", "direction"}
    assert float(np.asarray(outs["dx"].data[0]).mean()) > 0
    assert abs(float(np.asarray(outs["dy"].data[0]).mean())) < 1e-9
    inner = np.asarray(outs["direction"].data[0])[1:-1]  # edges use one-sided diff
    assert np.allclose(inner, 0.0, atol=1e-6) or np.allclose(inner, 1.0, atol=1e-6)
    assert float(np.asarray(outs["magnitude"].data[0]).mean()) > 0


# -- metadata -------------------------------------------------------------------
def test_metadata_table_and_numbers(tmp_path):
    p = tmp_path / "m.png"
    PILImage.new("RGB", (10, 6), (200, 10, 10)).save(p)
    from megane.io.image_io import load_image

    outs = MetadataNode().cook({"image": load_image(str(p))})
    table = outs["table"]
    assert table["width"] == 10 and table["height"] == 6
    assert table["format"] == "PNG" and table["file_size"] > 0
    nums = np.asarray(outs["numbers"].data[0])
    assert len(nums) >= 4 and np.all(np.isfinite(nums))


def test_metadata_without_file():
    img = Image(data=np.zeros((4, 4), dtype=np.float32))  # no source_path
    outs = MetadataNode().cook({"image": img})
    assert outs["numbers"].num_samples >= 3


# -- split ----------------------------------------------------------------------
def test_split_quadrants_and_channels():
    arr = np.zeros((8, 8, 3), dtype=np.float32)
    arr[..., 0] = 1.0  # pure red
    img = Image(data=arr)
    quads = SplitNode(mode="quadrants").cook({"image": img})
    assert all(quads[f"p{i}"].data.shape[:2] == (4, 4) for i in (1, 2, 3, 4))
    chans = SplitNode(mode="channels").cook({"image": img})
    assert float(np.asarray(chans["p1"].data).mean()) == pytest.approx(1.0)  # R
    assert float(np.asarray(chans["p2"].data).mean()) == pytest.approx(0.0)  # G
    assert float(np.asarray(chans["p4"].data).max()) == 0.0  # no alpha -> black


def test_split_to_multipart_graph():
    """The multi-part composition path: split -> 2 chains -> mix."""
    import megane  # noqa: F401
    from megane.core.graph import Graph
    from megane.nodes.oscillator import OscillatorNode
    from megane.nodes.raster_scan import RasterScanNode

    arr = np.random.default_rng(0).random((16, 32, 3)).astype(np.float32)
    g = Graph()
    from megane.core.node import get_node_class

    split = g.add(SplitNode(node_id="sp", mode="channels"))
    s1 = g.add(RasterScanNode(node_id="s1"))
    s2 = g.add(RasterScanNode(node_id="s2"))
    o1 = g.add(OscillatorNode(node_id="o1", total_seconds=0.3, sample_rate=8000.0))
    o2 = g.add(OscillatorNode(node_id="o2", total_seconds=0.5, sample_rate=8000.0))
    mx = g.add(MixNode(node_id="mx", normalize=True))

    class Src(get_node_class("image_input")):
        def cook(self, inputs):
            return {"image": Image(data=arr)}

    src = g.add(Src(node_id="src"))
    g.connect("src", "image", "sp", "image")
    g.connect("sp", "p1", "s1", "image")
    g.connect("sp", "p2", "s2", "image")
    g.connect("s1", "values", "o1", "values")
    g.connect("s2", "values", "o2", "values")
    g.connect("o1", "audio", "mx", "a")
    g.connect("o2", "audio", "mx", "b")
    out = g.cook("mx")["audio"]
    assert out.num_samples == 4000  # padded to the longer chain (0.5 s)
    assert float(np.abs(np.asarray(out.data)).max()) <= 1.0 + 1e-6

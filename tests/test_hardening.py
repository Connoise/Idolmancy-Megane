"""Edge-case and regression tests added while hardening Phase 1."""
import numpy as np
import pytest
from PIL import Image as PILImage

import megane  # noqa: F401
from megane.core import backend, project
from megane.core.graph import Graph
from megane.core.types import Channel
from megane.dsp import spectral
from megane.io.image_io import load_image
from megane.nodes.audio_output import AudioOutputNode
from megane.nodes.oscillator import OscillatorNode
from megane.nodes.raster_scan import RasterScanNode
from megane.nodes.raw_bytes import RawBytesNode


@pytest.fixture(autouse=True)
def _reset_precision():
    yield
    backend.set_precision("fp32")
    backend.set_gpu(False)


# -- precision lever ------------------------------------------------------
def test_fp16_render_is_sane():
    backend.set_precision("fp16")
    ch = Channel.mono(np.linspace(0, 1, 64, dtype=np.float32), sample_rate=64.0)
    osc = OscillatorNode(total_seconds=1.0, sample_rate=22050.0, amplitude=0.8)
    audio = np.asarray(osc.cook({"values": ch})["audio"].data, dtype=np.float64)
    assert np.all(np.isfinite(audio))
    assert np.abs(audio).max() <= 0.8 + 1e-2  # fp16 rounding headroom


def test_fp16_normalize_survives_large_values():
    backend.set_precision("fp16")
    from megane.dsp import synth

    out = np.asarray(synth.normalize(np.array([0.0, 30000.0, 65535.0])), dtype=np.float64)
    assert np.all(np.isfinite(out))
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0, abs=1e-3)


def test_phase_accuracy_long_render():
    # A constant 440 Hz held for 4 s must still be 440 Hz at the end:
    # zero crossings count ~= 2 * f * t.
    from megane.dsp import synth

    sr = 48000.0
    audio = np.asarray(synth.oscillate(np.full(int(4 * sr), 440.0), sr, 1.0))
    crossings = np.sum(np.abs(np.diff(np.signbit(audio[-int(sr) :]))))
    assert abs(crossings - 2 * 440) <= 4


# -- raw bytes ------------------------------------------------------------
def test_raw_bytes_odd_length_int16(tmp_path):
    p = tmp_path / "odd.bin"
    p.write_bytes(b"\x01" * 1001)  # not a multiple of 2
    out = RawBytesNode(path=str(p), dtype="int16", sample_rate=8000.0).cook({})
    assert out["audio"].num_samples == 500


# -- graph validation ------------------------------------------------------
def test_connect_rejects_type_mismatch():
    g = Graph()
    scan = g.add(RasterScanNode())
    out = g.add(AudioOutputNode())
    osc = g.add(OscillatorNode())
    with pytest.raises(TypeError, match="mismatch"):
        g.connect(out.id, "audio", scan.id, "image")  # CHANNEL -> IMAGE
    g.connect(scan.id, "values", osc.id, "values")  # CHANNEL -> CHANNEL ok


def test_disconnect_and_remove():
    g = Graph()
    scan = g.add(RasterScanNode())
    osc = g.add(OscillatorNode())
    g.connect(scan.id, "values", osc.id, "values")
    g.disconnect(scan.id, "values", osc.id, "values")
    assert g.connections == []
    g.connect(scan.id, "values", osc.id, "values")
    g.remove(scan.id)
    assert scan.id not in g.nodes and g.connections == []


# -- cached cook ------------------------------------------------------------
def test_cook_cached_reuses_and_invalidates():
    from megane.core import types as t
    from megane.core.node import Node, Port
    from megane.core.types import Image

    calls = {"src": 0, "scan": 0}

    class FakeInput(Node):
        type_name = "image_input"  # reuse a registered name; not re-registered
        outputs = [Port("image", t.IMAGE)]

        def cook(self, inputs):
            calls["src"] += 1
            return {"image": Image(data=np.tile(np.linspace(0, 1, 32, dtype=np.float32), (4, 1)))}

    class CountingScan(RasterScanNode):
        def cook(self, inputs):
            calls["scan"] += 1
            return super().cook(inputs)

    g = Graph()
    g.add(FakeInput(node_id="src"))
    g.add(CountingScan(node_id="scan"))
    g.connect("src", "image", "scan", "image")

    cache: dict = {}
    g.cook_cached("scan", cache)
    g.cook_cached("scan", cache)
    assert calls == {"src": 1, "scan": 1}  # second cook fully cached

    g.nodes["scan"].set("reduction", "max")  # param change invalidates scan only
    g.cook_cached("scan", cache)
    assert calls == {"src": 1, "scan": 2}  # upstream still cached


# -- image formats ----------------------------------------------------------
def test_palette_png_and_gif_first_frame(tmp_path):
    # palette PNG
    pal = PILImage.new("P", (8, 8))
    pal.putpalette([i for rgb in [(i, 0, 255 - i) for i in range(256)] for i in rgb])
    pal_path = tmp_path / "pal.png"
    pal.save(pal_path)
    img = load_image(str(pal_path))
    assert img.channels in (3, 4) and img.width == 8

    # 2-frame GIF -> first frame
    f0 = PILImage.new("L", (8, 8), 0)
    f1 = PILImage.new("L", (8, 8), 255)
    gif_path = tmp_path / "anim.gif"
    f0.save(gif_path, save_all=True, append_images=[f1], duration=100)
    img = load_image(str(gif_path))
    assert float(np.asarray(img.data).mean()) < 0.5  # frame 0 is the dark one


# -- project asset tracking ---------------------------------------------------
def test_check_assets_detects_change(tmp_path):
    img_path = tmp_path / "a.png"
    PILImage.new("L", (4, 4), 10).save(img_path)
    g = Graph()
    from megane.nodes.image_input import ImageInputNode

    g.add(ImageInputNode(node_id="i", path=str(img_path)))
    raw = project.to_dict(g)
    assert project.check_assets(raw) == []
    PILImage.new("L", (4, 4), 200).save(img_path)  # modify the file
    assert project.check_assets(raw) == [str(img_path)]


# -- oscillator timing ---------------------------------------------------------
def test_bpm_mode_duration():
    ch = Channel.mono(np.linspace(0, 1, 8, dtype=np.float32), sample_rate=8.0)
    osc = OscillatorNode(duration_mode="bpm", bpm=120.0, steps_per_beat=1.0,
                         sample_rate=48000.0)
    audio = osc.cook({"values": ch})["audio"]
    # 8 steps at 120 BPM = 8 * 0.5 s = 4 s
    assert audio.num_samples == 8 * 24000


# -- spectral -------------------------------------------------------------------
def test_spectrogram_peak_bin():
    sr, f = 8000.0, 440.0
    t = np.arange(int(sr)) / sr
    x = np.sin(2 * np.pi * f * t)
    mag, duration, fmax = spectral.spectrogram_db(x, sr, n_fft=1024, hop=512)
    assert duration == pytest.approx(1.0)
    peak_bin = int(np.argmax(mag.mean(axis=0)))
    peak_freq = peak_bin * sr / 1024
    assert abs(peak_freq - f) < sr / 1024 * 1.5

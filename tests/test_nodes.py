"""Tests for the Phase 1 nodes and the end-to-end graph."""
import numpy as np
import pytest
from PIL import Image as PILImage

import megane  # noqa: F401  (registers nodes)
from megane.core.graph import Graph
from megane.core.types import Channel, Image
from megane.nodes.audio_output import AudioOutputNode
from megane.nodes.image_input import ImageInputNode
from megane.nodes.oscillator import OscillatorNode
from megane.nodes.raster_scan import RasterScanNode
from megane.nodes.raw_bytes import RawBytesNode


def _gradient_image(width=64, height=16):
    x = np.linspace(0, 1, width, dtype=np.float32)
    return Image(data=np.tile(x, (height, 1)))


def test_raster_scan_column_length():
    img = _gradient_image(width=64, height=16)
    out = RasterScanNode(axis="column", channel="luminance").cook({"image": img})
    ch = out["values"]
    assert isinstance(ch, Channel)
    assert ch.num_samples == 64  # one value per column


def test_oscillator_duration_and_bounds():
    ch = Channel.mono(np.linspace(0, 1, 32, dtype=np.float32), sample_rate=32.0)
    osc = OscillatorNode(total_seconds=2.0, sample_rate=48000.0, amplitude=0.5)
    audio = osc.cook({"values": ch})["audio"]
    # ~2 s at 48 kHz (rounding per step is allowed)
    assert abs(audio.num_samples - 96000) < 32 * 50
    assert float(np.abs(np.asarray(audio.data)).max()) <= 0.5 + 1e-6


def test_raw_bytes(tmp_path):
    p = tmp_path / "blob.bin"
    p.write_bytes(bytes(range(256)) * 8)
    out = RawBytesNode(path=str(p), dtype="uint8", sample_rate=8000.0).cook({})
    audio = out["audio"]
    assert audio.num_samples == 256 * 8
    assert float(np.asarray(audio.data).min()) >= -1.0
    assert float(np.asarray(audio.data).max()) <= 1.0


def test_full_graph_renders_wav(tmp_path):
    img_path = tmp_path / "g.png"
    PILImage.fromarray((np.tile(np.linspace(0, 255, 80), (8, 1))).astype(np.uint8),
                       mode="L").save(img_path)

    g = Graph()
    img = g.add(ImageInputNode(path=str(img_path)))
    scan = g.add(RasterScanNode())
    osc = g.add(OscillatorNode(total_seconds=1.0))
    out = g.add(AudioOutputNode(filename="out.wav", directory=str(tmp_path)))
    g.connect(img.id, "image", scan.id, "image")
    g.connect(scan.id, "values", osc.id, "values")
    g.connect(osc.id, "audio", out.id, "audio")

    g.cook(out.id)
    written = tmp_path / "out.wav"
    assert written.exists() and written.stat().st_size > 1000


def test_cycle_detection():
    g = Graph()
    a = g.add(RasterScanNode())
    b = g.add(OscillatorNode())
    # Wire b.audio -> a.image is type-nonsense but exercises the cycle guard via ports.
    g.connect(a.id, "values", b.id, "values")
    # Manually craft a back-edge into a's input to form a cycle.
    g.connections.append((b.id, "audio", a.id, "image"))
    with pytest.raises(ValueError, match="cycle"):
        g.cook(b.id)

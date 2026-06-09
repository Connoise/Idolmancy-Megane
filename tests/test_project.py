"""Tests for project save/load and reproducibility."""
import numpy as np

from megane.core import project
from megane.core.graph import Graph
from megane.core.types import Channel
from megane.nodes.oscillator import OscillatorNode
from megane.nodes.raster_scan import RasterScanNode


def _build_graph():
    g = Graph()
    scan = g.add(RasterScanNode(node_id="scan1", axis="row", reduction="max"))
    osc = g.add(OscillatorNode(node_id="osc1", pitch_mode="scale", total_seconds=3.0))
    g.connect(scan.id, "values", osc.id, "values")
    return g


def test_save_load_roundtrip(tmp_path):
    g = _build_graph()
    path = tmp_path / "proj.megane"
    project.save(g, str(path))
    loaded = project.load(str(path))

    assert set(loaded.nodes) == set(g.nodes)
    assert loaded.connections == g.connections
    assert loaded.nodes["scan1"].values["reduction"] == "max"
    assert loaded.nodes["osc1"].values["pitch_mode"] == "scale"


def test_reproducible_output():
    # Identical settings -> bit-identical audio (deterministic CPU path).
    ch = Channel.mono(np.linspace(0, 1, 50, dtype=np.float32), sample_rate=50.0)
    osc = OscillatorNode(node_id="o", total_seconds=2.0, sample_rate=22050.0)
    a = np.asarray(osc.cook({"values": ch})["audio"].data)
    b = np.asarray(osc.cook({"values": ch})["audio"].data)
    assert np.array_equal(a, b)

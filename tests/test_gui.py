"""Headless GUI tests (offscreen Qt). Skipped entirely if the gui extra
is not installed, so the engine test suite stays runnable everywhere."""
import os

import numpy as np
import pytest
from PIL import Image as PILImage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_PREFERRED_BINDING", "PySide6")

pytest.importorskip("PySide6")
pytest.importorskip("NodeGraphQt")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets  # noqa: E402

import megane  # noqa: E402,F401
from megane.core.node import registered_nodes  # noqa: E402
from megane.core.types import Channel  # noqa: E402
from megane.gui.bridge import EngineBridge  # noqa: E402
from megane.gui.main_window import MainWindow  # noqa: E402
from megane.gui.param_panel import ParamPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture()
def demo_image(tmp_path):
    path = tmp_path / "demo.png"
    grad = (np.tile(np.linspace(0, 255, 64), (8, 1))).astype(np.uint8)
    PILImage.fromarray(grad, mode="L").save(path)
    return str(path)


def _wire(bridge, src, sport, dst, dport):
    """Wire two engine nodes through the *visual* ports (exercises signals)."""
    out_p = bridge.ng_node_for(src.id).outputs()[sport]
    in_p = bridge.ng_node_for(dst.id).inputs()[dport]
    out_p.connect_to(in_p)


def test_bridge_mirrors_user_actions(qapp, demo_image, tmp_path):
    bridge = EngineBridge()
    img = bridge.add_node("image_input")
    scan = bridge.add_node("raster_scan")
    osc = bridge.add_node("oscillator")
    out = bridge.add_node("audio_output")
    assert len(bridge.engine.nodes) == 4

    _wire(bridge, img, "image", scan, "image")
    _wire(bridge, scan, "values", osc, "values")
    _wire(bridge, osc, "audio", out, "audio")
    assert len(bridge.engine.connections) == 3

    bridge.set_param(img.id, "path", demo_image)
    bridge.set_param(osc.id, "total_seconds", 1.0)
    bridge.set_param(out.id, "directory", str(tmp_path))
    outputs = bridge.cook(out.id)
    audio = outputs["audio"]
    assert isinstance(audio, Channel) and audio.duration > 0.5
    assert (tmp_path / "output.wav").exists()


def test_bridge_rejects_bad_wire(qapp):
    bridge = EngineBridge()
    scan = bridge.add_node("raster_scan")
    out = bridge.add_node("audio_output")
    # CHANNEL -> IMAGE: engine must refuse, view wire must be undone.
    _wire(bridge, out, "audio", scan, "image")
    qapp.processEvents()  # let the deferred visual undo run
    assert bridge.engine.connections == []
    in_port = bridge.ng_node_for(scan.id).inputs()["image"]
    assert not in_port.connected_ports()


def test_bridge_save_load_roundtrip(qapp, demo_image, tmp_path):
    bridge = EngineBridge()
    img = bridge.add_node("image_input", pos=[10, 20])
    scan = bridge.add_node("raster_scan", pos=[300, 40])
    _wire(bridge, img, "image", scan, "image")
    bridge.set_param(img.id, "path", demo_image)
    bridge.set_param(scan.id, "reduction", "max")
    proj = tmp_path / "p.megane"
    bridge.save(str(proj))

    other = EngineBridge()
    other.load(str(proj))
    assert set(other.engine.nodes) == set(bridge.engine.nodes)
    assert other.engine.connections == bridge.engine.connections
    loaded_scan = other.engine.nodes[scan.id]
    assert loaded_scan.values["reduction"] == "max"
    # visual nodes restored and positioned
    ng_img = other.ng_node_for(img.id)
    assert ng_img is not None
    # cook the loaded session
    out = other.cook(scan.id)
    assert out["values"].num_samples == 64


def test_node_deletion_updates_engine(qapp):
    bridge = EngineBridge()
    scan = bridge.add_node("raster_scan")
    osc = bridge.add_node("oscillator")
    _wire(bridge, scan, "values", osc, "values")
    bridge.ng.delete_node(bridge.ng_node_for(scan.id))
    assert scan.id not in bridge.engine.nodes
    assert bridge.engine.connections == []


def test_param_panel_builds_and_edits(qapp):
    from megane.nodes.oscillator import OscillatorNode

    panel = ParamPanel()
    node = OscillatorNode()
    edits = {}
    panel.valueEdited.connect(lambda n, v: edits.__setitem__(n, v))
    panel.show_node(node)
    # title row + preset row + one row per param
    assert panel._form.rowCount() == 2 + len(node.params)

    # find the pitch_mode combo and switch it
    combos = panel.findChildren(QtWidgets.QComboBox)
    pitch_combo = next(c for c in combos
                       if {c.itemData(i) for i in range(c.count())}
                       >= {"continuous", "scale"})
    pitch_combo.setCurrentIndex(pitch_combo.findData("scale"))
    assert edits.get("pitch_mode") == "scale"


def test_main_window_smoke(qapp, demo_image, tmp_path):
    win = MainWindow()
    img = win.bridge.add_node("image_input")
    scan = win.bridge.add_node("raster_scan")
    osc = win.bridge.add_node("oscillator")
    _wire(win.bridge, img, "image", scan, "image")
    _wire(win.bridge, scan, "values", osc, "values")
    win.bridge.set_param(img.id, "path", demo_image)
    win.bridge.set_param(osc.id, "total_seconds", 0.5)

    # cook synchronously (same code the worker runs) and feed the preview
    outputs = win.bridge.cook(osc.id)
    win.preview.show_outputs(outputs, "oscillator")
    assert "duration" in win.preview.info.toPlainText()

    win.resize(1200, 800)
    pix = win.grab()
    assert pix.width() > 0 and not pix.isNull()
    win.close()


def test_all_node_types_have_ng_classes(qapp):
    bridge = EngineBridge()
    assert set(bridge._ng_classes) == set(registered_nodes())

"""Tests for Phase 6: presets, templates, render polish, export options."""
import json
import os

import numpy as np
import pytest
from PIL import Image as PILImage

import megane
from megane.core import presets, project
from megane.nodes.oscillator import OscillatorNode


# -- presets --------------------------------------------------------------
def test_builtin_presets_apply_cleanly():
    """Every built-in preset must reference real params with legal-ish values."""
    from megane.core.node import get_node_class

    for type_name, group in presets.BUILTIN.items():
        node = get_node_class(type_name)()
        for preset_name, values in group.items():
            for key, value in values.items():
                node.set(key, value)  # raises KeyError on a bad param name
        assert node.values  # node still consistent


def test_user_preset_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(tmp_path)))
    path = presets.save_user_preset("oscillator", "my sound",
                                    {"f_min": 200.0, "waveform": "square"})
    assert os.path.exists(path)
    merged = presets.list_presets("oscillator")
    assert merged["my sound"]["waveform"] == "square"
    assert "pentatonic steps" in merged  # built-ins still present


# -- templates ------------------------------------------------------------
@pytest.mark.parametrize("name", ["raster_melody", "color_field",
                                  "spectral_paint", "rgb_trio"])
def test_example_templates_load_and_cook(name, tmp_path):
    """Templates must load, and cook end-to-end once an image is supplied."""
    g = project.load(f"examples/{name}.megane")
    assert g.nodes and g.connections

    img_path = tmp_path / "t.png"
    arr = (np.random.default_rng(0).random((24, 48, 3)) * 255).astype(np.uint8)
    PILImage.fromarray(arr).save(img_path)
    for node in g.nodes_of_type("image_input"):
        node.set("path", str(img_path))
    # shrink renders for test speed
    for node in g.nodes.values():
        if "total_seconds" in node.values:
            node.set("total_seconds", 0.2)
        if "sample_rate" in node.values:
            node.set("sample_rate", 8000.0)
        if "max_partials" in node.values:
            node.set("max_partials", 16)
        if "directory" in node.values:
            node.set("directory", str(tmp_path))
    for sink in g.sink_nodes():
        g.cook(sink.id)
    written = list(tmp_path.glob("*.wav")) + list(tmp_path.glob("*.mid"))
    assert written  # every template produced output


def test_project_records_app_version(tmp_path):
    from megane.core.graph import Graph

    g = Graph()
    g.add(OscillatorNode(node_id="o"))
    project.save(g, str(tmp_path / "v.megane"))
    raw = project.read(str(tmp_path / "v.megane"))
    assert raw["app_version"] == megane.__version__


# -- render polish ------------------------------------------------------------
def test_render_resolves_paths_against_project_dir(tmp_path, monkeypatch):
    """A project with a *relative* image path renders from any cwd."""
    from megane.cli import main
    from megane.core.graph import Graph
    from megane.core.node import get_node_class

    arr = (np.tile(np.linspace(0, 255, 32), (8, 1))).astype(np.uint8)
    PILImage.fromarray(arr, mode="L").save(tmp_path / "img.png")

    g = Graph()
    n = lambda t, i, **p: g.add(get_node_class(t)(node_id=i, **p))
    n("image_input", "im", path="img.png")  # relative!
    n("raster_scan", "sc")
    n("oscillator", "os", total_seconds=0.2, sample_rate=8000.0)
    n("audio_output", "out", filename="rel.wav", directory="rendered")
    g.connect("im", "image", "sc", "image")
    g.connect("sc", "values", "os", "values")
    g.connect("os", "audio", "out", "audio")
    project.save(g, str(tmp_path / "rel.megane"))

    monkeypatch.chdir("/")  # somewhere unrelated
    rc = main(["render", str(tmp_path / "rel.megane")])
    assert rc == 0
    assert (tmp_path / "rendered" / "rel.wav").exists()


def test_render_writes_midi_sinks(tmp_path, monkeypatch):
    from megane.cli import main
    from megane.core.graph import Graph
    from megane.core.node import get_node_class

    PILImage.new("L", (16, 8), 128).save(tmp_path / "img.png")
    g = Graph()
    n = lambda t, i, **p: g.add(get_node_class(t)(node_id=i, **p))
    n("image_input", "im", path="img.png")
    n("raster_scan", "sc")
    n("to_midi", "tm")
    n("midi_output", "mo", filename="out.mid")
    g.connect("im", "image", "sc", "image")
    g.connect("sc", "values", "tm", "values")
    g.connect("tm", "midi", "mo", "midi")
    project.save(g, str(tmp_path / "m.megane"))

    monkeypatch.chdir(str(tmp_path))
    rc = main(["render", "m.megane", "--out-dir", "mout"])
    assert rc == 0
    assert (tmp_path / "mout" / "out.mid").exists()


# -- export options ------------------------------------------------------------
def test_flac_export(tmp_path):
    pytest.importorskip("soundfile")
    from megane.io import audio_io

    data = np.sin(np.linspace(0, 100, 8000)).astype(np.float32)
    path = audio_io.write_wav(str(tmp_path / "x.flac"), data, 8000.0,
                              subtype="FLOAT")  # FLOAT falls back to PCM_24
    assert os.path.exists(path)
    import soundfile as sf

    info = sf.info(path)
    assert info.format == "FLAC" and info.frames == 8000

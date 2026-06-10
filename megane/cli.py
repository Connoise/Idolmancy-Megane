"""Command-line interface for the Phase 1 engine.

Commands
--------
* ``info``        -- show backend/precision and the registered node types.
* ``quick-scan``  -- image -> raster scan -> oscillator -> WAV (one shot).
* ``raw``         -- reinterpret a file's raw bytes as audio -> WAV.
* ``demo``        -- generate a synthetic test image and run quick-scan.
* ``render``      -- load a saved ``.megane`` project and render its sinks.
* ``gui``         -- launch the node-graph interface (requires the gui extra).
* ``bench``       -- time the heavy nodes (compare CPU vs ``--gpu`` on CUDA).

These let you exercise the full vertical slice from the terminal and listen to
the result -- the Phase 1 feedback checkpoint.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import __version__
from .core import backend, project
from .core.graph import Graph
from .core.node import registered_nodes
from .nodes.audio_output import AudioOutputNode
from .nodes.image_input import ImageInputNode
from .nodes.oscillator import OscillatorNode
from .nodes.raster_scan import RasterScanNode
from .nodes.raw_bytes import RawBytesNode


def _apply_global(graph: Graph, args: argparse.Namespace) -> None:
    if getattr(args, "gpu", False):
        graph.settings["use_gpu"] = True
    if getattr(args, "precision", None):
        graph.settings["precision"] = args.precision


def _render_sink(graph: Graph, sink: AudioOutputNode, out_dir: str, play: bool) -> str:
    if not sink.values.get("directory"):
        sink.values["directory"] = out_dir
    outputs = graph.cook(sink.id)
    path = sink.last_path or os.path.join(out_dir, sink.values["filename"])
    print(f"  wrote {path}")
    if play:
        ok = sink.play(outputs["audio"])
        print("  played preview" if ok else "  (no audio device for preview)")
    return path


# -- commands ------------------------------------------------------------
def cmd_info(args: argparse.Namespace) -> int:
    print(f"Megane {__version__}")
    print(f"  backend     : {'GPU (CuPy)' if backend.gpu_available() else 'CPU (NumPy)'} available")
    print(f"  precision   : {backend.precision_name()}")
    print("  node types  :")
    for name in sorted(registered_nodes()):
        print(f"    - {name}")
    return 0


def cmd_quick_scan(args: argparse.Namespace) -> int:
    graph = Graph()
    _apply_global(graph, args)
    img = graph.add(ImageInputNode(path=args.image))
    scan = graph.add(RasterScanNode(axis=args.axis, channel=args.channel,
                                    reduction=args.reduction))
    osc = graph.add(OscillatorNode(
        pitch_mode=args.pitch_mode, f_min=args.f_min, f_max=args.f_max,
        scale=args.scale, total_seconds=args.duration, sample_rate=args.sr,
    ))
    out = graph.add(AudioOutputNode(filename=os.path.basename(args.out)))
    graph.connect(img.id, "image", scan.id, "image")
    graph.connect(scan.id, "values", osc.id, "values")
    graph.connect(osc.id, "audio", out.id, "audio")
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "output"
    _render_sink(graph, out, out_dir, args.play)
    if args.save:
        project.save(graph, args.save)
        print(f"  saved project {args.save}")
    return 0


def cmd_raw(args: argparse.Namespace) -> int:
    graph = Graph()
    _apply_global(graph, args)
    raw = graph.add(RawBytesNode(path=args.file, dtype=args.dtype,
                                 channels=args.channels, sample_rate=args.sr))
    out = graph.add(AudioOutputNode(filename=os.path.basename(args.out)))
    graph.connect(raw.id, "audio", out.id, "audio")
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "output"
    _render_sink(graph, out, out_dir, args.play)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from PIL import Image as PILImage

    os.makedirs(args.out_dir, exist_ok=True)
    img_path = os.path.join(args.out_dir, "demo_gradient.png")
    width, height = 512, 128
    x = np.linspace(0, 1, width)
    # A few sine sweeps stacked -> an audible melodic contour when scanned.
    contour = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * x) * np.cos(2 * np.pi * 0.5 * x)
    field = np.tile(contour, (height, 1))
    PILImage.fromarray((field * 255).astype(np.uint8), mode="L").save(img_path)
    print(f"Megane demo: generated {img_path}")

    args.image = img_path
    args.out = os.path.join(args.out_dir, "demo.wav")
    args.axis = "column"
    args.channel = "luminance"
    args.reduction = "mean"
    args.pitch_mode = "scale"
    args.scale = "pentatonic"
    args.f_min, args.f_max = 110.0, 1760.0
    args.duration = 5.0
    args.sr = 48000.0
    args.save = None
    return cmd_quick_scan(args)


def cmd_render(args: argparse.Namespace) -> int:
    raw = project.read(args.project)
    data = project.from_dict(raw)
    changed = project.check_assets(raw)
    if changed:
        print("  WARNING: referenced assets changed since save:")
        for p in changed:
            print(f"    ! {p}")
    out_dir = args.out_dir or data.settings.get("output_dir", "output")
    sinks = [n for n in data.nodes.values() if isinstance(n, AudioOutputNode)]
    if not sinks:
        print("  no audio_output nodes to render")
        return 1
    for sink in sinks:
        _render_sink(data, sink, out_dir, args.play)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Time the heavy paths on the active backend (run with/without --gpu)."""
    import time

    from .core.types import Image
    from .nodes.oscillator import OscillatorNode
    from .nodes.raster_scan import RasterScanNode
    from .nodes.spectral import SpectralNode
    from .nodes.wavetable import WavetableNode

    backend.set_precision(args.precision or "fp32")
    gpu_on = backend.set_gpu(args.gpu)
    if args.gpu and not gpu_on:
        print("(--gpu requested but CuPy/CUDA is unavailable; running on CPU)")
    label = "GPU (CuPy)" if gpu_on else "CPU (NumPy)"
    size = int(args.size)
    print(f"Megane bench — backend: {label}, precision: {backend.precision_name()}, "
          f"image: {size}x{size}")

    rng = np.random.default_rng(0)
    img = Image(data=rng.random((size, size)).astype(np.float32))

    def timed(name: str, fn) -> None:
        t0 = time.perf_counter()
        out = fn()
        # touch the result so lazy GPU kernels finish before the clock stops
        from .core import backend as _b
        for v in out.values():
            if hasattr(v, "data"):
                _b.to_cpu(v.data)
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"  {name:<46}: {dt:9.1f} ms")

    scan = RasterScanNode()
    osc = OscillatorNode(total_seconds=args.duration)
    timed("raster_scan + oscillator",
          lambda: osc.cook({"values": scan.cook({"image": img})["values"]}))
    timed(f"wavetable ({args.duration:g} s, linear interp)",
          lambda: WavetableNode(duration=args.duration).cook({"image": img}))
    timed(f"spectral additive ({args.partials} partials, {args.duration:g} s)",
          lambda: SpectralNode(method="additive", max_partials=args.partials,
                               total_seconds=args.duration).cook({"image": img}))
    timed(f"spectral istft (GL x{args.iterations})",
          lambda: SpectralNode(method="istft",
                               iterations=args.iterations).cook({"image": img}))
    if not gpu_on and backend.gpu_available():
        print("hint: add --gpu to compare against the CUDA backend")
    backend.set_gpu(False)
    backend.set_precision("fp32")
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui.app import main as gui_main
    except ImportError as exc:
        print("GUI dependencies missing. Install them with:")
        print('  pip install -e ".[gui]"      (PySide6, NodeGraphQt, pyqtgraph)')
        print(f"(import error: {exc})")
        return 1
    argv = ["megane-gui"] + ([args.project] if args.project else [])
    return gui_main(argv)


# -- parser --------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="megane", description="Image-to-audio sonification engine.")
    p.add_argument("--gpu", action="store_true", help="Use the CUDA backend if available.")
    p.add_argument("--precision", choices=["fp16", "fp32", "fp64"], help="Global float precision.")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Show engine info and node types.").set_defaults(func=cmd_info)

    qs = sub.add_parser("quick-scan", help="Image -> raster scan -> oscillator -> WAV.")
    qs.add_argument("image")
    qs.add_argument("-o", "--out", default="output/quick_scan.wav")
    qs.add_argument("--axis", default="column", choices=["column", "row"])
    qs.add_argument("--channel", default="luminance",
                    choices=["luminance", "red", "green", "blue", "alpha"])
    qs.add_argument("--reduction", default="mean",
                    choices=["mean", "max", "min", "median", "first"])
    qs.add_argument("--pitch-mode", dest="pitch_mode", default="continuous",
                    choices=["continuous", "scale", "note_set"])
    qs.add_argument("--scale", default="major")
    qs.add_argument("--f-min", dest="f_min", type=float, default=110.0)
    qs.add_argument("--f-max", dest="f_max", type=float, default=1760.0)
    qs.add_argument("--duration", type=float, default=4.0)
    qs.add_argument("--sr", type=float, default=48000.0)
    qs.add_argument("--save", help="Also save the graph as a .megane project.")
    qs.add_argument("--play", action="store_true", help="Preview after rendering.")
    qs.set_defaults(func=cmd_quick_scan)

    rw = sub.add_parser("raw", help="Reinterpret a file's raw bytes as audio.")
    rw.add_argument("file")
    rw.add_argument("-o", "--out", default="output/raw.wav")
    rw.add_argument("--dtype", default="uint8", choices=["uint8", "int8", "int16", "float32"])
    rw.add_argument("--channels", type=int, default=1, choices=[1, 2])
    rw.add_argument("--sr", type=float, default=8000.0)
    rw.add_argument("--play", action="store_true")
    rw.set_defaults(func=cmd_raw)

    dm = sub.add_parser("demo", help="Generate a test image and sonify it.")
    dm.add_argument("--out-dir", dest="out_dir", default="output")
    dm.add_argument("--play", action="store_true")
    dm.set_defaults(func=cmd_demo)

    rn = sub.add_parser("render", help="Render a saved .megane project.")
    rn.add_argument("project")
    rn.add_argument("--out-dir", dest="out_dir", default="")
    rn.add_argument("--play", action="store_true")
    rn.set_defaults(func=cmd_render)

    gu = sub.add_parser("gui", help="Launch the node-graph GUI.")
    gu.add_argument("project", nargs="?", default="",
                    help="Optional .megane project to open.")
    gu.set_defaults(func=cmd_gui)

    be = sub.add_parser("bench", help="Benchmark heavy nodes (CPU vs --gpu).")
    be.add_argument("--size", type=int, default=1024, help="Synthetic image side.")
    be.add_argument("--duration", type=float, default=4.0, help="Render seconds.")
    be.add_argument("--partials", type=int, default=512, help="Additive partial cap.")
    be.add_argument("--iterations", type=int, default=16, help="Griffin-Lim iterations.")
    be.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)

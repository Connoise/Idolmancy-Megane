"""Wavetable node: play the image itself as a waveform.

The spec's "waveform creation via the base and statistical values of the
image" tool. One cycle of the output waveform is read directly from the image:

* ``row`` / ``column`` -- each row (or column) is one waveform cycle; the
  table position can sweep through the image over time (``scan_speed``), so
  the timbre morphs as the playhead moves through the picture.
* ``histogram``        -- the luminance histogram becomes the single cycle:
  the image's *statistical* shape heard as a tone.

Pitch (``frequency`` param or per-sample ``freq`` input), speed
(``scan_speed``), dynamics (``amplitude`` + ``amp`` input), and the
``interpolation`` choice (nearest vs linear -- the "neighbor function" /
quality-vs-cost lever) are all exposed. Heavy math runs through the backend
abstraction, so this node is GPU-ready.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import backend, types
from ..core.node import Node, Param, Port, register
from ..core.types import Channel
from ..dsp import synth
from .raster_scan import _select_channel


@register
class WavetableNode(Node):
    type_name = "wavetable"
    inputs = [Port("image", types.IMAGE), Port("freq", types.CHANNEL),
              Port("amp", types.CHANNEL)]
    outputs = [Port("audio", types.CHANNEL)]
    params = [
        Param("source", "row", choices=["row", "column", "histogram"],
              help="What one waveform cycle is read from."),
        Param("channel", "luminance",
              choices=["luminance", "red", "green", "blue", "alpha"],
              help="Image channel the table is built from."),
        Param("position", 0.0,
              help="Start position in the table stack, 0..1 (row/column)."),
        Param("scan_speed", 0.0,
              help="Rows (or columns) swept per second; 0 holds position."),
        Param("bins", 256, help="histogram: number of bins (= cycle length)."),
        Param("frequency", 220.0, help="Base pitch in Hz (freq input overrides)."),
        Param("interpolation", "linear", choices=["linear", "nearest"],
              help="Table lookup quality (neighbor function)."),
        Param("duration", 4.0, help="Output length in seconds."),
        Param("amplitude", 0.8, help="Peak amplitude (amp input multiplies)."),
        Param("sample_rate", 48000.0, help="Output sample rate (Hz)."),
    ]

    def _build_table(self, img: types.Image) -> np.ndarray:
        plane = _select_channel(np.asarray(img.data), self.values["channel"])
        source = self.values["source"]
        if source == "histogram":
            hist, _ = np.histogram(plane, bins=max(8, int(self.values["bins"])),
                                   range=(0.0, 1.0))
            peak = hist.max()
            wave = hist / peak if peak > 0 else np.zeros_like(hist, dtype=np.float64)
            return wave[None, :]  # one cycle, nothing to scan through
        table = plane if source == "row" else plane.T
        return np.asarray(table, dtype=np.float64)

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        img: types.Image = inputs["image"]
        xp = backend.xp()
        sr = float(self.values["sample_rate"])
        n = max(1, int(round(float(self.values["duration"]) * sr)))

        table = backend.asarray(self._build_table(img))
        n_rows, cycle_len = int(table.shape[0]), int(table.shape[1])

        # per-sample frequency: input channel overrides the param
        if "freq" in inputs:
            fps = xp.abs(synth.resample_nearest(inputs["freq"].data[0], n))
        else:
            fps = xp.full(n, float(self.values["frequency"]))

        # in-cycle position from accumulated phase (float64, wrap-safe)
        phase = synth.phase_accumulate(fps, sr) / (2.0 * np.pi)  # cycles [0,1)
        x = phase * cycle_len

        # table-row position sweeps with time
        t = xp.arange(n, dtype=xp.float64) / sr
        row = float(self.values["position"]) * max(n_rows - 1, 0) \
            + t * float(self.values["scan_speed"])
        row = xp.clip(row, 0.0, max(n_rows - 1, 0))

        if self.values["interpolation"] == "nearest":
            ci = xp.rint(x).astype(int) % cycle_len
            ri = xp.rint(row).astype(int)
            wave = table[ri, ci]
        else:  # bilinear: across the cycle and across neighboring rows
            c0 = xp.floor(x).astype(int) % cycle_len
            c1 = (c0 + 1) % cycle_len
            fx = x - xp.floor(x)
            r0 = xp.floor(row).astype(int)
            r1 = xp.minimum(r0 + 1, max(n_rows - 1, 0))
            fr = row - r0
            top = (1.0 - fx) * table[r0, c0] + fx * table[r0, c1]
            bot = (1.0 - fx) * table[r1, c0] + fx * table[r1, c1]
            wave = (1.0 - fr) * top + fr * bot

        audio = (wave * 2.0 - 1.0) * float(self.values["amplitude"])
        if "amp" in inputs:
            audio = audio * synth.resample_nearest(inputs["amp"].data[0], n)
        audio = synth.apply_fades(audio.astype(backend.float_dtype()), sr)
        return {"audio": Channel.mono(audio, sample_rate=sr)}

"""Harmonic node: image values shape the harmonic series of a base frequency.

Unlike the spectral node (free-frequency partials), every component here is an
integer multiple of one fundamental, so the result is *pitched*: the image
sculpts the timbre of a single tone over time. Image rows are band-averaged
onto harmonics 1..K (bottom row = fundamental when ``flip_vertical``), and
each row's pixels drive that harmonic's amplitude across the duration.

The fundamental can be a constant or a per-sample ``f0`` Channel (vibrato,
glide -- harmonics track it exactly, staying phase-coherent because integer
multiples of a wrapped phase are still wrap-safe). Heavy-ish but bounded by
``harmonics``; runs through the backend abstraction (GPU-ready).
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
class HarmonicNode(Node):
    type_name = "harmonic"
    inputs = [Port("image", types.IMAGE), Port("f0", types.CHANNEL),
              Port("amp", types.CHANNEL)]
    outputs = [Port("audio", types.CHANNEL)]
    params = [
        Param("channel", "luminance",
              choices=["luminance", "red", "green", "blue", "alpha"],
              help="Image channel driving the harmonic amplitudes."),
        Param("flip_vertical", True, choices=[True, False],
              help="On: image bottom row = fundamental."),
        Param("harmonics", 16, help="Number of harmonics (rows band-average)."),
        Param("f0", 110.0, help="Base frequency in Hz (f0 input overrides)."),
        Param("gamma", 1.0, help="Brightness -> amplitude exponent."),
        Param("interpolation", "linear", choices=["linear", "nearest"],
              help="Column -> time amplitude interpolation."),
        Param("total_seconds", 4.0, help="Output length."),
        Param("normalize", True, choices=[True, False],
              help="Peak-normalize the harmonic sum to 'amplitude'."),
        Param("amplitude", 0.8, help="Peak amplitude."),
        Param("sample_rate", 48000.0, help="Output sample rate (Hz)."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        img: types.Image = inputs["image"]
        xp = backend.xp()
        sr = float(self.values["sample_rate"])
        n = max(1, int(round(float(self.values["total_seconds"]) * sr)))

        plane = _select_channel(np.asarray(img.data), self.values["channel"])
        plane = np.clip(np.asarray(plane, dtype=np.float64), 0.0, 1.0)
        if self.values["flip_vertical"]:
            plane = np.flipud(plane)  # row 0 = image bottom = fundamental
        plane = plane ** float(self.values["gamma"])

        k = max(1, int(self.values["harmonics"]))
        n_rows, n_cols = plane.shape
        if n_rows != k:  # band-average rows onto k harmonics
            edges = np.linspace(0, n_rows, k + 1).astype(int)
            edges = np.maximum(edges, np.arange(k + 1))  # ensure non-empty bands
            plane = np.stack([plane[a:b].mean(axis=0) if b > a else plane[min(a, n_rows - 1)]
                              for a, b in zip(edges[:-1], edges[1:])])

        # fundamental phase (wrapped; integer-harmonic-safe)
        if "f0" in inputs:
            f0 = xp.abs(synth.resample_nearest(inputs["f0"].data[0], n))
        else:
            f0 = xp.full(n, float(self.values["f0"]))
        base_phase = synth.phase_accumulate(f0, sr)

        # column -> per-sample amplitude lookup
        pos = xp.linspace(0.0, n_cols - 1.0, n)
        if self.values["interpolation"] == "nearest":
            c0 = xp.rint(pos).astype(int)
            c1, frac = c0, None
        else:
            c0 = xp.floor(pos).astype(int)
            c1 = xp.minimum(c0 + 1, n_cols - 1)
            frac = pos - c0

        amps = backend.asarray(plane)  # (k, cols)
        f0_cpu_max = float(backend.to_cpu(f0).max())
        audio = xp.zeros(n, dtype=xp.float64)
        for h in range(1, k + 1):
            if h * f0_cpu_max >= sr / 2:  # silence harmonics beyond Nyquist
                break
            a = amps[h - 1, c0]
            if frac is not None:
                a = a * (1.0 - frac) + amps[h - 1, c1] * frac
            audio += a * xp.sin(h * base_phase)

        peak = float(backend.to_cpu(xp.abs(audio).max()))
        if self.values["normalize"] and peak > 0:
            audio = audio / peak
        audio = audio * float(self.values["amplitude"])
        audio = synth.apply_fades(audio.astype(backend.float_dtype()), sr)
        return {"audio": Channel.mono(audio, sample_rate=sr)}

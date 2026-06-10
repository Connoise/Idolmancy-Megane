"""Spectral node: paint the image into a spectrogram and synthesize the audio.

The MetaSynth / ARSS lineage: x-axis is time, y-axis is frequency, pixel
brightness is energy. Two synthesis methods:

* ``additive``  -- one sine partial per image row (up to ``max_partials``),
  amplitude-driven by that row's pixels over time. Log or linear frequency
  axis; clean and musical for sparse images. Runs through the backend
  abstraction, so this is the GPU-accelerated path.
* ``istft``     -- image rows map onto linear FFT bins; phase is estimated by
  Griffin-Lim. Dense/noisy images keep their texture; frequency axis is
  inherently linear and the duration follows ``columns * hop / sample_rate``.

This is the project's first genuinely heavy node: ``realtime_capable`` is
False, so the GUI requires an explicit Cook (bake) instead of auto-cooking.

Faithfulness notes: rows beyond ``max_partials`` are averaged into bands
(documented downsampling, per spec); ``seed`` fixes the random phases so
identical settings render identical audio.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import backend, types
from ..core.node import Node, Param, Port, register
from ..core.types import Channel
from ..dsp import spectral, synth
from .raster_scan import _select_channel

_PARTIAL_CHUNK = 64  # partials synthesized per batch (bounds memory)


@register
class SpectralNode(Node):
    type_name = "spectral"
    realtime_capable = False  # bake-gated in the GUI
    inputs = [Port("image", types.IMAGE)]
    outputs = [Port("audio", types.CHANNEL)]
    params = [
        Param("method", "additive", choices=["additive", "istft"],
              help="Oscillator-bank resynthesis vs FFT + Griffin-Lim."),
        Param("channel", "luminance",
              choices=["luminance", "red", "green", "blue", "alpha"],
              help="Image channel used as spectral energy."),
        Param("flip_vertical", True, choices=[True, False],
              help="On: image bottom = lowest frequency (natural orientation)."),
        Param("level_mode", "linear", choices=["linear", "db"],
              help="Brightness -> level curve."),
        Param("gamma", 1.0, help="linear mode: brightness exponent."),
        Param("dynamic_range_db", 60.0,
              help="db mode: brightness 0..1 spans this many dB."),
        # additive
        Param("freq_scale", "log", choices=["log", "linear"],
              help="additive: frequency spacing of the partials."),
        Param("f_min", 55.0, help="additive: bottom row frequency (Hz)."),
        Param("f_max", 8000.0, help="additive: top row frequency (Hz)."),
        Param("max_partials", 512,
              help="additive: row cap; excess rows average into bands."),
        Param("interpolation", "linear", choices=["linear", "nearest"],
              help="additive: column->time amplitude interpolation."),
        Param("phase_mode", "random", choices=["random", "zero"],
              help="additive: partial start phases."),
        Param("total_seconds", 5.0, help="additive: output length."),
        Param("normalize", True, choices=[True, False],
              help="additive: peak-normalize the partial sum to 'amplitude'."),
        # istft
        Param("n_fft", 2048, help="istft: FFT size (bins = n_fft/2+1)."),
        Param("hop", 512, help="istft: hop; duration = columns*hop/sample_rate."),
        Param("iterations", 32, help="istft: Griffin-Lim iterations (0 = fast/noisy)."),
        # common
        Param("seed", 0, help="Random-phase seed (reproducibility)."),
        Param("amplitude", 0.8, help="Peak amplitude."),
        Param("sample_rate", 48000.0, help="Output sample rate (Hz)."),
    ]

    # -- shared: image -> magnitude matrix (rows = low..high freq) -----------
    def _magnitude(self, img: types.Image) -> np.ndarray:
        plane = _select_channel(np.asarray(img.data), self.values["channel"])
        plane = np.clip(np.asarray(plane, dtype=np.float64), 0.0, None)
        if self.values["flip_vertical"]:
            plane = np.flipud(plane)  # row 0 becomes the image's bottom = low freq
        if self.values["level_mode"] == "db":
            rng = float(self.values["dynamic_range_db"])
            mag = 10.0 ** ((np.clip(plane, 0.0, 1.0) - 1.0) * rng / 20.0)
            return np.where(plane <= 0.0, 0.0, mag)
        return np.clip(plane, 0.0, 1.0) ** float(self.values["gamma"])

    # -- additive oscillator bank (GPU-ready) ---------------------------------
    def _cook_additive(self, mag: np.ndarray, sr: float) -> Any:
        xp = backend.xp()
        n_rows, n_cols = mag.shape
        cap = max(1, int(self.values["max_partials"]))
        if n_rows > cap:
            # average rows into `cap` bands (documented spectral downsampling)
            edges = np.linspace(0, n_rows, cap + 1).astype(int)
            mag = np.stack([mag[a:b].mean(axis=0) for a, b in zip(edges[:-1], edges[1:])])
            n_rows = cap

        f_lo, f_hi = float(self.values["f_min"]), float(self.values["f_max"])
        ratio = np.linspace(0.0, 1.0, n_rows)
        if self.values["freq_scale"] == "log":
            freqs = f_lo * (f_hi / max(f_lo, 1e-3)) ** ratio
        else:
            freqs = f_lo + ratio * (f_hi - f_lo)
        freqs = np.minimum(freqs, sr / 2 * 0.999)  # stay below Nyquist

        n = max(1, int(round(float(self.values["total_seconds"]) * sr)))
        t = xp.arange(n, dtype=xp.float64) / sr

        if self.values["phase_mode"] == "random":
            rng = np.random.default_rng(int(self.values["seed"]))
            phases = rng.random(n_rows) * 2.0 * np.pi
        else:
            phases = np.zeros(n_rows)

        # column -> per-sample amplitude index
        pos = xp.linspace(0.0, n_cols - 1.0, n)
        if self.values["interpolation"] == "nearest":
            c0 = xp.rint(pos).astype(int)
            c1, frac = c0, None
        else:
            c0 = xp.floor(pos).astype(int)
            c1 = xp.minimum(c0 + 1, n_cols - 1)
            frac = pos - c0

        amps = backend.asarray(mag)  # (rows, cols) on the active backend
        audio = xp.zeros(n, dtype=xp.float64)
        for start in range(0, n_rows, _PARTIAL_CHUNK):
            stop = min(start + _PARTIAL_CHUNK, n_rows)
            a = amps[start:stop, c0]
            if frac is not None:
                a = a * (1.0 - frac) + amps[start:stop, c1] * frac
            f = backend.asarray(freqs[start:stop]).astype(xp.float64)[:, None]
            ph = backend.asarray(phases[start:stop]).astype(xp.float64)[:, None]
            audio += (a * xp.sin(2.0 * np.pi * f * t[None, :] + ph)).sum(axis=0)

        peak = float(backend.to_cpu(xp.abs(audio).max()))
        if self.values["normalize"] and peak > 0:
            audio = audio / peak
        return audio * float(self.values["amplitude"])

    # -- ISTFT + Griffin-Lim (CPU) ---------------------------------------------
    def _cook_istft(self, mag: np.ndarray, sr: float) -> np.ndarray:
        n_fft = max(64, int(self.values["n_fft"]))
        hop = max(16, int(self.values["hop"]))
        bins = n_fft // 2 + 1
        # rows -> FFT bins, columns -> frames (transpose to (frames, bins))
        spec_mag = spectral.resize_axis0(mag, bins).T
        audio = spectral.griffin_lim(spec_mag, n_fft=n_fft, hop=hop,
                                     iterations=int(self.values["iterations"]),
                                     seed=int(self.values["seed"]))
        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio / peak
        return audio * float(self.values["amplitude"])

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        img: types.Image = inputs["image"]
        sr = float(self.values["sample_rate"])
        mag = self._magnitude(img)
        if self.values["method"] == "additive":
            audio = self._cook_additive(mag, sr)
        else:
            audio = self._cook_istft(mag, sr)
        xp = backend.xp()
        audio = synth.apply_fades(xp.asarray(audio).astype(backend.float_dtype()), sr)
        return {"audio": Channel.mono(audio, sample_rate=sr)}

"""Stereo and mixing nodes: pan, stereo_merge, stereo_width, mix.

Per the spec, most tools emit true mono; stereo exists only where a node
deliberately creates or manipulates it. Channels with different sample rates
or lengths are aligned automatically (linear resample to the highest rate,
zero-pad to the longest) so experimental patches "just wire up".
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import backend, types
from ..core.node import Node, Param, Port, register
from ..core.types import Channel
from ..dsp import synth


def _resample_streams(data: np.ndarray, old_rate: float, new_rate: float) -> np.ndarray:
    """Linear-resample (S, N) stream data from old_rate to new_rate."""
    if old_rate == new_rate:
        return data
    n_in = data.shape[1]
    n_out = max(1, int(round(n_in * new_rate / old_rate)))
    t_in = np.arange(n_in) / old_rate
    t_out = np.arange(n_out) / new_rate
    return np.stack([np.interp(t_out, t_in, s) for s in data])


def _align(channels: list[Channel]) -> tuple[list[np.ndarray], float, int, int]:
    """Bring Channels to a common rate/length/stream-count for mixing.

    Returns (stream arrays, rate, n_samples, n_streams). Mono is broadcast to
    the widest stream count; shorter buffers are zero-padded (silence).
    """
    rate = max(ch.sample_rate for ch in channels)
    datas = [_resample_streams(backend.to_cpu(ch.data).astype(np.float64),
                               ch.sample_rate, rate) for ch in channels]
    n = max(d.shape[1] for d in datas)
    streams = max(d.shape[0] for d in datas)
    out = []
    for d in datas:
        if d.shape[0] == 1 and streams > 1:
            d = np.repeat(d, streams, axis=0)
        if d.shape[1] < n:
            d = np.pad(d, ((0, 0), (0, n - d.shape[1])))
        out.append(d)
    return out, rate, n, streams


@register
class PanNode(Node):
    """Place a mono signal in the stereo field (equal-power law).

    The optional ``pan`` input is a Channel in [-1, 1] resampled across the
    output -- wire e.g. gradient ``direction`` (via range_map) here to let the
    image steer the sound's position.
    """

    type_name = "pan"
    inputs = [Port("audio", types.CHANNEL), Port("pan", types.CHANNEL)]
    outputs = [Port("audio", types.CHANNEL)]
    params = [Param("pan", 0.0, help="-1 = hard left, 0 = center, +1 = hard right "
                                     "(pan input overrides).")]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ch: Channel = inputs["audio"]
        mono = backend.to_cpu(ch.data).astype(np.float64).mean(axis=0)  # downmix
        n = mono.shape[0]
        if "pan" in inputs:
            pan = backend.to_cpu(synth.resample_nearest(inputs["pan"].data[0], n))
            pan = np.clip(pan.astype(np.float64), -1.0, 1.0)
        else:
            pan = float(np.clip(self.values["pan"], -1.0, 1.0))
        theta = (pan + 1.0) * (np.pi / 4.0)  # 0..pi/2
        left = mono * np.cos(theta)
        right = mono * np.sin(theta)
        return {"audio": Channel(np.stack([left, right]).astype(np.float32),
                                 sample_rate=ch.sample_rate)}


@register
class StereoMergeNode(Node):
    """Two mono signals -> one stereo signal (left, right)."""

    type_name = "stereo_merge"
    inputs = [Port("left", types.CHANNEL), Port("right", types.CHANNEL)]
    outputs = [Port("audio", types.CHANNEL)]
    params = []

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        names = [n for n in ("left", "right") if n in inputs]
        if not names:
            raise ValueError("stereo_merge: connect at least one input")
        datas, rate, n, _ = _align([inputs[name] for name in names])
        by_name = dict(zip(names, datas))
        sides = [by_name[name][0] if name in by_name else np.zeros(n)
                 for name in ("left", "right")]
        return {"audio": Channel(np.stack(sides).astype(np.float32), sample_rate=rate)}


@register
class StereoWidthNode(Node):
    """Mid/side width and balance control over a stereo signal."""

    type_name = "stereo_width"
    inputs = [Port("audio", types.CHANNEL)]
    outputs = [Port("audio", types.CHANNEL)]
    params = [
        Param("width", 1.0, help="0 = mono, 1 = unchanged, 2 = doubled side."),
        Param("balance", 0.0, help="-1 = left only ... +1 = right only."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ch: Channel = inputs["audio"]
        data = backend.to_cpu(ch.data).astype(np.float64)
        if data.shape[0] == 1:
            data = np.repeat(data, 2, axis=0)
        left, right = data[0], data[1]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right) * float(self.values["width"])
        left, right = mid + side, mid - side
        bal = float(np.clip(self.values["balance"], -1.0, 1.0))
        left *= min(1.0, 1.0 - bal)
        right *= min(1.0, 1.0 + bal)
        return {"audio": Channel(np.stack([left, right]).astype(np.float32),
                                 sample_rate=ch.sample_rate)}


@register
class MixNode(Node):
    """Sum up to four Channels (auto-aligned), with per-input gain.

    The bus for multi-part compositions: split an image, synthesize each part,
    mix the parts back together (or export them separately as stems).
    """

    type_name = "mix"
    inputs = [Port("a", types.CHANNEL), Port("b", types.CHANNEL),
              Port("c", types.CHANNEL), Port("d", types.CHANNEL)]
    outputs = [Port("audio", types.CHANNEL)]
    params = [
        Param("gain_a", 1.0), Param("gain_b", 1.0),
        Param("gain_c", 1.0), Param("gain_d", 1.0),
        Param("normalize", False, choices=[True, False],
              help="Peak-normalize the sum (off = faithful raw sum)."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        names = [n for n in ("a", "b", "c", "d") if n in inputs]
        if not names:
            raise ValueError("mix: connect at least one input")
        datas, rate, n, streams = _align([inputs[n] for n in names])
        total = np.zeros((streams, n))
        for name, d in zip(names, datas):
            total += d * float(self.values[f"gain_{name}"])
        if self.values["normalize"]:
            peak = float(np.abs(total).max())
            if peak > 0:
                total /= peak
        return {"audio": Channel(total.astype(np.float32), sample_rate=rate)}

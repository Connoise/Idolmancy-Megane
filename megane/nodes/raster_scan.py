"""Raster scan node: Image -> Channel (the core "raster values to data" tool).

Traverses the image by row or column, reducing the perpendicular axis to a
single value per step, producing a 1-D control Channel. This is the data stream
that downstream translators (oscillator, and later MIDI/spectral) consume.

The output Channel's ``sample_rate`` is set to its length (a 1-second nominal
span); consumer nodes decide the real timing (e.g. via BPM / duration).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.node import Node, Param, Port, register
from ..core import types
from ..core.types import Channel
from ..io.image_io import luminance


def _select_channel(arr: np.ndarray, channel: str) -> np.ndarray:
    """Return an (H, W) plane for the requested channel selection."""
    if channel == "luminance":
        return luminance(arr)
    if arr.ndim == 2:
        if channel in ("red", "green", "blue"):
            return arr  # grayscale: treat single plane as the value
        raise ValueError(f"channel {channel!r} unavailable on grayscale image")
    index = {"red": 0, "green": 1, "blue": 2, "alpha": 3}
    if channel not in index:
        raise ValueError(f"unknown channel {channel!r}")
    i = index[channel]
    if i >= arr.shape[2]:
        raise ValueError(f"channel {channel!r} not present (image has {arr.shape[2]} channels)")
    return arr[..., i]


def _reduce(plane: np.ndarray, axis: int, how: str) -> np.ndarray:
    if how == "mean":
        return plane.mean(axis=axis)
    if how == "max":
        return plane.max(axis=axis)
    if how == "min":
        return plane.min(axis=axis)
    if how == "median":
        return np.median(plane, axis=axis)
    if how == "first":
        return plane[0, :] if axis == 0 else plane[:, 0]
    raise ValueError(f"unknown reduction {how!r}")


@register
class RasterScanNode(Node):
    type_name = "raster_scan"
    inputs = [Port("image", types.IMAGE)]
    outputs = [Port("values", types.CHANNEL)]
    params = [
        Param("axis", "column", choices=["column", "row"],
              help="Emit one value per column (scan L->R) or per row (top->bottom)."),
        Param("channel", "luminance",
              choices=["luminance", "red", "green", "blue", "alpha"],
              help="Which image channel to read."),
        Param("reduction", "mean", choices=["mean", "max", "min", "median", "first"],
              help="How to collapse the perpendicular axis to one value."),
        Param("direction", "forward", choices=["forward", "reverse"],
              help="Scan order along the chosen axis."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        image: types.Image = inputs["image"]
        plane = _select_channel(np.asarray(image.data), self.values["channel"])
        # axis=column -> one value per column -> reduce over rows (axis 0)
        red_axis = 0 if self.values["axis"] == "column" else 1
        values = _reduce(plane, red_axis, self.values["reduction"])
        if self.values["direction"] == "reverse":
            values = values[::-1]
        values = np.ascontiguousarray(values, dtype=np.float32)
        return {"values": Channel.mono(values, sample_rate=float(len(values)))}

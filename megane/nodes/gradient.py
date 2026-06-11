"""Gradient node: image vector information as direction/magnitude data.

Computes the spatial gradient of the image and scans it like raster_scan,
emitting four Channels: the mean horizontal (dx) and vertical (dy) components,
the gradient magnitude, and the gradient *direction* as an angle in [0, 1)
turns (reduced with a circular mean, like hue).

"Vector information as audio direction data": wire ``direction`` through a
range_map (0..1 -> -1..1) into a pan node and the image's edge orientations
literally steer the sound across the stereo field; ``magnitude`` makes a
natural dynamics source (edges = loud, flat areas = quiet).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import types
from ..core.node import Node, Param, Port, register
from ..core.types import Channel
from ..dsp.color import circular_mean
from .raster_scan import _select_channel


@register
class GradientNode(Node):
    type_name = "gradient"
    inputs = [Port("image", types.IMAGE)]
    outputs = [Port("dx", types.CHANNEL), Port("dy", types.CHANNEL),
               Port("magnitude", types.CHANNEL), Port("direction", types.CHANNEL)]
    params = [
        Param("channel", "luminance",
              choices=["luminance", "red", "green", "blue", "alpha"],
              help="Image channel the gradient is computed on."),
        Param("axis", "column", choices=["column", "row"],
              help="One value per column (L->R) or per row (top->bottom)."),
        Param("direction_mode", "circular_mean", choices=["circular_mean", "strongest"],
              help="Reduce angles by circular mean, or take the strongest pixel's angle."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        img: types.Image = inputs["image"]
        plane = _select_channel(np.asarray(img.data), self.values["channel"])
        plane = np.asarray(plane, dtype=np.float64)
        dy, dx = np.gradient(plane)  # d/d(row) = vertical, d/d(col) = horizontal
        red_axis = 0 if self.values["axis"] == "column" else 1

        mdx = dx.mean(axis=red_axis)
        mdy = dy.mean(axis=red_axis)
        magnitude = np.hypot(dx, dy).mean(axis=red_axis)

        angle01 = np.mod(np.arctan2(dy, dx) / (2.0 * np.pi), 1.0)
        if self.values["direction_mode"] == "strongest":
            idx = np.argmax(np.hypot(dx, dy), axis=red_axis)
            direction = np.take_along_axis(
                angle01, np.expand_dims(idx, red_axis), axis=red_axis).squeeze(red_axis)
        else:
            direction = circular_mean(angle01, axis=red_axis)

        outs = {}
        for name, vals in (("dx", mdx), ("dy", mdy),
                           ("magnitude", magnitude), ("direction", direction)):
            v = np.ascontiguousarray(vals, dtype=np.float32)
            outs[name] = Channel.mono(v, sample_rate=float(len(v)))
        return outs

"""Color scan node: traverse an image and emit three color-component Channels.

Like raster_scan, but color-aware: pixels are converted to the chosen color
space *before* reduction, and the hue component is reduced with a circular
mean (hues 0.99 and 0.01 average to ~0.0/red, not 0.5/cyan).

Per space, the outputs are:

========  ==============  ==================  =================
space     c1              c2                  c3
========  ==============  ==================  =================
hsv       hue [0,1]       saturation [0,1]    value [0,1]
cielab    hue_ab [0,1]    chroma (/128)       L (/100)
xyz       X (/D65)        Y (/D65)            Z (/D65)
rgb       red             green               blue
========  ==============  ==================  =================

Typical wiring: ``c1 -> oscillator/to_midi`` (with input_range=unit, so the
hue wheel maps directly onto the pitch range), ``c3 -> amp`` (brightness as
dynamics), ``c2 -> shape`` (saturation as wave structure).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import types
from ..core.node import Node, Param, Port, register
from ..core.types import Channel
from ..dsp import color


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
class ColorScanNode(Node):
    type_name = "color_scan"
    inputs = [Port("image", types.IMAGE)]
    outputs = [Port("c1", types.CHANNEL), Port("c2", types.CHANNEL),
               Port("c3", types.CHANNEL)]
    params = [
        Param("space", "hsv", choices=["hsv", "cielab", "xyz", "rgb"],
              help="Color space the components are read in."),
        Param("axis", "column", choices=["column", "row"],
              help="One value per column (L->R) or per row (top->bottom)."),
        Param("reduction", "mean", choices=["mean", "max", "min", "median", "first"],
              help="Collapse of the perpendicular axis (hue always uses a "
                   "circular mean when 'mean' is selected)."),
        Param("direction", "forward", choices=["forward", "reverse"],
              help="Scan order."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        img: types.Image = inputs["image"]
        rgb = color.ensure_rgb3(np.asarray(img.data, dtype=np.float64))
        axis = 0 if self.values["axis"] == "column" else 1
        how = self.values["reduction"]
        space = self.values["space"]

        if space == "hsv":
            hsv = color.rgb_to_hsv(rgb)
            h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            c1 = color.circular_mean(h, axis) if how == "mean" else _reduce(h, axis, how)
            c2 = _reduce(s, axis, how)
            c3 = _reduce(v, axis, how)
        elif space == "cielab":
            lab = color.rgb_to_lab(rgb)
            a = _reduce(lab[..., 1], axis, how)
            b = _reduce(lab[..., 2], axis, how)
            c1 = np.mod(np.arctan2(b, a) / (2.0 * np.pi), 1.0)  # hue angle of ab plane
            c2 = np.hypot(a, b) / 128.0
            c3 = _reduce(lab[..., 0], axis, how) / 100.0
        elif space == "xyz":
            xyz = color.rgb_to_xyz(rgb) / np.asarray(color.D65)
            c1 = _reduce(xyz[..., 0], axis, how)
            c2 = _reduce(xyz[..., 1], axis, how)
            c3 = _reduce(xyz[..., 2], axis, how)
        elif space == "rgb":
            c1 = _reduce(rgb[..., 0], axis, how)
            c2 = _reduce(rgb[..., 1], axis, how)
            c3 = _reduce(rgb[..., 2], axis, how)
        else:
            raise ValueError(f"unknown space {space!r}")

        outs = {}
        for name, vals in (("c1", c1), ("c2", c2), ("c3", c3)):
            if self.values["direction"] == "reverse":
                vals = vals[::-1]
            vals = np.ascontiguousarray(vals, dtype=np.float32)
            outs[name] = Channel.mono(vals, sample_rate=float(len(vals)))
        return outs

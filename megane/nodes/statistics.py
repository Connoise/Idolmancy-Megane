"""Statistics node: global image statistics as a Table and constant Channels.

The constant outputs (1-sample Channels) are the "math values fed from
another source" of the spec: wire ``mean`` into an oscillator's ``bpm`` input,
``std`` into an expression node, etc.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import types
from ..core.node import Node, Param, Port, register
from ..core.types import Channel, Table
from .raster_scan import _select_channel


@register
class StatisticsNode(Node):
    type_name = "statistics"
    inputs = [Port("image", types.IMAGE)]
    outputs = [Port("table", types.TABLE), Port("mean", types.CHANNEL),
               Port("std", types.CHANNEL), Port("min", types.CHANNEL),
               Port("max", types.CHANNEL)]
    params = [
        Param("channel", "luminance",
              choices=["luminance", "red", "green", "blue", "alpha"],
              help="Image channel the statistics are computed on."),
        Param("scale", 1.0,
              help="Multiplier applied to the constant outputs (e.g. 240 to "
                   "turn a 0..1 mean into a BPM)."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        img: types.Image = inputs["image"]
        plane = _select_channel(np.asarray(img.data), self.values["channel"]).astype(np.float64)
        k = float(self.values["scale"])

        stats = {
            "width": img.width, "height": img.height, "channels": img.channels,
            "channel": self.values["channel"],
            "mean": float(plane.mean()), "std": float(plane.std()),
            "min": float(plane.min()), "max": float(plane.max()),
            "median": float(np.median(plane)),
        }
        return {
            "table": Table(rows=stats),
            "mean": Channel.constant(stats["mean"] * k),
            "std": Channel.constant(stats["std"] * k),
            "min": Channel.constant(stats["min"] * k),
            "max": Channel.constant(stats["max"] * k),
        }

"""Split node: divide one image into four parts for multi-part composition.

Each output is a full Image, so every part can drive its own synthesis chain
(then meet again in a ``mix`` node, or export separately as stems):

* ``quadrants`` -- p1..p4 = top-left, top-right, bottom-left, bottom-right.
* ``rows``      -- four horizontal bands, top to bottom.
* ``columns``   -- four vertical bands, left to right.
* ``channels``  -- p1..p4 = R, G, B, A planes as grayscale images
  (a missing plane yields silence-black).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import types
from ..core.node import Node, Param, Port, register
from ..core.types import Image


@register
class SplitNode(Node):
    type_name = "split"
    inputs = [Port("image", types.IMAGE)]
    outputs = [Port("p1", types.IMAGE), Port("p2", types.IMAGE),
               Port("p3", types.IMAGE), Port("p4", types.IMAGE)]
    params = [
        Param("mode", "quadrants", choices=["quadrants", "rows", "columns", "channels"],
              help="How the image divides into the four parts."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        img: types.Image = inputs["image"]
        arr = np.asarray(img.data)
        mode = self.values["mode"]

        if mode == "quadrants":
            h2, w2 = arr.shape[0] // 2, arr.shape[1] // 2
            parts = [arr[:h2, :w2], arr[:h2, w2:], arr[h2:, :w2], arr[h2:, w2:]]
        elif mode == "rows":
            parts = np.array_split(arr, 4, axis=0)
        elif mode == "columns":
            parts = np.array_split(arr, 4, axis=1)
        elif mode == "channels":
            planes = arr.shape[2] if arr.ndim == 3 else 1
            parts = []
            for i in range(4):
                if arr.ndim == 3 and i < planes:
                    parts.append(arr[..., i])
                elif arr.ndim == 2 and i == 0:
                    parts.append(arr)
                else:
                    parts.append(np.zeros(arr.shape[:2], dtype=arr.dtype))
        else:
            raise ValueError(f"unknown mode {mode!r}")

        return {f"p{i + 1}": Image(data=np.ascontiguousarray(p),
                                   color_space=img.color_space,
                                   source_path=img.source_path)
                for i, p in enumerate(parts)}

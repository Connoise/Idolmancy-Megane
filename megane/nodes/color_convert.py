"""Color convert node: re-express an sRGB image in another color space.

Output planes carry the converted components (e.g. H/S/V). With ``normalize``
on, components are scaled into [0,1] so the result stays viewable in the
preview and consumable by raster_scan; off gives the raw scientific values
(L in 0..100, a/b in ~±128, XYZ ~0..1.09).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import types
from ..core.node import Node, Param, Port, register
from ..dsp import color


@register
class ColorConvertNode(Node):
    type_name = "color_convert"
    inputs = [Port("image", types.IMAGE)]
    outputs = [Port("image", types.IMAGE)]
    params = [
        Param("space", "hsv", choices=["hsv", "cielab", "xyz", "linear_rgb"],
              help="Target color space (input is assumed sRGB in [0,1])."),
        Param("normalize", True, choices=[True, False],
              help="Scale components into [0,1] (Lab/XYZ); off = raw values."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        img: types.Image = inputs["image"]
        rgb = color.ensure_rgb3(np.asarray(img.data, dtype=np.float64))
        space = self.values["space"]

        if space == "hsv":
            out = color.rgb_to_hsv(rgb)  # already [0,1]
        elif space == "linear_rgb":
            out = color.srgb_to_linear(rgb)
        elif space == "xyz":
            out = color.rgb_to_xyz(rgb)
            if self.values["normalize"]:
                out = out / np.asarray(color.D65)
        elif space == "cielab":
            out = color.rgb_to_lab(rgb)
            if self.values["normalize"]:
                out = np.stack([out[..., 0] / 100.0,
                                (out[..., 1] + 128.0) / 255.0,
                                (out[..., 2] + 128.0) / 255.0], axis=-1)
        else:
            raise ValueError(f"unknown space {space!r}")

        return {"image": types.Image(data=out.astype(np.float32),
                                     color_space=space,
                                     source_path=img.source_path)}

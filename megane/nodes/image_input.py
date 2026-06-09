"""Image input node: decode a file from disk into an :class:`Image`."""
from __future__ import annotations

from typing import Any

from ..core.node import Node, Param, Port, register
from ..core import types
from ..io.image_io import load_image


@register
class ImageInputNode(Node):
    type_name = "image_input"
    outputs = [Port("image", types.IMAGE)]
    params = [
        Param("path", "", help="Path to the source image file."),
        Param("normalize", True, choices=[True, False],
              help="Scale pixel values to [0,1] by bit depth (recommended)."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        path = self.values["path"]
        if not path:
            raise ValueError("image_input: 'path' is empty")
        img = load_image(path, normalize=self.values["normalize"])
        return {"image": img}

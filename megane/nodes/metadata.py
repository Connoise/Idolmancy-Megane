"""Metadata node: image file metadata as a Table -- and as sonifiable numbers.

Reads basic image properties plus the source file's stats and EXIF tags (when
the Image still references a file on disk). Two outputs:

* ``table``   -- everything human-readable, shown in the preview's Info tab.
* ``numbers`` -- every numeric metadata value, key-sorted, as one Channel:
  the playful/faithful path where the *metadata itself* becomes material
  (wire it straight into an oscillator or to_midi).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import ExifTags
from PIL import Image as PILImage

from ..core import types
from ..core.node import Node, Port, register
from ..core.types import Channel, Table


def _as_number(value: Any) -> float | None:
    try:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            f = float(value)
            return f if np.isfinite(f) else None
        # PIL's IFDRational and friends
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError):
        return None


@register
class MetadataNode(Node):
    type_name = "metadata"
    inputs = [Port("image", types.IMAGE)]
    outputs = [Port("table", types.TABLE), Port("numbers", types.CHANNEL)]
    params = []

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        img: types.Image = inputs["image"]
        rows: dict[str, Any] = {
            "width": img.width, "height": img.height,
            "channels": img.channels, "color_space": img.color_space,
        }

        path = img.source_path
        if path and os.path.exists(path):
            st = os.stat(path)
            rows["file"] = os.path.basename(path)
            rows["file_size"] = st.st_size
            rows["modified"] = int(st.st_mtime)
            try:
                with PILImage.open(path) as pim:
                    rows["format"] = pim.format or ""
                    for tag_id, value in pim.getexif().items():
                        name = ExifTags.TAGS.get(tag_id, f"tag_{tag_id}")
                        rows[f"exif.{name}"] = (value if isinstance(value, (int, float, str))
                                                else repr(value))
            except OSError:
                pass  # file vanished/unreadable mid-cook; keep the basics

        numbers = []
        for key in sorted(rows):
            num = _as_number(rows[key])
            if num is not None:
                numbers.append(num)
        # always non-empty: width/height/channels are numeric
        data = np.asarray(numbers, dtype=np.float32)
        return {"table": Table(rows=rows),
                "numbers": Channel.mono(data, sample_rate=float(len(data)))}

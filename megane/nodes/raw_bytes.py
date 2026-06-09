"""Raw bytes node: reinterpret a file's raw bytes as PCM audio.

This is the "raw file data converted to audio" tool -- the classic data-bending
technique. It reads the file verbatim and reinterprets the byte stream as
samples of the chosen integer/float type, with no decoding. Any file works; the
result is a faithful sonification of the bytes on disk.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core.node import Node, Param, Port, register
from ..core import types
from ..core.types import Channel

# dtype name -> (numpy dtype, conversion to float [-1, 1])
_DTYPES = {
    "uint8": (np.uint8, lambda x: (x.astype(np.float32) - 128.0) / 128.0),
    "int8": (np.int8, lambda x: x.astype(np.float32) / 128.0),
    "int16": (np.dtype("<i2"), lambda x: x.astype(np.float32) / 32768.0),
    "float32": (np.dtype("<f4"), lambda x: np.clip(x.astype(np.float32), -1.0, 1.0)),
}


@register
class RawBytesNode(Node):
    type_name = "raw_bytes"
    outputs = [Port("audio", types.CHANNEL)]
    params = [
        Param("path", "", help="File whose raw bytes become audio."),
        Param("dtype", "uint8", choices=list(_DTYPES),
              help="How to interpret the bytes as samples."),
        Param("channels", 1, choices=[1, 2], help="Mono or interleaved stereo."),
        Param("sample_rate", 8000.0, help="Playback rate (Hz). Low rates are classic."),
        Param("max_bytes", 1_048_576, help="Read at most this many bytes (0 = all)."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        path = self.values["path"]
        if not path:
            raise ValueError("raw_bytes: 'path' is empty")
        np_dtype, to_float = _DTYPES[self.values["dtype"]]

        with open(path, "rb") as f:
            raw = f.read()
        max_bytes = int(self.values["max_bytes"])
        if max_bytes > 0:
            raw = raw[:max_bytes]

        samples = np.frombuffer(raw, dtype=np_dtype)
        # Trim to a whole number of frames for the requested channel count.
        channels = int(self.values["channels"])
        usable = (samples.size // channels) * channels
        samples = samples[:usable]
        audio = to_float(samples)
        if channels == 2:
            audio = audio.reshape(-1, 2).T  # (2, N)
        return {"audio": Channel(audio, sample_rate=float(self.values["sample_rate"]))}

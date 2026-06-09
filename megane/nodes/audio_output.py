"""Audio output node: write a Channel to a WAV file (and optionally preview).

This is a sink: cooking it writes the file as a side effect and passes the audio
through unchanged so it can still be chained. The destination is
``<directory>/<filename>``; ``directory`` defaults to the graph's output dir
when left blank (resolved by the CLI/GUI).
"""
from __future__ import annotations

import os
from typing import Any

from ..core import backend, types
from ..core.node import Node, Param, Port, register
from ..core.types import Channel
from ..io import audio_io


@register
class AudioOutputNode(Node):
    type_name = "audio_output"
    inputs = [Port("audio", types.CHANNEL)]
    outputs = [Port("audio", types.CHANNEL)]  # pass-through for chaining
    params = [
        Param("filename", "output.wav", help="Output file name."),
        Param("directory", "", help="Output directory (blank = project output dir)."),
        Param("subtype", "PCM_16", choices=["PCM_16", "PCM_24", "FLOAT"],
              help="WAV sample format (24-bit/float need libsndfile)."),
        Param("normalize", False, choices=[True, False],
              help="Peak-normalize before writing (off = faithful levels)."),
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_path: str | None = None

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ch: Channel = inputs["audio"]
        data = ch.data
        if self.values["normalize"]:
            cpu = backend.to_cpu(data)
            peak = float(abs(cpu).max()) if cpu.size else 0.0
            if peak > 0:
                data = data / peak

        directory = self.values["directory"] or "output"
        path = os.path.join(directory, self.values["filename"])
        self.last_path = audio_io.write_wav(
            path, data, ch.sample_rate, subtype=self.values["subtype"]
        )
        return {"audio": ch}

    def play(self, ch: Channel) -> bool:
        """Best-effort preview playback of a rendered channel."""
        return audio_io.play(ch.data, ch.sample_rate)

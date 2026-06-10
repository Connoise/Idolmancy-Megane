"""MIDI output node: write MIDIData to a .mid file (passes data through)."""
from __future__ import annotations

import os
from typing import Any

from ..core import types
from ..core.node import Node, Param, Port, register
from ..core.types import MIDIData
from ..io import midi_io


@register
class MidiOutputNode(Node):
    type_name = "midi_output"
    inputs = [Port("midi", types.MIDI)]
    outputs = [Port("midi", types.MIDI)]  # pass-through for chaining
    params = [
        Param("filename", "output.mid", help="Output file name."),
        Param("directory", "", help="Output directory (blank = project output dir)."),
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_path: str | None = None

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        midi: MIDIData = inputs["midi"]
        directory = self.values["directory"] or "output"
        path = os.path.join(directory, self.values["filename"])
        self.last_path = midi_io.write_midi(midi, path)
        return {"midi": midi}

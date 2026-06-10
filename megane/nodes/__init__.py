"""Node library. Importing this package registers all built-in node types.

Phase 1 vertical slice:
    image_input -> raster_scan -> oscillator -> audio_output
    raw_bytes   -> audio_output

Phase 3 toolbox:
    color_convert, color_scan        -- color pipeline (HSV/CIELAB/XYZ)
    wavetable, statistics            -- waveform-from-image / image stats
    to_midi, midi_output             -- math-data -> MIDI -> .mid
    constant, expression,
    range_map, resample              -- control/math glue
"""
from __future__ import annotations

from .audio_output import AudioOutputNode
from .color_convert import ColorConvertNode
from .color_scan import ColorScanNode
from .control import ConstantNode, ExpressionNode, RangeMapNode, ResampleNode
from .image_input import ImageInputNode
from .midi_output import MidiOutputNode
from .oscillator import OscillatorNode
from .raster_scan import RasterScanNode
from .raw_bytes import RawBytesNode
from .statistics import StatisticsNode
from .to_midi import ToMidiNode
from .wavetable import WavetableNode

__all__ = [
    "ImageInputNode",
    "RasterScanNode",
    "OscillatorNode",
    "RawBytesNode",
    "AudioOutputNode",
    "ColorConvertNode",
    "ColorScanNode",
    "WavetableNode",
    "StatisticsNode",
    "ToMidiNode",
    "MidiOutputNode",
    "ConstantNode",
    "ExpressionNode",
    "RangeMapNode",
    "ResampleNode",
]

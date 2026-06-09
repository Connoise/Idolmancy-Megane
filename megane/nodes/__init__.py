"""Node library. Importing this package registers all built-in node types.

Phase 1 vertical slice:
    image_input -> raster_scan -> oscillator -> audio_output
    raw_bytes   -> audio_output
"""
from __future__ import annotations

from .image_input import ImageInputNode
from .raster_scan import RasterScanNode
from .oscillator import OscillatorNode
from .raw_bytes import RawBytesNode
from .audio_output import AudioOutputNode

__all__ = [
    "ImageInputNode",
    "RasterScanNode",
    "OscillatorNode",
    "RawBytesNode",
    "AudioOutputNode",
]

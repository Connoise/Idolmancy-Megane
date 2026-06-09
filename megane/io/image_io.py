"""Image loading via Pillow, returning Megane :class:`Image` objects.

Pillow covers the v1 input formats: JPG, PNG, GIF, TIFF, WebP, BMP, ICO.
(The legacy raster ``.img`` format is intentionally out of scope for now.)

Images are decoded to float arrays. With ``normalize=True`` (default) values are
scaled to ``[0, 1]`` by the source bit depth, which keeps mappings reproducible
and independent of whether a file was 8- or 16-bit.
"""
from __future__ import annotations

import numpy as np
from PIL import Image as PILImage

from ..core.types import Image

# Modes we map straight through; palette/CMYK/other modes are converted first.
_PASSTHROUGH_MODES = {"L", "LA", "RGB", "RGBA", "I", "I;16", "F"}


def load_image(path: str, normalize: bool = True) -> Image:
    """Load ``path`` into an :class:`Image` (H, W) or (H, W, C), float32."""
    with PILImage.open(path) as im:
        # Take the first frame of animated formats (GIF/WebP) for v1.
        if getattr(im, "is_animated", False):
            im.seek(0)
        if im.mode == "P":  # palette -> RGB(A)
            im = im.convert("RGBA" if "transparency" in im.info else "RGB")
        elif im.mode == "CMYK":
            im = im.convert("RGB")
        elif im.mode not in _PASSTHROUGH_MODES:
            im = im.convert("RGB")
        arr = np.asarray(im).astype(np.float32)

    if normalize:
        scale = _full_scale(arr)
        if scale > 0:
            arr = arr / scale
    return Image(data=arr, color_space="sRGB", source_path=path)


def _full_scale(arr: np.ndarray) -> float:
    """Infer the full-scale value for normalization from the observed peak."""
    peak = float(arr.max()) if arr.size else 0.0
    if peak <= 1.0:
        return 1.0
    if peak <= 255.0:
        return 255.0
    if peak <= 65535.0:
        return 65535.0
    return peak


def luminance(arr: np.ndarray) -> np.ndarray:
    """Rec.709 luma from an (H, W[, C]) float array -> (H, W)."""
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 1:
        return arr[..., 0]
    rgb = arr[..., :3]
    weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=arr.dtype)
    return rgb @ weights

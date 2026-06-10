"""Color-space conversions (vectorized, NumPy).

All converters take/return arrays shaped ``(..., 3)`` with sRGB inputs in
``[0, 1]``. Conversions that require linear light (XYZ, CIELAB) apply the
proper sRGB decoding first -- this matters for faithfulness: hue/lightness
relationships are only correct in the right space.

Component ranges:

* HSV      -- H, S, V all in [0, 1] (H is the angle / 360).
* XYZ      -- D65 white point; X,Y,Z roughly [0, 1.09].
* CIELAB   -- L in [0, 100], a/b roughly [-128, 127].
"""
from __future__ import annotations

import numpy as np

# D65 reference white
D65 = (0.95047, 1.0, 1.08883)

# sRGB (linear) -> XYZ, D65
_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Decode sRGB gamma to linear light."""
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """sRGB ``(..., 3)`` -> HSV, all components in [0, 1]."""
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc

    s = np.zeros_like(maxc)
    np.divide(delta, maxc, out=s, where=maxc > 0)

    h = np.zeros_like(maxc)
    mask = delta > 0
    safe = np.where(mask, delta, 1.0)
    h_r = np.mod((g - b) / safe, 6.0)
    h_g = (b - r) / safe + 2.0
    h_b = (r - g) / safe + 4.0
    h = np.where(maxc == r, h_r, np.where(maxc == g, h_g, h_b))
    h = np.where(mask, h / 6.0, 0.0)
    return np.stack([h, s, maxc], axis=-1)


def rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """sRGB ``(..., 3)`` -> CIE XYZ (D65), via linear light."""
    lin = srgb_to_linear(rgb)
    return lin @ _M_RGB2XYZ.T


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """CIE XYZ (D65) -> CIELAB."""
    xyz = np.asarray(xyz, dtype=np.float64)
    xn = xyz / np.asarray(D65)
    eps = (6.0 / 29.0) ** 3
    f = np.where(xn > eps, np.cbrt(xn), xn / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB ``(..., 3)`` -> CIELAB."""
    return xyz_to_lab(rgb_to_xyz(rgb))


def circular_mean(angles01: np.ndarray, axis: int) -> np.ndarray:
    """Mean of angular values expressed in [0, 1) turns, the right way.

    The arithmetic mean of hues 0.01 and 0.99 is 0.5 (cyan) -- wrong; they sit
    on either side of red. The circular mean via unit vectors returns ~0.0.
    """
    theta = np.asarray(angles01, dtype=np.float64) * 2.0 * np.pi
    mean_sin = np.sin(theta).mean(axis=axis)
    mean_cos = np.cos(theta).mean(axis=axis)
    return np.mod(np.arctan2(mean_sin, mean_cos) / (2.0 * np.pi), 1.0)


def ensure_rgb3(arr: np.ndarray) -> np.ndarray:
    """Coerce (H, W[, C]) image data to (H, W, 3): gray is replicated, alpha dropped."""
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        return np.repeat(arr, 3, axis=-1)
    return arr[..., :3]

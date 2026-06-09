"""Compute backend abstraction: CPU (NumPy) by default, optional GPU (CuPy).

Every heavy numeric operation in Megane is written against the array module
returned by :func:`xp`, so the *same* node code runs on CPU or GPU. This is the
single seam that lets us add CUDA acceleration (e.g. an RTX 3090) later without
rewriting any node logic -- we only flip ``CONFIG.use_gpu`` and provide CuPy.

Design notes
------------
* GPU is *optional*. If CuPy is not importable, the GPU switch is silently
  ignored and everything runs on NumPy.
* ``float_dtype`` is the global precision lever from the spec (fp16/fp32/fp64).
  fp32 is the default -- a good speed/quality balance and the most portable.
* ``to_cpu`` always returns a plain NumPy array, so I/O code (file export,
  serialization) never needs to know which backend produced the data.
"""
from __future__ import annotations

import numpy as np

# Try to make CuPy available, but never hard-require it.
_GPU_AVAILABLE = False
try:  # pragma: no cover - depends on host having CUDA + CuPy
    import cupy as _cupy

    _GPU_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import/runtime error means "no GPU"
    _cupy = None


_PRECISIONS = {"fp16": np.float16, "fp32": np.float32, "fp64": np.float64}


class _Config:
    """Process-wide compute configuration."""

    use_gpu: bool = False  # master switch; only honored when GPU is available
    float_dtype = np.float32  # precision lever


CONFIG = _Config()


def gpu_available() -> bool:
    """True if a CuPy/CUDA backend was importable at startup."""
    return _GPU_AVAILABLE


def set_gpu(enabled: bool) -> bool:
    """Request the GPU backend. Returns the effective state (False if no GPU)."""
    CONFIG.use_gpu = bool(enabled) and _GPU_AVAILABLE
    return CONFIG.use_gpu


def set_precision(name: str) -> None:
    """Set global float precision: one of ``fp16``, ``fp32``, ``fp64``."""
    if name not in _PRECISIONS:
        raise ValueError(f"unknown precision {name!r}; choose from {list(_PRECISIONS)}")
    CONFIG.float_dtype = _PRECISIONS[name]


def precision_name() -> str:
    """Return the active precision as a short string."""
    for name, dtype in _PRECISIONS.items():
        if dtype == CONFIG.float_dtype:
            return name
    return str(CONFIG.float_dtype)


def float_dtype():
    """The active floating-point dtype for new buffers."""
    return CONFIG.float_dtype


def xp():
    """Return the active array module (``numpy`` or ``cupy``)."""
    if CONFIG.use_gpu and _GPU_AVAILABLE:
        return _cupy
    return np


def asarray(a):
    """Move/convert ``a`` onto the active backend at the active float precision."""
    return xp().asarray(a, dtype=float_dtype())


def to_cpu(a) -> np.ndarray:
    """Return ``a`` as a plain NumPy array regardless of the active backend."""
    if _GPU_AVAILABLE and isinstance(a, _cupy.ndarray):  # pragma: no cover
        return _cupy.asnumpy(a)
    return np.asarray(a)

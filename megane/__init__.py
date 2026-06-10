"""Megane -- a faithful image-to-audio sonification engine.

Importing the package registers all built-in node types, so callers can build
or load graphs immediately::

    import megane
    g = megane.Graph()
    ...
"""
from __future__ import annotations

__version__ = "0.4.0"

from . import nodes  # noqa: F401  (import for side effect: node registration)
from .core import Graph, project

__all__ = ["Graph", "project", "__version__"]

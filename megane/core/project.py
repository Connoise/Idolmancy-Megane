"""Project save/load: serialize a :class:`Graph` to/from JSON.

Per the spec, source images are referenced *by path* (not embedded) to keep
project files small, and a content hash is stored alongside each image path so
the tool can warn if a referenced file has changed -- supporting reproducible
output for identical settings.

File layout (JSON)::

    {
      "format": "megane-project",
      "version": 1,
      "settings": {...},
      "nodes": [{"type": ..., "id": ..., "params": {...}}, ...],
      "connections": [["srcId","srcPort","dstId","dstPort"], ...],
      "assets": {"<path>": "<sha256>", ...}
    }
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .graph import Graph
from .node import Node

FORMAT = "megane-project"
VERSION = 1


def file_hash(path: str) -> str | None:
    """SHA-256 of a file, or None if it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _collect_assets(graph: Graph) -> dict[str, str]:
    assets: dict[str, str] = {}
    for node in graph.nodes.values():
        path = node.values.get("path")
        if isinstance(path, str) and path and os.path.exists(path):
            digest = file_hash(path)
            if digest:
                assets[path] = digest
    return assets


def to_dict(graph: Graph, ui: dict[str, Any] | None = None) -> dict[str, Any]:
    from .. import __version__  # runtime import; package init imports this module

    data = {
        "format": FORMAT,
        "version": VERSION,
        "app_version": __version__,  # informational: which Megane wrote this
        "settings": dict(graph.settings),
        "nodes": [n.to_dict() for n in graph.nodes.values()],
        "connections": [list(c) for c in graph.connections],
        "assets": _collect_assets(graph),
    }
    if ui:
        # Presentation-only data (e.g. node positions); ignored by the engine.
        data["ui"] = ui
    return data


def save(graph: Graph, path: str, ui: dict[str, Any] | None = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(graph, ui=ui), f, indent=2)
    return path


def read(path: str) -> dict[str, Any]:
    """Read a project file to its raw dict (for callers that need ``ui`` etc.)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def from_dict(data: dict[str, Any]) -> Graph:
    if data.get("format") != FORMAT:
        raise ValueError("not a Megane project file")
    graph = Graph()
    graph.settings.update(data.get("settings", {}))
    for nd in data.get("nodes", []):
        graph.add(Node.from_dict(nd))
    for c in data.get("connections", []):
        graph.connect(*c)
    return graph


def load(path: str) -> Graph:
    return from_dict(read(path))


def check_assets(data: dict[str, Any]) -> list[str]:
    """Return a list of asset paths whose current hash differs from the saved one."""
    changed = []
    for path, saved in data.get("assets", {}).items():
        if file_hash(path) != saved:
            changed.append(path)
    return changed

"""Import shim for NodeGraphQt on PySide6.

NodeGraphQt goes through the ``Qt.py`` abstraction layer; we pin its binding
choice to PySide6 *before* it is imported, and patch the couple of classes Qt6
moved from QtWidgets to QtGui when the installed Qt.py predates those moves.
All NodeGraphQt imports in the GUI go through this module.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_PREFERRED_BINDING", "PySide6")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402  (binding must load first)

try:  # patch Qt.py namespace for Qt6 class relocations, if needed
    import Qt as _qtpy  # type: ignore

    for _name in ("QUndoStack", "QUndoCommand", "QAction", "QShortcut"):
        if not hasattr(_qtpy.QtWidgets, _name) and hasattr(_qtpy.QtGui, _name):
            setattr(_qtpy.QtWidgets, _name, getattr(_qtpy.QtGui, _name))
except ImportError:
    pass

from NodeGraphQt import BaseNode, NodeGraph  # noqa: E402

__all__ = ["BaseNode", "NodeGraph", "QtCore", "QtGui", "QtWidgets",
           "port_name", "port_is_input", "node_pos", "set_node_pos"]


def port_name(port) -> str:
    """Port name across NodeGraphQt versions (attr or callable)."""
    n = port.name
    return n() if callable(n) else n


def port_is_input(port) -> bool:
    t = port.type_() if callable(getattr(port, "type_", None)) else getattr(port, "type_", "")
    return "in" in str(t).lower()


def node_pos(node) -> list[float]:
    try:
        p = node.pos()
        return [float(p[0]), float(p[1])]
    except Exception:  # noqa: BLE001 - presentation data only, never fatal
        return [0.0, 0.0]


def set_node_pos(node, xy) -> None:
    try:
        node.set_pos(float(xy[0]), float(xy[1]))
    except Exception:  # noqa: BLE001
        pass

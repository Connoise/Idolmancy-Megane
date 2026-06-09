"""EngineBridge: keep a NodeGraphQt view and a Megane engine Graph in sync.

The engine :class:`~megane.core.graph.Graph` is the single source of truth:
the visual graph is a *view* of it. User actions in the view (create node,
wire ports, delete) are forwarded to the engine through NodeGraphQt signals;
loading a project rebuilds the view from the engine (with signals suppressed
via the ``_loading`` flag). Illegal wires (port dtype mismatch) are rejected
by the engine and visually undone.
"""
from __future__ import annotations

import os
from typing import Any

from ..core import project
from ..core.graph import Graph
from ..core.node import Node, registered_nodes
from ..nodes.audio_output import AudioOutputNode
from .ng_compat import (BaseNode, NodeGraph, QtCore, node_pos, port_is_input,
                        port_name, set_node_pos)


def _make_ng_class(type_name: str, engine_cls: type[Node]):
    """Build a NodeGraphQt node class mirroring an engine node's ports."""

    def __init__(self):  # noqa: N807 - Qt-style ctor for generated class
        BaseNode.__init__(self)
        for p in engine_cls.inputs:
            self.add_input(p.name, multi_input=False)
        for p in engine_cls.outputs:
            self.add_output(p.name, multi_output=True)

    return type(type_name, (BaseNode,), {
        "__identifier__": "megane",
        "NODE_NAME": type_name,
        "__init__": __init__,
    })


class EngineBridge(QtCore.QObject):
    graphChanged = QtCore.Signal()        # structure changed (nodes/wires)
    paramChanged = QtCore.Signal(str)     # engine node id
    selectionChanged = QtCore.Signal(object)  # engine Node | None
    message = QtCore.Signal(str)          # status-bar feedback

    def __init__(self, ng: NodeGraph | None = None, parent=None) -> None:
        super().__init__(parent)
        self.ng = ng or NodeGraph()
        self.engine = Graph()
        self.cache: dict = {}             # cook cache: node_id -> (sig, outputs)
        self._ng_to_engine: dict[str, str] = {}
        self._loading = False
        self._ng_classes = {}
        for tname, ecls in registered_nodes().items():
            cls = _make_ng_class(tname, ecls)
            self.ng.register_node(cls)
            self._ng_classes[tname] = cls
        self._wire_signals()

    # -- signal plumbing ---------------------------------------------------
    def _wire_signals(self) -> None:
        g = self.ng
        g.node_created.connect(self._on_node_created)
        g.nodes_deleted.connect(self._on_nodes_deleted)
        g.port_connected.connect(self._on_port_connected)
        g.port_disconnected.connect(self._on_port_disconnected)
        if hasattr(g, "node_selection_changed"):
            g.node_selection_changed.connect(self._on_selection_changed)

    def _on_node_created(self, ng_node) -> None:
        if self._loading:
            return
        engine_node = Node.from_dict({"type": ng_node.NODE_NAME, "params": {}})
        self.engine.add(engine_node)
        self._ng_to_engine[ng_node.id] = engine_node.id
        self.graphChanged.emit()

    def _on_nodes_deleted(self, ng_ids: list) -> None:
        if self._loading:
            return
        for ng_id in ng_ids:
            eid = self._ng_to_engine.pop(ng_id, None)
            if eid:
                self.engine.remove(eid)
                self.cache.pop(eid, None)
        self.graphChanged.emit()

    def _split_ports(self, p1, p2):
        """Return (output_port, input_port) regardless of signal arg order."""
        return (p2, p1) if port_is_input(p1) else (p1, p2)

    def _on_port_connected(self, p1, p2) -> None:
        if self._loading:
            return
        out_p, in_p = self._split_ports(p1, p2)
        src = self._ng_to_engine.get(out_p.node().id)
        dst = self._ng_to_engine.get(in_p.node().id)
        if not src or not dst:
            return
        try:
            self.engine.connect(src, port_name(out_p), dst, port_name(in_p))
            self.graphChanged.emit()
        except (TypeError, KeyError) as exc:
            self.message.emit(f"connection rejected: {exc}")
            # undo the visual wire after the signal settles
            QtCore.QTimer.singleShot(0, lambda: self._undo_wire(out_p, in_p))

    def _undo_wire(self, out_p, in_p) -> None:
        self._loading = True  # suppress the resulting port_disconnected
        try:
            in_p.disconnect_from(out_p)
        except Exception:  # noqa: BLE001 - best-effort visual cleanup
            pass
        finally:
            self._loading = False

    def _on_port_disconnected(self, p1, p2) -> None:
        if self._loading:
            return
        out_p, in_p = self._split_ports(p1, p2)
        src = self._ng_to_engine.get(out_p.node().id)
        dst = self._ng_to_engine.get(in_p.node().id)
        if src and dst:
            self.engine.disconnect(src, port_name(out_p), dst, port_name(in_p))
            self.graphChanged.emit()

    def _on_selection_changed(self, selected, _deselected) -> None:
        node = self.engine_node(selected[0]) if selected else None
        self.selectionChanged.emit(node)

    # -- lookups -------------------------------------------------------------
    def engine_node(self, ng_node) -> Node | None:
        eid = self._ng_to_engine.get(ng_node.id)
        return self.engine.nodes.get(eid) if eid else None

    def ng_node_for(self, engine_id: str):
        for ng_id, eid in self._ng_to_engine.items():
            if eid == engine_id:
                return self.ng.get_node_by_id(ng_id)
        return None

    # -- user operations -------------------------------------------------------
    def add_node(self, type_name: str, pos=None) -> Node:
        """Create a node (view + engine) and return the engine node."""
        if pos is None:
            pos = [40 * (len(self.engine.nodes) % 12), 60 * (len(self.engine.nodes) % 8)]
        ng_node = self.ng.create_node(f"megane.{type_name}", pos=list(pos))
        return self.engine.nodes[self._ng_to_engine[ng_node.id]]

    def set_param(self, engine_id: str, name: str, value: Any) -> None:
        self.engine.nodes[engine_id].set(name, value)
        self.paramChanged.emit(engine_id)

    def chain_is_realtime(self, engine_id: str) -> bool:
        """True if the node and all upstream dependencies are realtime-capable."""
        node = self.engine.nodes[engine_id]
        if not node.realtime_capable:
            return False
        for port in node.inputs:
            src = self.engine._source_of(engine_id, port.name)
            if src and not self.chain_is_realtime(src[0]):
                return False
        return True

    def preview_target(self, selected: Node | None = None) -> str | None:
        """Pick the node to cook: selection, else an audio sink, else any sink."""
        if selected is not None:
            return selected.id
        sinks = self.engine.sink_nodes()
        for n in sinks:
            if isinstance(n, AudioOutputNode):
                return n.id
        return sinks[0].id if sinks else None

    def cook(self, engine_id: str) -> dict[str, Any]:
        """Cook a node through the incremental cache (call off the UI thread)."""
        # Match CLI semantics: blank audio_output directories fall back to the
        # project's output dir (set once; becomes part of the node's params).
        out_dir = self.engine.settings.get("output_dir", "output")
        for n in self.engine.nodes.values():
            if isinstance(n, AudioOutputNode) and not n.values.get("directory"):
                n.values["directory"] = out_dir
        return self.engine.cook_cached(engine_id, self.cache)

    # -- session ----------------------------------------------------------------
    def new(self) -> None:
        self._loading = True
        try:
            self.ng.clear_session()
        finally:
            self._loading = False
        self.engine = Graph()
        self.cache.clear()
        self._ng_to_engine.clear()
        self.graphChanged.emit()

    def save(self, path: str) -> str:
        positions = {}
        for ng_id, eid in self._ng_to_engine.items():
            ng_node = self.ng.get_node_by_id(ng_id)
            if ng_node is not None:
                positions[eid] = node_pos(ng_node)
        return project.save(self.engine, path, ui={"positions": positions})

    def load(self, path: str) -> None:
        raw = project.read(path)
        engine = project.from_dict(raw)
        positions = raw.get("ui", {}).get("positions", {})
        changed = project.check_assets(raw)

        self._loading = True
        try:
            self.ng.clear_session()
            self._ng_to_engine.clear()
            self.cache.clear()
            self.engine = engine

            ng_by_eid = {}
            for i, node in enumerate(engine.nodes.values()):
                pos = positions.get(node.id, [i * 220.0, (i % 4) * 120.0])
                ng_node = self.ng.create_node(f"megane.{node.type_name}", pos=list(pos))
                self._ng_to_engine[ng_node.id] = node.id
                ng_by_eid[node.id] = ng_node
            for src, sport, dst, dport in engine.connections:
                out_p = ng_by_eid[src].outputs().get(sport)
                in_p = ng_by_eid[dst].inputs().get(dport)
                if out_p is not None and in_p is not None:
                    out_p.connect_to(in_p)
        finally:
            self._loading = False

        if changed:
            self.message.emit("assets changed since save: " + ", ".join(changed))
        self.graphChanged.emit()

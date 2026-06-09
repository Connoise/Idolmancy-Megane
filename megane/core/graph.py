"""The node graph and its on-demand ("cook") evaluation.

Evaluation is *pull-based*: asking for a node's output recursively cooks its
upstream dependencies, memoizing each node once per pass. This matches the
project's pseudo-live model -- there is no continuously running audio thread;
we recompute on demand and preview/export the result.

Cycles are rejected. Heavy nodes (``realtime_capable == False``) cook the same
way here; the GUI layer is what will decide to "bake" them to a file.
"""
from __future__ import annotations

from typing import Any

from . import backend
from .node import Node


DEFAULT_SETTINGS = {
    "sample_rate": 48000.0,  # default audio output rate
    "precision": "fp32",     # global float precision lever
    "use_gpu": False,        # request CUDA backend if available
    "output_dir": "output",  # where sink nodes write files
}


class Graph:
    """A collection of nodes plus the connections between their ports."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        # connections: (src_id, src_port, dst_id, dst_port)
        self.connections: list[tuple[str, str, str, str]] = []
        self.settings: dict[str, Any] = dict(DEFAULT_SETTINGS)

    # -- construction ----------------------------------------------------
    def add(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise ValueError(f"node id {node.id!r} already in graph")
        self.nodes[node.id] = node
        return node

    def connect(self, src_id: str, src_port: str, dst_id: str, dst_port: str) -> None:
        for nid, port, kind, ports in (
            (src_id, src_port, "output", "outputs"),
            (dst_id, dst_port, "input", "inputs"),
        ):
            if nid not in self.nodes:
                raise KeyError(f"no such node {nid!r}")
            names = {p.name for p in getattr(self.nodes[nid], ports)}
            if port not in names:
                raise KeyError(f"node {nid!r} has no {kind} port {port!r}")
        # one source per input port: drop any existing wire into the target
        self.connections = [
            c for c in self.connections if not (c[2] == dst_id and c[3] == dst_port)
        ]
        self.connections.append((src_id, src_port, dst_id, dst_port))

    def _source_of(self, dst_id: str, dst_port: str) -> tuple[str, str] | None:
        for s_id, s_port, d_id, d_port in self.connections:
            if d_id == dst_id and d_port == dst_port:
                return s_id, s_port
        return None

    # -- evaluation ------------------------------------------------------
    def _apply_settings(self) -> None:
        backend.set_precision(self.settings.get("precision", "fp32"))
        backend.set_gpu(self.settings.get("use_gpu", False))

    def cook(self, node_id: str) -> dict[str, Any]:
        """Cook ``node_id`` and return its ``{port_name: data}`` outputs."""
        self._apply_settings()
        return self._cook(node_id, cache={}, visiting=set())

    def _cook(self, node_id: str, cache: dict, visiting: set) -> dict[str, Any]:
        if node_id in cache:
            return cache[node_id]
        if node_id in visiting:
            raise ValueError(f"cycle detected at node {node_id!r}")
        visiting.add(node_id)

        node = self.nodes[node_id]
        inputs: dict[str, Any] = {}
        for port in node.inputs:
            src = self._source_of(node_id, port.name)
            if src is None:
                continue
            s_id, s_port = src
            inputs[port.name] = self._cook(s_id, cache, visiting)[s_port]

        outputs = node.cook(inputs)
        visiting.discard(node_id)
        cache[node_id] = outputs
        return outputs

    # -- queries ---------------------------------------------------------
    def nodes_of_type(self, type_name: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.type_name == type_name]

    def sink_nodes(self) -> list[Node]:
        """Nodes with no outgoing connections (render targets)."""
        with_out = {c[0] for c in self.connections}
        return [n for n in self.nodes.values() if n.id not in with_out]

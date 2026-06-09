"""Node base class, ports, parameters, and the type registry.

A :class:`Node` declares typed input/output :class:`Port` objects and a list of
:class:`Param` specs. Instances hold only their parameter *values*; wiring lives
on the :class:`~megane.core.graph.Graph`. ``cook`` is the single compute entry
point: it receives a dict of input data and returns a dict of output data.

Every concrete node registers itself with :func:`register` so projects can be
serialized to/from JSON by ``type_name``.
"""
from __future__ import annotations

import uuid
from typing import Any


class Port:
    """A named input or output slot with a data-type tag (see ``types``)."""

    __slots__ = ("name", "dtype")

    def __init__(self, name: str, dtype: str) -> None:
        self.name = name
        self.dtype = dtype


class Param:
    """A node parameter spec: name, default, optional fixed choices + help.

    ``choices`` (when given) is advisory metadata for the GUI/CLI; the engine
    does not hard-reject other values, keeping nodes open to full user control.
    """

    __slots__ = ("name", "default", "choices", "help")

    def __init__(self, name: str, default: Any, choices: list | None = None, help: str = "") -> None:
        self.name = name
        self.default = default
        self.choices = choices
        self.help = help


_REGISTRY: dict[str, type["Node"]] = {}


def register(cls: type["Node"]) -> type["Node"]:
    """Class decorator: register a node type under its ``type_name``."""
    if not cls.type_name:
        raise ValueError(f"{cls.__name__} must define a non-empty type_name")
    if cls.type_name in _REGISTRY and _REGISTRY[cls.type_name] is not cls:
        raise ValueError(f"duplicate node type_name {cls.type_name!r}")
    _REGISTRY[cls.type_name] = cls
    return cls


def get_node_class(type_name: str) -> type["Node"]:
    if type_name not in _REGISTRY:
        raise KeyError(f"unknown node type {type_name!r}")
    return _REGISTRY[type_name]


def registered_nodes() -> dict[str, type["Node"]]:
    """A copy of the type registry (name -> class)."""
    return dict(_REGISTRY)


def _gen_id() -> str:
    return uuid.uuid4().hex[:8]


class Node:
    """Base class for all nodes.

    Subclasses set ``type_name``, ``inputs``, ``outputs``, ``params`` and
    implement :meth:`cook`. ``realtime_capable`` advertises whether the node is
    cheap enough for the interactive (pseudo-live) recompute path; heavy nodes
    set it to ``False`` and are expected to be "baked" to a file.
    """

    type_name: str = ""
    realtime_capable: bool = True
    inputs: list[Port] = []
    outputs: list[Port] = []
    params: list[Param] = []

    def __init__(self, node_id: str | None = None, **param_values: Any) -> None:
        self.id = node_id or _gen_id()
        self.values: dict[str, Any] = {p.name: p.default for p in self.params}
        unknown = set(param_values) - set(self.values)
        if unknown:
            raise KeyError(f"{self.type_name}: unknown params {sorted(unknown)}")
        self.values.update(param_values)

    # -- compute ---------------------------------------------------------
    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Compute outputs from a ``{port_name: data}`` dict of inputs."""
        raise NotImplementedError

    # -- convenience -----------------------------------------------------
    def get(self, name: str) -> Any:
        return self.values[name]

    def set(self, name: str, value: Any) -> None:
        if name not in self.values:
            raise KeyError(f"{self.type_name}: unknown param {name!r}")
        self.values[name] = value

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type_name, "id": self.id, "params": dict(self.values)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Node":
        node_cls = get_node_class(d["type"])
        return node_cls(node_id=d.get("id"), **d.get("params", {}))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.type_name} {self.id}>"

"""Control/math nodes: constant, expression, range_map, resample.

The glue of the toolbox: shape, combine, and re-time Channel data between
function nodes and translators.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..core import backend, types
from ..core.node import Node, Param, Port, register
from ..core.types import Channel
from ..dsp import synth


@register
class ConstantNode(Node):
    """A single constant value as a 1-sample Channel (BPM sources, offsets...)."""

    type_name = "constant"
    outputs = [Port("value", types.CHANNEL)]
    params = [Param("value", 1.0, help="The constant value.")]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"value": Channel.constant(float(self.values["value"]))}


# Whitelisted names available inside expression strings (NumPy-vectorized).
_EXPR_NAMESPACE = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "arctan2": np.arctan2,
    "abs": np.abs, "sqrt": np.sqrt, "log": np.log, "exp": np.exp,
    "floor": np.floor, "ceil": np.ceil, "round": np.round, "sign": np.sign,
    "clip": np.clip, "minimum": np.minimum, "maximum": np.maximum,
    "mean": np.mean, "std": np.std, "mod": np.mod, "power": np.power,
    "where": np.where, "pi": np.pi, "e": np.e,
}


@register
class ExpressionNode(Node):
    """Evaluate a NumPy expression over inputs ``a`` and ``b``.

    Examples: ``a * 2 + 0.5``, ``sin(a * pi) * b``, ``where(a > 0.5, a, b)``.
    Inputs of different lengths are aligned by nearest-index resampling; a
    missing input reads as 0. (Convenience for a local tool, not a sandbox.)
    """

    type_name = "expression"
    inputs = [Port("a", types.CHANNEL), Port("b", types.CHANNEL)]
    outputs = [Port("out", types.CHANNEL)]
    params = [Param("expr", "a", help="NumPy expression over a, b "
                                      "(sin, cos, clip, where, pi, ...).")]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        a_ch = inputs.get("a")
        b_ch = inputs.get("b")
        a = backend.to_cpu(a_ch.data[0]).astype(np.float64) if a_ch else np.zeros(1)
        b = backend.to_cpu(b_ch.data[0]).astype(np.float64) if b_ch else np.zeros(1)
        n = max(len(a), len(b))
        a = backend.to_cpu(synth.resample_nearest(a, n))
        b = backend.to_cpu(synth.resample_nearest(b, n))
        rate = max((c.sample_rate for c in (a_ch, b_ch) if c is not None), default=1.0)

        result = eval(  # noqa: S307 - namespace is whitelisted, builtins stripped
            compile(str(self.values["expr"]), "<expression-node>", "eval"),
            {"__builtins__": {}},
            {**_EXPR_NAMESPACE, "a": a, "b": b},
        )
        out = np.atleast_1d(np.asarray(result, dtype=np.float64))
        if out.ndim > 1:
            out = out.reshape(-1)
        return {"out": Channel.mono(out.astype(np.float32), sample_rate=rate)}


@register
class RangeMapNode(Node):
    """Remap a Channel's value range onto a new range with a chosen curve."""

    type_name = "range_map"
    inputs = [Port("values", types.CHANNEL)]
    outputs = [Port("values", types.CHANNEL)]
    params = [
        Param("in_mode", "auto", choices=["auto", "fixed"],
              help="'auto' uses the observed min/max; 'fixed' uses in_min/in_max."),
        Param("in_min", 0.0, help="fixed: input range low."),
        Param("in_max", 1.0, help="fixed: input range high."),
        Param("clip", True, choices=[True, False], help="Clip t into [0,1]."),
        Param("curve", "linear", choices=["linear", "pow", "geometric"],
              help="Mapping curve ('geometric' needs out_min > 0)."),
        Param("gamma", 1.0, help="pow: exponent applied to t."),
        Param("out_min", 0.0, help="Output range low."),
        Param("out_max", 1.0, help="Output range high."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ch: Channel = inputs["values"]
        x = backend.to_cpu(ch.data[0]).astype(np.float64)
        if self.values["in_mode"] == "auto":
            lo, hi = float(x.min()), float(x.max())
        else:
            lo, hi = float(self.values["in_min"]), float(self.values["in_max"])
        span = hi - lo
        t = (x - lo) / span if span != 0 else np.full_like(x, 0.5)
        if self.values["clip"]:
            t = np.clip(t, 0.0, 1.0)

        om, oM = float(self.values["out_min"]), float(self.values["out_max"])
        curve = self.values["curve"]
        if curve == "pow":
            t = t ** float(self.values["gamma"])
        if curve == "geometric":
            if om <= 0:
                raise ValueError("geometric curve requires out_min > 0")
            out = om * (oM / om) ** t
        else:
            out = om + t * (oM - om)
        return {"values": Channel.mono(out.astype(np.float32), ch.sample_rate)}


@register
class ResampleNode(Node):
    """Change a Channel's sample rate.

    * ``interpolate`` -- resample content to the new rate (duration preserved).
    * ``reinterpret`` -- keep the samples, relabel the rate (varispeed: pitch
      and speed shift together, like spinning a record).
    """

    type_name = "resample"
    inputs = [Port("values", types.CHANNEL)]
    outputs = [Port("values", types.CHANNEL)]
    params = [
        Param("mode", "interpolate", choices=["interpolate", "reinterpret"]),
        Param("target_rate", 48000.0, help="New sample rate (Hz)."),
    ]

    def cook(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ch: Channel = inputs["values"]
        target = float(self.values["target_rate"])
        if target <= 0:
            raise ValueError("target_rate must be > 0")
        if self.values["mode"] == "reinterpret" or ch.sample_rate == target:
            return {"values": Channel(ch.data, sample_rate=target)}

        data = backend.to_cpu(ch.data).astype(np.float64)
        n_in = data.shape[1]
        n_out = max(1, int(round(n_in * target / ch.sample_rate)))
        t_in = np.arange(n_in) / ch.sample_rate
        t_out = np.arange(n_out) / target
        streams = [np.interp(t_out, t_in, s) for s in data]
        return {"values": Channel(np.stack(streams).astype(np.float32),
                                  sample_rate=target)}

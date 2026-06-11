"""Node presets: curated starting points plus user-saved ones.

A preset is just a partial ``{param: value}`` dict for a node type. Built-in
presets ship below ("lenses" in the Megane flavor); user presets persist to
``~/.megane/presets.json`` and override built-ins on name collision. Applying
a preset only sets the listed params -- everything else keeps its current
value, so presets compose with manual tweaking.
"""
from __future__ import annotations

import json
import os
from typing import Any

# -- built-in presets ----------------------------------------------------
BUILTIN: dict[str, dict[str, dict[str, Any]]] = {
    "oscillator": {
        "pentatonic steps": {"pitch_mode": "scale", "scale": "pentatonic",
                             "root_midi": 48, "octaves": 4, "waveform": "sine"},
        "continuous sweep": {"pitch_mode": "continuous", "f_min": 110.0,
                             "f_max": 1760.0, "curve": "log", "waveform": "sine"},
        "raw saw chromatic": {"pitch_mode": "scale", "scale": "chromatic",
                              "waveform": "saw"},
    },
    "color_scan": {
        "hue wheel (HSV)": {"space": "hsv", "reduction": "mean"},
        "perceptual (CIELAB)": {"space": "cielab", "reduction": "mean"},
    },
    "to_midi": {
        "dense chromatic": {"pitch_mode": "chromatic", "low_note": 36,
                            "high_note": 96, "merge_repeats": False, "gate": 0.9},
        "sparse merged scale": {"pitch_mode": "scale", "scale": "major",
                                "merge_repeats": True, "gate": 0.8,
                                "steps_per_beat": 2.0},
    },
    "spectral": {
        "clean additive": {"method": "additive", "freq_scale": "log",
                           "f_min": 55.0, "f_max": 8000.0, "max_partials": 256,
                           "level_mode": "linear", "gamma": 2.0},
        "textured istft": {"method": "istft", "n_fft": 2048, "hop": 512,
                           "iterations": 32, "level_mode": "db",
                           "dynamic_range_db": 60.0},
    },
    "wavetable": {
        "slow morph": {"source": "row", "scan_speed": 8.0,
                       "interpolation": "linear", "frequency": 110.0},
        "histogram drone": {"source": "histogram", "bins": 256,
                            "frequency": 55.0},
    },
    "harmonic": {
        "organ-ish": {"harmonics": 8, "f0": 110.0, "gamma": 1.5},
        "bright comb": {"harmonics": 32, "f0": 220.0, "gamma": 1.0},
    },
    "raw_bytes": {
        "classic databend": {"dtype": "uint8", "sample_rate": 8000.0, "channels": 1},
    },
}


def user_presets_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".megane", "presets.json")


def _load_user() -> dict[str, dict[str, dict[str, Any]]]:
    try:
        with open(user_presets_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def list_presets(type_name: str) -> dict[str, dict[str, Any]]:
    """All presets for a node type: built-ins overlaid by user presets."""
    merged = dict(BUILTIN.get(type_name, {}))
    merged.update(_load_user().get(type_name, {}))
    return merged


def save_user_preset(type_name: str, name: str, params: dict[str, Any]) -> str:
    """Persist a user preset; returns the file path written."""
    data = _load_user()
    data.setdefault(type_name, {})[name] = dict(params)
    path = user_presets_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=repr)
    return path

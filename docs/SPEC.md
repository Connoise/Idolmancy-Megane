# Megane — Specification & Development Plan

> **Megane** (眼鏡, "glasses") is a standalone tool for **faithfully translating image
> data into audio data**. It is a lens for *perceiving images as sound* — an
> instrument for sound/image experimentation, not a music-making aid. It is the
> first of a planned family of conceptual "tools" in the **Idolmancy** field.

Status: **Phases 1–2 implemented** (engine vertical slice + node-graph GUI).
This document is the living design reference; sections marked _(planned)_ are
not yet built.

---

## 1. Guiding principles

| Principle | Meaning |
|---|---|
| **Faithful** | Audio is derived only from the image + the chosen functions. No data from outside the image, no aesthetic "sweetening." |
| **Experimental** | The point is to explore the space of image→sound mappings, not to produce conventionally pleasing music. |
| **Advanced user** | Built for a user fluent in analysis, music theory, and image processing. Nothing is dumbed down; presets exist but everything is open to full manual control. |
| **Composable** | Function nodes emit generic **math/data ("Channel") streams**; dedicated **translator nodes** convert those to audio, MIDI, etc. |
| **Reproducible** | Identical settings produce identical output (deterministic CPU path). Projects reference images by path + hash. |

## 2. Platform & distribution

- **Target:** Windows 11 (development is OS-agnostic Python; only packaging is Windows-specific).
- **Standalone.** No Idolmancer integration; Idolmancer may *launch* Megane as a
  separate process. The headless CLI (`megane render project.megane`) is the
  clean seam for that, and requires no coupling.
- **Non-commercial**, distributed via GitHub.
- **No Android / no mobile.**

## 3. Technology

| Concern | Choice |
|---|---|
| Language | **Python ≥ 3.10** |
| Numeric (CPU) | **NumPy** (+ SciPy later for spectral) |
| Numeric (GPU) | **CuPy** drop-in + **Numba** kernels _(planned, see §8)_ |
| Image I/O | **Pillow** (JPG, PNG, GIF, TIFF, WebP, BMP, ICO) |
| Audio export | **soundfile** (libsndfile); stdlib `wave` fallback (16-bit PCM) |
| Preview | **sounddevice** (best-effort; headless-safe) |
| MIDI | **mido** _(Phase 3)_ |
| GUI / node graph | **PySide6 + NodeGraphQt** ✅ |
| Spectrogram view | **pyqtgraph** ✅ (view; image↔spectral *synthesis* is Phase 4) |

The legacy raster **`.img`** format is **out of scope**.

## 4. Architecture

Headless engine + thin UI. Packages:

```
megane/
  core/     data types, backend abstraction, node model, graph engine, projects
  dsp/      pitch mapping, synthesis, spectral analysis
  io/       image decode, audio export/playback
  nodes/    the node library (registers all node types)
  cli.py    command-line front-end (and the Idolmancer launch seam)
  gui/      Qt node-graph front-end (PySide6 + NodeGraphQt + pyqtgraph)
```

**GUI architecture (Phase 2).** The engine `Graph` is the single source of
truth; the NodeGraphQt view is kept in sync by an `EngineBridge`
(`gui/bridge.py`) that forwards user actions (create/wire/delete) into the
engine and rebuilds the view on project load. Node classes for the view and the
parameter editors are **auto-generated** from each engine node's `Port`/`Param`
specs, so new nodes added in later phases appear in the GUI with zero GUI code.
Cooks run on the Qt thread pool (UI never blocks) through an **incremental cook
cache** keyed by content signatures (`Graph.signature`): a node re-cooks only
when its params, settings, source file, or upstream chain actually changed.
Auto-cook (debounced ~300 ms) fires only when the whole chain is
realtime-capable; heavy chains require a manual Cook ("bake"), per the
pseudo-live model. Known v1 limitations: copy/paste and undo-of-delete restore
structure but reset parameters to defaults; parameter edits are not undoable.

**Backend abstraction (`core/backend.py`).** All heavy math goes through
`backend.xp()` (NumPy or CuPy) and `backend.float_dtype()`. Flipping
`use_gpu`/`precision` changes the compute backend without touching node code —
this is the seam for CUDA acceleration and the **float-precision** processing-cost
lever.

**Cook model.** Evaluation is **pull-based and on-demand** (`Graph.cook(node_id)`
recursively cooks upstream, memoized per pass; cycles are rejected). This is the
**pseudo-live** model: light nodes recompute fast for preview; heavy nodes
(`realtime_capable = False`) are "baked" to a file by the GUI. There is no
continuously-running real-time audio thread in v1 (a deliberate reliability
choice; real-time streaming is a possible later add-on).

## 5. Data types (the typed ports)

Modeled on TouchDesigner's operator families:

| Type | TD analogy | Carries |
|---|---|---|
| `Image` | TOP | 2-D raster `(H, W[, C])`, float |
| `Channel` | CHOP | `(streams, samples)` + sample rate — the universal "math data"; **audio is a Channel at audio rate**, a control value is a 1-sample Channel |
| `MIDIData` | — | note events with timing _(Phase 3)_ |
| `Table` | DAT | structured stats / metadata _(Phase 5)_ |

## 6. Behavioral specs

- **Pitch — per-method toggle:** `continuous` (value→freq range, log/linear),
  `scale` (snap to scale/tuning), `note_set` (explicit MIDI notes).
- **Time:** scan nodes emit one value per row/column; the consumer node sets
  timing via **`total_seconds`** or **`bpm`** (a BPM that can be a constant or,
  later, driven by another node).
- **Color** _(Phase 3)_: pluggable color space — default **HSV**, plus **CIELAB**
  and **RGB→XYZ**; brightness/saturation as secondary/tertiary modulators.
- **Channels:** **mono by default**; stereo only via dedicated stereo nodes _(Phase 5)_.
- **Image size:** target ≤ 5000×5000; soft warning above, no hard block; minimal
  downsampling for now. **Float precision** (fp16/fp32/fp64) is the main cost lever.
- **Projects:** images by path + SHA-256; `check_assets()` warns on change;
  deterministic CPU output for identical settings.

## 7. Node library

**Phase 1 (built).** Coverage of requested methods in *italics*.

| Node | I/O | Method |
|---|---|---|
| `image_input` | → Image | decode file |
| `raster_scan` | Image → Channel | *raster values → data (row/column, reductions)* |
| `oscillator` | Channel → Audio | *value→pitch translation + sample-and-hold synth* |
| `raw_bytes` | → Audio | *raw file bytes → audio (data-bending)* |
| `audio_output` | Audio → file | *preview + export to directory* |

**Planned nodes** (mapped to the original method list):

| Method | Node(s) | Phase |
|---|---|---|
| Waveform from statistics (S&H, pitch/speed, dynamics) | `statistics`, `synth` | 3 |
| Color → audio (HSV/CIELAB/XYZ; freq-range/scale/note) | `color_convert`, `color_map` | 3 |
| Math values → MIDI | `to_midi`, `midi_output` | 3 |
| Image → spectral shape (spectrogram) | `spectral`, spectrogram view | 4 |
| Sample-rate / neighbor functions | node options + engine | 3–4 |
| Stereo manipulation | `stereo_*` | 5 |
| Harmonic affectation of a base frequency | `harmonic_synth` | 5 |
| Vector/gradient → direction | `gradient` | 5 |
| Image splitting → multi-part | `split`, `mixer` | 5 |
| Metadata processing | `metadata` | 5 |

## 8. Development plan

🔵 = stop for user feedback. Risks are front-loaded: node-graph reliability
(Phase 2) and GPU + spectral (Phase 4).

- **Phase 0 — Foundations.** Stack/architecture/naming confirmed. ✅
- **Phase 1 — Engine spine + vertical slice (headless).** Data types, backend,
  node/graph engine, projects, and the pipeline
  `image_input → raster_scan → oscillator → audio_output` plus `raw_bytes`. CLI to
  render & listen. **✅ Implemented — 🔵 listening checkpoint.**
- **Phase 2 — Node-graph GUI.** PySide6 + NodeGraphQt; wire/edit/cook/preview;
  waveform/spectrogram/image views; incremental cook cache; bake gating for
  heavy chains; project open/save with node positions. **✅ Implemented — 🔵
  go/no-go on node-graph vs. simple-UI fallback (verified headless; needs a
  hands-on pass on Windows).**
- **Phase 3 — Core toolbox.** Color pipeline + per-method pitch; statistics→synth;
  **MIDI** translator + export; control/math nodes (constant, expression,
  BPM/clock, range/curve); float-precision UI. **🔵**
- **Phase 4 — Spectral + GPU.** Image→spectrogram→audio (inverse STFT/Griffin-Lim)
  + spectrogram view; CuPy/Numba on heavy nodes; benchmark on 5000×5000. **🔵**
- **Phase 5 — Breadth.** Stereo, harmonic/additive, vector/gradient, metadata,
  single-image splitting → multi-track. **🔵**
- **Phase 6 — Projects/templates, presets, packaging.** Reproducibility hardening,
  export options, CLI/headless polish (Idolmancer seam), GitHub packaging. **🔵**

### Future (post-v1)
Multi-image composition (series/transitions/overlay/negative/combined stats — its
own image-op library), video→audio, 3D/point-cloud, audio→image, custom CUDA
kernels — and a future **"Mangekyo"** (kaleidoscope) tool for *transformation* and
multi-image recombination, complementing Megane's faithful *lens*.

## 9. CUDA note

GPU acceleration is designed in from the start **via high-level libraries**
(CuPy as a NumPy drop-in; Numba only where a custom neighbor kernel is needed) —
**not** hand-written CUDA C++. The backend abstraction already isolates this, so
enabling it is low-risk. The deterministic guarantee is on the **CPU** path; GPU
reductions may differ in the last bits unless deterministic ops are forced.

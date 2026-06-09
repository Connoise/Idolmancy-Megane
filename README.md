# Megane 眼鏡

**Megane** is a standalone tool that **faithfully translates image data into audio
data**. It treats an image as something to be *perceived as sound* — a lens for
sound/image experimentation rather than a music-making aid. The goal is to
explore the space of image→sound mappings; outputs are driven only by the image
and the functions you choose, with no aesthetic "sweetening."

Megane is the first of a planned family of conceptual "tools" in the **Idolmancy**
field. It is built around a **node graph**: function nodes emit generic numeric
("Channel") data, and translator nodes turn that into audio, MIDI, and more.

> **Status: Phase 1 — engine vertical slice.** A headless engine + CLI can already
> load an image, scan it to a data stream, synthesize a tone from it, and export a
> WAV. The node-graph GUI, color/MIDI/spectral toolbox, and GPU acceleration are
> on the roadmap. See [`docs/SPEC.md`](docs/SPEC.md) for the full spec and plan.

## Install

Requires Python ≥ 3.10.

```bash
pip install -e .            # core (numpy, Pillow)
pip install -e ".[audio]"   # + soundfile/sounddevice for 24-bit/float export & preview
# GPU (optional): install the CuPy build matching your CUDA toolkit, e.g.
#   pip install cupy-cuda12x
```

Or just `pip install -r requirements.txt`.

## Quick start

```bash
# 1. Generate a synthetic test image and sonify it (writes output/demo.wav):
python -m megane demo

# 2. Sonify your own image — one value per column, mapped to a pentatonic scale:
python -m megane quick-scan path/to/image.png -o out.wav \
    --axis column --channel luminance --pitch-mode scale --scale pentatonic

# 3. Reinterpret any file's raw bytes as audio (data-bending):
python -m megane raw path/to/anyfile -o raw.wav --dtype uint8 --sr 8000

# 4. Save a pipeline as a project and re-render it later:
python -m megane quick-scan image.png -o out.wav --save my.megane
python -m megane render my.megane

# Engine info / available nodes:
python -m megane info
```

Add `--play` to preview through your speakers (needs `sounddevice`), `--gpu` to
request the CUDA backend, and `--precision {fp16,fp32,fp64}` to trade accuracy
for speed.

## How it works

```
image_input ─▶ raster_scan ─▶ oscillator ─▶ audio_output
                (Image→data)   (data→audio)   (WAV + preview)

raw_bytes ───────────────────────────────▶ audio_output
```

Data flows between nodes as typed ports (modeled on TouchDesigner):

| Type | Carries |
|---|---|
| **Image** | a 2-D raster |
| **Channel** | numeric stream(s) + sample rate — control data *and* audio |
| **MIDIData** | note events *(planned)* |
| **Table** | statistics / metadata *(planned)* |

All heavy math runs through a backend abstraction (NumPy now, CuPy/CUDA later),
and evaluation is on-demand ("cook"): light nodes recompute fast for preview;
heavy nodes bake to a file.

## Project layout

```
megane/core    data types, backend, node + graph engine, project save/load
megane/dsp     pitch mapping, synthesis
megane/io      image decode, audio export/playback
megane/nodes   the node library
megane/cli.py  command-line interface
tests/         pytest suite
docs/SPEC.md   full specification & development plan
```

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

## Roadmap (short form)

1. ✅ **Engine vertical slice** (this release)
2. Node-graph GUI (PySide6 + NodeGraphQt)
3. Color / statistics / **MIDI** toolbox + per-method pitch
4. Spectral (image↔spectrogram) + GPU acceleration
5. Stereo, harmonics, vectors, metadata, image splitting
6. Projects/templates, presets, Windows packaging

Later: multi-image composition, video→audio, 3D/point clouds, audio→image, and a
companion **Mangekyo** (kaleidoscope) tool for *transformation* rather than
faithful translation. Full detail in [`docs/SPEC.md`](docs/SPEC.md).

## License

**TBD.** Not yet chosen. Since the project is intended to be non-commercial, a
permissive license like MIT (which allows commercial reuse) may not fit — a
non-commercial or "all rights reserved" license may be more appropriate.

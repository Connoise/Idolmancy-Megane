# Megane example templates

Open any of these in the GUI (`megane gui examples/<name>.megane` or
*File → Open*), select the `image_input` node, point its **path** at one of
your images, and cook (`F5`). Each template's `image_input.path` is left blank
on purpose.

| Template | What it does |
|---|---|
| `raster_melody.megane` | Column scan → pentatonic oscillator **and** parallel MIDI export — one analysis, two translations. |
| `color_field.megane` | Hue→pitch, brightness→dynamics, saturation→timbre (sine→saw morph), with the image's gradient direction steering the stereo pan. |
| `spectral_paint.megane` | The image *is* the spectrogram (additive resynthesis). Heavy node: the GUI asks for an explicit `F5` bake. |
| `rgb_trio.megane` | R/G/B planes split into three voices an octave apart, mixed to one output. |

Headless render (writes into the project's folder by default):

```bash
megane render examples/raster_melody.megane --out-dir output
```

Relative paths inside a project resolve against the project file's directory,
so a template plus an image in the same folder is portable as a unit.

# Installing Megane on Windows 11

## A. Run from source (recommended)

1. Install **Python 3.11+** from https://python.org (check *"Add python.exe to
   PATH"* during setup).
2. In PowerShell, from the repository folder:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -e ".[gui,audio]"
   ```

3. Launch:

   ```powershell
   megane gui                 # the node-graph interface
   megane demo --play         # quick engine check: renders + plays a test tone
   ```

### Optional: GPU acceleration (NVIDIA)

CUDA-accelerated synthesis (oscillator, wavetable, harmonic, spectral
additive) via CuPy:

```powershell
pip install cupy-cuda12x     # pick the build matching your CUDA version
megane bench --size 5000 --gpu   # compare against CPU timings
```

Then enable *Engine → Use GPU (CUDA)* in the GUI, or pass `--gpu` on the CLI.
If CuPy is missing or no CUDA device is present, Megane silently runs on the
CPU (NumPy) backend — results are equivalent; the CPU path is the
bit-reproducibility reference.

## B. Build a standalone folder app (no Python required to run)

From an activated venv with the `gui` extra installed:

```powershell
pip install pyinstaller
pyinstaller packaging/megane.spec
```

The app lands in `dist/Megane/` — launch `Megane.exe`. The build is
self-contained and can be zipped and shared. *(The spec file is maintained but
only exercised on Windows; if PyInstaller reports a missing module, add it to
`hiddenimports` in `packaging/megane.spec` and rebuild.)*

## Notes

- Audio preview needs an output device; export to WAV/FLAC works regardless.
- Projects (`.megane`) reference images by path. Keep an image next to its
  project file (relative path) to move them between machines as a unit.
- User presets live in `%USERPROFILE%\.megane\presets.json`.

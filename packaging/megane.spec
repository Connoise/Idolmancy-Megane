# PyInstaller spec for a standalone Megane build (one-folder app).
#
# Usage (on Windows, from the repo root in a venv with ".[gui,audio]"):
#   pip install pyinstaller
#   pyinstaller packaging/megane.spec
#
# Output: dist/Megane/Megane.exe
#
# Maintained but only exercised on Windows; if a module fails to load at
# runtime, add it to hiddenimports below and rebuild.

import sys

block_cipher = None

a = Analysis(
    ["megane_launcher.py"],
    pathex=["."],
    binaries=[],
    datas=[("../examples", "examples")],
    hiddenimports=[
        "megane",
        "megane.gui.app",
        "NodeGraphQt",
        "Qt",
        "pyqtgraph",
        "soundfile",
        "sounddevice",
        "mido",
        "mido.backends.rtmidi",
        "PIL",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "cupy"],  # cupy: install system-side if wanted
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Megane",
    console=False,  # GUI app; flip to True to debug startup issues
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="Megane",
)

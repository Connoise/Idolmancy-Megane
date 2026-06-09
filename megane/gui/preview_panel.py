"""Preview panel: inspect a cooked node's outputs.

Tabs: Waveform (min/max envelope), Spectrogram (STFT heatmap), Image, Info.
Toolbar: Cook, Play/Stop (when an audio device exists), Export WAV.

Long buffers are reduced to a min/max envelope before plotting so even
multi-minute renders stay responsive in the view.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from ..core import backend
from ..core.types import Channel, Image, MIDIData, Table
from ..dsp import spectral
from ..io import audio_io
from .ng_compat import QtCore, QtGui, QtWidgets

pg.setConfigOptions(imageAxisOrder="col-major")

_SPEC_MAX_SAMPLES = 20_000_000  # skip spectrogram beyond this (keeps UI snappy)


def _envelope(y: np.ndarray, max_buckets: int = 2000):
    """Reduce a long signal to interleaved per-bucket min/max for plotting."""
    n = y.shape[-1]
    if n <= max_buckets:
        return np.arange(n, dtype=np.float64), y.astype(np.float64)
    bucket = int(np.ceil(n / max_buckets))
    pad = (-n) % bucket
    yp = np.pad(y, (0, pad), mode="edge")
    yb = yp.reshape(-1, bucket)
    xs = np.repeat(np.arange(yb.shape[0], dtype=np.float64) * bucket, 2)
    ys = np.empty(xs.shape[0])
    ys[0::2] = yb.min(axis=1)
    ys[1::2] = yb.max(axis=1)
    return xs, ys


class PreviewPanel(QtWidgets.QWidget):
    cookRequested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._audio: Channel | None = None

        bar = QtWidgets.QHBoxLayout()
        self.cook_btn = QtWidgets.QPushButton("Cook  (F5)")
        self.play_btn = QtWidgets.QPushButton("Play")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.export_btn = QtWidgets.QPushButton("Export WAV…")
        self.status = QtWidgets.QLabel("")
        for b in (self.cook_btn, self.play_btn, self.stop_btn, self.export_btn):
            bar.addWidget(b)
        bar.addWidget(self.status, 1)

        self.tabs = QtWidgets.QTabWidget()
        self.wave_plot = pg.PlotWidget(title="waveform")
        self.wave_plot.setLabel("bottom", "time", units="s")
        self.spec_plot = pg.PlotWidget(title="spectrogram")
        self.spec_plot.setLabel("bottom", "time", units="s")
        self.spec_plot.setLabel("left", "frequency", units="Hz")
        self.spec_img = pg.ImageItem()
        self.spec_plot.addItem(self.spec_img)
        try:
            self.spec_img.setLookupTable(pg.colormap.get("inferno").getLookupTable())
        except Exception:  # noqa: BLE001 - colormap availability varies
            pass
        self.image_label = QtWidgets.QLabel("no image")
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        image_scroll = QtWidgets.QScrollArea()
        image_scroll.setWidget(self.image_label)
        image_scroll.setWidgetResizable(True)
        self.info = QtWidgets.QPlainTextEdit()
        self.info.setReadOnly(True)

        self.tabs.addTab(self.wave_plot, "Waveform")
        self.tabs.addTab(self.spec_plot, "Spectrogram")
        self.tabs.addTab(image_scroll, "Image")
        self.tabs.addTab(self.info, "Info")

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(bar)
        lay.addWidget(self.tabs, 1)

        self.cook_btn.clicked.connect(self.cookRequested)
        self.play_btn.clicked.connect(self._play)
        self.stop_btn.clicked.connect(self._stop)
        self.export_btn.clicked.connect(self._export)
        self._update_buttons()

    # -- displaying results ---------------------------------------------------
    def show_error(self, text: str) -> None:
        self.status.setText("cook failed — see Info tab")
        self.info.setPlainText(text)
        self.tabs.setCurrentWidget(self.info)

    def show_outputs(self, outputs: dict, node_label: str = "") -> None:
        self._audio = None
        lines = [f"cooked: {node_label}", ""]
        image: Image | None = None
        for name, value in outputs.items():
            lines.append(f"[{name}] {value!r}")
            if isinstance(value, Channel) and self._audio is None:
                self._audio = value
            elif isinstance(value, Image) and image is None:
                image = value
            elif isinstance(value, (Table, MIDIData)):
                lines.append(f"    {value}")

        if self._audio is not None:
            data = backend.to_cpu(self._audio.data)
            peak = float(np.abs(data).max()) if data.size else 0.0
            lines.append("")
            lines.append(f"duration: {self._audio.duration:.3f} s   peak: {peak:.3f}   "
                         f"streams: {self._audio.num_streams}")
            self._show_audio(self._audio)
        else:
            self.wave_plot.clear()
            self.spec_img.clear()
        self._show_image(image)
        self.info.setPlainText("\n".join(lines))
        self.status.setText("cooked ✓")
        self._update_buttons()

    def _show_audio(self, ch: Channel) -> None:
        data = backend.to_cpu(ch.data)
        y = data[0]
        xs, ys = _envelope(y)
        self.wave_plot.clear()
        self.wave_plot.plot(xs / ch.sample_rate, ys, pen=pg.mkPen(width=1))

        if 0 < y.shape[-1] <= _SPEC_MAX_SAMPLES and ch.sample_rate > 100:
            mag, duration, fmax = spectral.spectrogram_db(y, ch.sample_rate)
            self.spec_img.setImage(mag, levels=(-90.0, 0.0))
            self.spec_img.setRect(QtCore.QRectF(0, 0, duration, fmax))
        else:
            self.spec_img.clear()

    def _show_image(self, img: Image | None) -> None:
        if img is None:
            self.image_label.setText("no image output")
            self.image_label.setPixmap(QtGui.QPixmap())
            return
        arr = backend.to_cpu(img.data)
        a8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        if a8.ndim == 2:
            qimg = QtGui.QImage(np.ascontiguousarray(a8).data, a8.shape[1], a8.shape[0],
                                a8.shape[1], QtGui.QImage.Format_Grayscale8)
        else:
            rgb = np.ascontiguousarray(a8[..., :3])
            qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                rgb.shape[1] * 3, QtGui.QImage.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg.copy()))
        self.image_label.setText("")

    # -- audio actions -----------------------------------------------------------
    def _update_buttons(self) -> None:
        have_audio = self._audio is not None
        playable = have_audio and audio_io._sd is not None
        self.play_btn.setEnabled(playable)
        self.stop_btn.setEnabled(playable)
        if have_audio and audio_io._sd is None:
            self.play_btn.setToolTip("no audio playback backend (install sounddevice)")
        self.export_btn.setEnabled(have_audio)

    def _play(self) -> None:
        if self._audio is not None:
            ok = audio_io.play(self._audio.data, self._audio.sample_rate)
            self.status.setText("playing…" if ok else "playback unavailable")

    def _stop(self) -> None:
        if audio_io._sd is not None:
            try:
                audio_io._sd.stop()
            except Exception:  # noqa: BLE001
                pass

    def _export(self) -> None:
        if self._audio is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export WAV", "render.wav", "WAV files (*.wav)")
        if path:
            audio_io.write_wav(path, self._audio.data, self._audio.sample_rate)
            self.status.setText(f"exported {path}")

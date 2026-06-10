"""Tests for ISTFT/Griffin-Lim and the spectral node (Phase 4)."""
import numpy as np
import pytest

from megane.core.types import Image
from megane.dsp import spectral
from megane.nodes.spectral import SpectralNode


def _peak_freq(audio: np.ndarray, sr: float) -> float:
    spec = np.abs(np.fft.rfft(audio))
    return float(np.argmax(spec)) * sr / len(audio)


# -- dsp ------------------------------------------------------------------
def test_stft_istft_roundtrip():
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(8192)
    n_fft, hop = 512, 128
    rec = spectral.istft(spectral.stft(sig, n_fft, hop), n_fft, hop)
    # interior must reconstruct closely (edges are window-attenuated)
    a, b = n_fft, 8192 - n_fft
    err = np.abs(rec[a:b] - sig[a:b]).max()
    assert err < 1e-6


def test_griffin_lim_recovers_tone():
    sr, f, n_fft, hop = 8000.0, 440.0, 512, 128
    t = np.arange(int(sr)) / sr
    mag = np.abs(spectral.stft(np.sin(2 * np.pi * f * t), n_fft, hop))
    audio = spectral.griffin_lim(mag, n_fft, hop, iterations=16, seed=0)
    assert abs(_peak_freq(audio, sr) - f) < sr / len(audio) * 4


def test_griffin_lim_deterministic():
    mag = np.ones((20, 257)) * 0.1
    a = spectral.griffin_lim(mag, 512, 128, iterations=4, seed=7)
    b = spectral.griffin_lim(mag, 512, 128, iterations=4, seed=7)
    assert np.array_equal(a, b)


def test_resize_axis0():
    arr = np.array([[0.0, 0.0], [1.0, 2.0]])
    out = spectral.resize_axis0(arr, 3)
    assert out.shape == (3, 2)
    assert np.allclose(out[1], [0.5, 1.0])  # midpoint interpolated


# -- spectral node ------------------------------------------------------------
def _row_image(height=64, width=32, bright_row=16):
    arr = np.zeros((height, width), dtype=np.float32)
    arr[bright_row, :] = 1.0
    return Image(data=arr)


def test_additive_single_row_pitch():
    h, row = 64, 16
    img = _row_image(height=h, bright_row=row)
    node = SpectralNode(method="additive", freq_scale="linear", f_min=100.0,
                        f_max=6400.0, max_partials=h, total_seconds=0.5,
                        sample_rate=16000.0, phase_mode="zero")
    audio = np.asarray(node.cook({"image": img})["audio"].data[0], dtype=np.float64)
    # flip_vertical: bright row 16 from top = h-1-16 = 47 from bottom
    expect = 100.0 + (h - 1 - row) / (h - 1) * 6300.0
    assert abs(_peak_freq(audio, 16000.0) - expect) < 40.0
    assert np.all(np.isfinite(audio))


def test_additive_partial_cap_and_log_scale():
    img = Image(data=np.random.default_rng(0).random((200, 24)).astype(np.float32))
    node = SpectralNode(method="additive", max_partials=16, total_seconds=0.2,
                        sample_rate=8000.0)
    audio = np.asarray(node.cook({"image": img})["audio"].data)
    assert np.all(np.isfinite(audio)) and np.abs(audio).max() <= 0.8 + 1e-6


def test_istft_method_duration_and_pitch():
    h, w, row = 64, 24, 8
    img = _row_image(height=h, width=w, bright_row=row)
    sr, n_fft, hop = 16000.0, 512, 128
    node = SpectralNode(method="istft", n_fft=n_fft, hop=hop, iterations=8,
                        sample_rate=sr)
    audio = np.asarray(node.cook({"image": img})["audio"].data[0], dtype=np.float64)
    # duration = frames*hop (+ window tail)
    assert abs(len(audio) - (n_fft + hop * (w - 1))) < n_fft
    # bright row 8 from top -> (h-1-8)/(h-1) of the way up the linear bin axis
    expect = (h - 1 - row) / (h - 1) * (sr / 2)
    assert abs(_peak_freq(audio, sr) - expect) < sr / 2 * 0.05


def test_spectral_node_reproducible():
    img = Image(data=np.random.default_rng(2).random((32, 16)).astype(np.float32))
    node = SpectralNode(method="additive", total_seconds=0.2, sample_rate=8000.0,
                        max_partials=32, seed=5)
    a = np.asarray(node.cook({"image": img})["audio"].data)
    b = np.asarray(node.cook({"image": img})["audio"].data)
    assert np.array_equal(a, b)


def test_spectral_marked_heavy():
    assert SpectralNode.realtime_capable is False


def test_db_level_mode():
    img = Image(data=np.array([[0.0, 0.5, 1.0]] * 4, dtype=np.float32))
    node = SpectralNode(method="additive", level_mode="db", dynamic_range_db=60.0,
                        total_seconds=0.1, sample_rate=8000.0, max_partials=4)
    audio = np.asarray(node.cook({"image": img})["audio"].data)
    assert np.all(np.isfinite(audio))


def test_bench_command_smoke(capsys):
    from megane.cli import main

    rc = main(["bench", "--size", "48", "--duration", "0.2",
               "--partials", "16", "--iterations", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "spectral additive" in out and "ms" in out

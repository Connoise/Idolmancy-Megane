"""Tests for color-space conversions and the color nodes."""
import numpy as np
import pytest

from megane.core.types import Image
from megane.dsp import color
from megane.nodes.color_convert import ColorConvertNode
from megane.nodes.color_scan import ColorScanNode


def test_hsv_primaries():
    rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]],
                   dtype=np.float64)
    hsv = color.rgb_to_hsv(rgb)
    assert hsv[0, 0] == pytest.approx(0.0)          # red hue
    assert hsv[1, 0] == pytest.approx(1 / 3)        # green hue
    assert hsv[2, 0] == pytest.approx(2 / 3)        # blue hue
    assert hsv[3, 1] == pytest.approx(0.0)          # white: zero saturation
    assert hsv[4, 2] == pytest.approx(0.0)          # black: zero value


def test_lab_white_and_black():
    lab = color.rgb_to_lab(np.array([[1, 1, 1], [0, 0, 0]], dtype=np.float64))
    assert lab[0, 0] == pytest.approx(100.0, abs=1e-3)  # white L*=100
    assert lab[1, 0] == pytest.approx(0.0, abs=1e-3)    # black L*=0
    assert np.allclose(lab[:, 1:], 0.0, atol=1e-3)      # neutral a*,b*


def test_xyz_known_value():
    # sRGB white -> XYZ ~ D65 white point
    xyz = color.rgb_to_xyz(np.array([[1.0, 1.0, 1.0]]))
    assert np.allclose(xyz[0], color.D65, atol=1e-3)


def test_circular_mean_wraps_red():
    # hues just above and below red average back to red, not cyan
    m = color.circular_mean(np.array([0.99, 0.01]), axis=0)
    assert float(m) == pytest.approx(0.0, abs=1e-6) or float(m) == pytest.approx(1.0, abs=1e-6)


def test_color_convert_hsv_shape_and_range():
    rgb = np.random.default_rng(0).random((6, 8, 3)).astype(np.float32)
    out = ColorConvertNode(space="hsv").cook({"image": Image(data=rgb)})["image"]
    assert out.data.shape == (6, 8, 3)
    assert out.color_space == "hsv"
    assert out.data.min() >= 0.0 and out.data.max() <= 1.0


def test_color_scan_three_outputs():
    # left half red, right half blue
    arr = np.zeros((4, 8, 3), dtype=np.float32)
    arr[:, :4, 0] = 1.0   # red
    arr[:, 4:, 2] = 1.0   # blue
    outs = ColorScanNode(space="hsv", axis="column").cook({"image": Image(data=arr)})
    assert set(outs) == {"c1", "c2", "c3"}
    hue = np.asarray(outs["c1"].data[0])
    assert hue.shape == (8,)
    assert hue[0] == pytest.approx(0.0, abs=1e-6)      # red columns
    assert hue[-1] == pytest.approx(2 / 3, abs=1e-6)   # blue columns


def test_color_scan_grayscale_ok():
    gray = np.tile(np.linspace(0, 1, 10, dtype=np.float32), (3, 1))
    outs = ColorScanNode(space="hsv").cook({"image": Image(data=gray)})
    assert outs["c3"].num_samples == 10  # value channel tracks the gradient

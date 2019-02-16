import pytest
import numpy as np

from acoustics.standards.iec_61672_1_2013 import *
from scipy.signal import freqresp


def signal_fs():
    fs = 4000.0
    f = 400.0
    duration = 3.0
    samples = int(duration * fs)
    t = np.arange(samples) / fs
    x = np.sin(2.0 * np.pi * f * t)
    return x, fs


def test_fast_level():
    """Test whether integration with time-constant FAST gives the correct level.

    Note that the reference sound pressure is used.

    In this test the amplitude of the sine is 1, which means the mean squared $MS$ is 0.5
    With a reference pressure $p_r$ of 2.0e-5 the level should be 91 decibel

    .. math:: L = 10 \cdot \\log_{10}{\\left(\\frac{MS}{p_r^2} \\right)}

    .. math:: L = 10 \cdot \\log_{10}{\\left(\\frac{0.5}{(2e-5)^2} \\right)} = 91

    """
    x, fs = signal_fs()

    times, levels = fast_level(x, fs)
    assert (abs(levels.mean() - 91) < 0.05)

    x *= 4.0
    times, levels = fast_level(x, fs)
    assert (abs(levels.mean() - 103) < 0.05)


def test_slow_level():
    """Test whether integration with time-constant SLOW gives the correct level.
    """
    x, fs = signal_fs()

    times, levels = fast_level(x, fs)
    assert (abs(levels.mean() - 91) < 0.05)

    x *= 4.0
    times, levels = fast_level(x, fs)
    assert (abs(levels.mean() - 103) < 0.05)


def test_time_weighted_sound_level():
    x, fs = signal_fs()
    fast = 0.125

    times, levels = time_weighted_sound_level(x, fs, fast)
    assert (abs(levels.mean() - 91) < 0.05)

    x *= 4.0
    times, levels = time_weighted_sound_level(x, fs, fast)
    assert (abs(levels.mean() - 103) < 0.05)


def test_time_averaged_sound_level():
    x, fs = signal_fs()
    fast = 0.125

    times, levels = time_averaged_sound_level(x, fs, fast)
    assert (abs(levels.mean() - 91) < 0.05)

    x *= 4.0
    times, levels = time_averaged_sound_level(x, fs, fast)
    assert (abs(levels.mean() - 103) < 0.05)


class TestWeighting():
    @pytest.fixture(params=['A', 'C', 'Z'])
    def weighting(self, request):
        return request.param

    def test_weighting_functions(self, weighting):
        frequencies = NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES
        values = WEIGHTING_VALUES[weighting]
        function_values = WEIGHTING_FUNCTIONS[weighting](frequencies)
        assert (np.abs(values - function_values).max() < 0.3)

    def test_weighting_systems(self, weighting):
        frequencies = NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES
        values = WEIGHTING_VALUES[weighting]
        w, H = freqresp((WEIGHTING_SYSTEMS[weighting]()), w=2.0 * np.pi * frequencies)
        results = 20.0 * np.log10(np.abs(H))
        assert (np.abs(values - results).max() < 0.3)

import numpy as np

import pytest

has_matplotlib = pytest.importorskip("matplotlib")

if has_matplotlib:
    from acoustics.bands import octave, third
    from acoustics.imaging import plot_octave, plot_third, plot_bands


def setup_module(imaging):
    imaging.octaves = octave(16, 16000)
    imaging.thirds = third(63, 8000)
    imaging.tl_oct = np.array([3, 4, 5, 12, 15, 24, 28, 23, 35, 45, 55])
    imaging.tl_third = np.array([0, 0, 0, 1, 1, 2, 3, 5, 8, 13, 21, 32, 41, 47, 46, 44, 58, 77, 61, 75, 56, 54])
    imaging.title = 'Title'
    imaging.label = 'Label'


def test_plot_octave():
    plot_octave(tl_oct, octaves)


def test_plot_octave_kHz():
    plot_octave(tl_oct, octaves, kHz=True, xlabel=label, ylabel=label, title=title, separator='.')


def test_plot_third_octave():
    plot_third(tl_third, thirds, marker='s', separator=',')


def test_plot_third_octave_kHz():
    plot_third(tl_third, thirds, marker='s', kHz=True, xlabel=label, ylabel=label, title=title)


def test_plot_band_oct():
    plot_bands(tl_oct, octaves, axes=None, band_type='octave')


def teardown_module(imaging):
    pass

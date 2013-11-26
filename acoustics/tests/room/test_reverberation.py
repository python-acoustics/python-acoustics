import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from ...room.reverberation import (t60_sabine, t60_eyring, t60_millington,
                                   t60_fitzroy, t60_arau)


def setup_module(reverberation):
    reverberation.surfaces = np.array([240, 600, 500])
    reverberation.alpha = np.array([0.1, 0.25, 0.45])
    reverberation.alpha_bands = np.array([[0.1,   0.1, 0.1, 0.1],
                                          [0.25, 0.25, 0.25, 0.25],
                                          [0.45, 0.45, 0.45, 0.45]])
    reverberation.volume = 3000


def test_t60_sabine():
    calculated = t60_sabine(surfaces, alpha, volume)
    real = 1.211382149
    assert_almost_equal(calculated, real)


def test_t60_sabine_bands():
    calculated = t60_sabine(surfaces, alpha_bands, volume)
    real = np.array([1.211382149, 1.211382149, 1.211382149, 1.211382149])
    assert_array_almost_equal(calculated, real)


def test_t60_eyring():
    calculated = t60_eyring(surfaces, alpha, volume)
    real = 1.020427763
    assert_almost_equal(calculated, real)


def test_t60_eyring_bands():
    calculated = t60_eyring(surfaces, alpha_bands, volume)
    real = np.array([1.020427763, 1.020427763, 1.020427763, 1.020427763])
    assert_array_almost_equal(calculated, real)


def test_t60_millington():
    calculated = t60_millington(surfaces, alpha, volume)
    real = 1.020427763
    assert_almost_equal(calculated, real)


def test_t60_millington_bands():
    calculated = t60_millington(surfaces, alpha_bands, volume)
    real = np.array([1.020427763, 1.020427763, 1.020427763,
                     1.020427763])
    assert_array_almost_equal(calculated, real)


def test_t60_fitzroy():
    surfaces_fitzroy = np.array([240, 240, 600, 600, 500, 500])
    alpha_fitzroy = np.array([0.1, 0.1, 0.25, 0.25, 0.45, 0.45])
    calculated = t60_fitzroy(surfaces_fitzroy, alpha_fitzroy, volume)
    real = 0.699854185
    assert_almost_equal(calculated, real)


def test_t60_fitzroy_bands():
    surfaces_fitzroy = np.array([240, 240, 600, 600, 500, 500])
    alpha_bands_f = np.array([[0.1, 0.1, 0.25, 0.25, 0.45, 0.45],
                              [0.1, 0.1, 0.25, 0.25, 0.45, 0.45],
                              [0.1, 0.1, 0.25, 0.25, 0.45, 0.45]])
    calculated = t60_fitzroy(surfaces_fitzroy, alpha_bands_f, volume)
    real = np.array([0.699854185, 0.699854185, 0.699854185])
    assert_array_almost_equal(calculated, real)


def test_t60_arau():
    Sx = surfaces[0]
    Sy = surfaces[1]
    Sz = surfaces[2]
    calculated = t60_arau(Sx, Sy, Sz, alpha, volume)
    real = 1.142442931
    assert_almost_equal(calculated, real)


def test_t60_arau_bands():
    Sx = surfaces[0]
    Sy = surfaces[1]
    Sz = surfaces[2]
    calculated = t60_arau(Sx, Sy, Sz, alpha_bands, volume)
    real = np.array([1.142442931, 1.142442931, 1.142442931, 1.142442931])
    assert_array_almost_equal(calculated, real)


def teardown_module(reverberation):
    pass

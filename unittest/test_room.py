import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from acoustics.room import (t60_sabine, t60_eyring, t60_millington,
                                   t60_fitzroy, t60_arau)

from acoustics.room import mean_alpha, nrc




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



def test_mean_alpha_float():
    alpha = 0.1
    surface = 10
    calculated = mean_alpha(alpha, surface)
    real = 0.1
    assert_almost_equal(calculated, real)


def test_mean_alpha_1d():
    alpha = np.array([0.1, 0.2, 0.3])
    surfaces = np.array([20, 30, 40])
    calculated = mean_alpha(alpha, surfaces)
    real = 0.222222222
    assert_almost_equal(calculated, real)


def test_mean_alpha_bands():
    alpha = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]])
    surfaces = np.array([20, 30, 40])
    calculated = mean_alpha(alpha, surfaces)
    real = np.array([0.222222222, 0.222222222, 0.222222222])
    assert_array_almost_equal(calculated, real)


def test_nrc_1d():
    alpha = np.array([0.1, 0.25, 0.5, 0.9])
    calculated = nrc(alpha)
    real = 0.4375
    assert_almost_equal(calculated, real)


def test_nrc_2d():
    alphas = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]])
    calculated = nrc(alphas)
    real = np.array([0.25, 0.55])
    assert_array_almost_equal(calculated, real)







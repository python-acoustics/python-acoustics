import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from ...room.absorption import mean_alpha, nrc


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







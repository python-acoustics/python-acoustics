import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from ...utils.physics import wavelength, w


def test_wavelength1():
    calculated = wavelength(200)
    quasi = 1.715
    assert_almost_equal(calculated, quasi)


def test_wavelength2():
    calculated = wavelength(200, 343)
    quasi = 1.715
    assert_almost_equal(calculated, quasi)


def test_wavelength3():
    freqs = np.array([555, 3333, 11111])
    calculated = wavelength(freqs)
    quasi = np.array([0.618018018, 0.102910291, 0.030870308])
    assert_array_almost_equal(calculated, quasi)


def test_w1():
    calculated = w(1000)
    quasi = 6283.185307179
    assert_almost_equal(calculated, quasi)


def test_w2():
    calculated = w(np.array([1000, 2000, 3000]))
    quasi = np.array([6283.185307179, 12566.370614359, 18849.555921538])
    assert_array_almost_equal(calculated, quasi)

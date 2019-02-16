import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal, assert_array_almost_equal)

from acoustics.utils import esum, _is_1d, mean_tl, wavelength, w


def test_mean_tl_1d():
    tl = np.array([20, 30, 40, 50])
    surfaces = np.array([50, 40, 30, 20])
    tl_real = 24.1116827
    calculated = mean_tl(tl, surfaces)
    assert_almost_equal(calculated, tl_real)


def test_mean_tl_2d():
    tl = np.array([[20, 30, 40, 50], [20, 30, 40, 50]])
    surfaces = np.array([[50, 40, 30, 20], [1, 10, 11, 22]])
    tl_real = np.array([24.1116827, 33.1466548])
    calculated = mean_tl(tl, surfaces)
    assert_array_almost_equal(calculated, tl_real)


def test_esum_1d():
    calculated = esum(np.array([90, 90, 90]))
    real = 94.77121255
    assert_almost_equal(calculated, real)


def test_esum_2d_default_axis():
    calculated = esum(np.array([[90, 90, 90], [80, 80, 80]]))
    real = np.array(95.18513939877889)
    #real = np.array([94.77121255, 84.77121255])
    assert_array_almost_equal(calculated, real)


def test_esum_2d_axis0():
    calculated = esum(np.array([[90, 90, 90], [80, 80, 80]]), axis=0)
    real = np.array([90.41392685, 90.41392685, 90.41392685])
    assert_almost_equal(calculated, real)


def test__is_1d_float():
    a = 0.9
    is_float = _is_1d(a)
    assert a == is_float


def test__is_1d_1darray():
    a = np.array([0.1, 0.2, 0.3])
    is_1d_array = _is_1d(a)
    a_return = np.array([a])
    assert_array_equal(a_return, is_1d_array)


def test__is_1d_2darray():
    a = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    is_2d_array = _is_1d(a)
    assert_array_equal(a, is_2d_array)


def test__is1d_vector_2darray():
    a = np.array([[0.1, 0.2, 0.3]])
    is_vector_2darray = _is_1d(a)
    a_return = np.array([0.1, 0.2, 0.3])
    assert_array_equal(a_return, is_vector_2darray)


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

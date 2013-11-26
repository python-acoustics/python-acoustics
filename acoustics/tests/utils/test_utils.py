import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_array_almost_equal)

from ...utils.utils import _e10, esum, _is_1d, mean_TL


def test__e10_int():
    calculated = _e10(90)
    real = 1e9
    assert calculated == real


def test__e10_array():
    calculated = _e10(np.array([90, 80, 70]))
    real = np.array([1e9, 1e8, 1e7])
    assert_array_almost_equal(calculated, real)


def test_mean_TL_1d():
    tl = np.array([20, 30, 40, 50])
    surfaces = np.array([50, 40, 30, 20])
    tl_real = 24.1116827
    calculated = mean_TL(tl, surfaces)
    assert_almost_equal(calculated, tl_real)


def test_mean_TL_2d():
    tl = np.array([[20, 30, 40, 50], [20, 30, 40, 50]])
    surfaces = np.array([[50, 40, 30, 20], [1, 10, 11, 22]])
    tl_real = np.array([24.1116827, 33.1466548])
    calculated = mean_TL(tl, surfaces)
    assert_array_almost_equal(calculated, tl_real)


def test_esum_1d():
    calculated = esum(np.array([90, 90, 90]))
    real = 94.77121255
    assert_almost_equal(calculated, real)


def test_esum_2d():
    calculated = esum(np.array([[90, 90, 90], [80, 80, 80]]))
    real = np.array([94.77121255, 84.77121255])
    assert_array_almost_equal(calculated, real)


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

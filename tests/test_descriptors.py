from __future__ import division

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from acoustics.descriptors import leq, sel, lw, lden, ldn


def setup_module(descriptors):
    descriptors.levels = [80, 80.3, 80.6, 80.4, 79.7, 76.9]
    descriptors.lday = np.array([80, 70])
    descriptors.levening = np.array([65, 55])
    descriptors.lnight = np.array([45, 55])
    descriptors.power = np.array([1e-3, 1e-5, 1e-7])


def test_leq():
    calculated = leq(levels)
    real = 79.806989166
    assert_almost_equal(calculated, real)


def test_leq_int_time():
    calculated = leq(levels, int_time=1/8)
    real = 88.837889036
    assert_almost_equal(calculated, real)


def test_sel():
    calculated = sel(levels)
    real = 87.588501670
    assert_almost_equal(calculated, real)


def test_lw_int():
    calculated = lw(power[0])
    real = 90
    assert calculated == real


def test_lw_1darray():
    calculated = lw(power)
    real = np.array([90, 70, 50])
    assert_array_almost_equal(calculated, real)


def test_lw_1darray_wref():
    calculated = lw(power, Wref=1e-11)
    real = np.array([80, 60, 40])
    assert_array_almost_equal(calculated, real)


def test_lden_float():
    calculated = lden(lday[0], levening[0], lnight[0])
    real = 77.14095579
    assert_almost_equal(calculated, real)


def test_lden_array():
    calculated = lden(lday, levening, lnight)
    real = np.array([77.14095579, 67.93843392])
    assert_array_almost_equal(calculated, real)


def test_ldn_float():
    calculated = ldn(lday[0], lnight[0])
    real = 77.96703252
    assert_almost_equal(calculated, real)


def test_ldn_array():
    calculated = ldn(lday, lnight)
    real = np.array([77.96703252, 68.71330861])
    assert_array_almost_equal(calculated, real)


def teardown_module(descriptors):
    pass

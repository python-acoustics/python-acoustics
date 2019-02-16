import pytest

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from acoustics.descriptors import leq, sel, lw, lden, ldn

@pytest.fixture
def levels():
    return [80, 80.3, 80.6, 80.4, 79.7, 76.9]

@pytest.fixture
def lday():
    return np.array([80, 70])

@pytest.fixture
def levening():
    return np.array([65, 55])

@pytest.fixture
def lnight():
    return np.array([45, 55])

@pytest.fixture
def power():
    return np.array([1e-3, 1e-5, 1e-7])

def test_leq(levels):
    calculated = leq(levels)
    real = 79.806989166
    assert_almost_equal(calculated, real)


def test_leq_int_time(levels):
    calculated = leq(levels, int_time=1/8)
    real = 88.837889036
    assert_almost_equal(calculated, real)


def test_sel(levels):
    calculated = sel(levels)
    real = 87.588501670
    assert_almost_equal(calculated, real)


def test_lw_int(power):
    calculated = lw(power[0])
    real = 90
    assert calculated == real


def test_lw_1darray(power):
    calculated = lw(power)
    real = np.array([90, 70, 50])
    assert_array_almost_equal(calculated, real)


def test_lw_1darray_wref(power):
    calculated = lw(power, Wref=1e-11)
    real = np.array([80, 60, 40])
    assert_array_almost_equal(calculated, real)


def test_lden_float(lday, levening, lnight):
    calculated = lden(lday[0], levening[0], lnight[0])
    real = 77.14095579
    assert_almost_equal(calculated, real)


def test_lden_array(lday, levening, lnight):
    calculated = lden(lday, levening, lnight)
    real = np.array([77.14095579, 67.93843392])
    assert_array_almost_equal(calculated, real)


def test_ldn_float(lday, lnight):
    calculated = ldn(lday[0], lnight[0])
    real = 77.96703252
    assert_almost_equal(calculated, real)


def test_ldn_array(lday, lnight):
    calculated = ldn(lday, lnight)
    real = np.array([77.96703252, 68.71330861])
    assert_array_almost_equal(calculated, real)

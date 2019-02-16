import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

import pytest

from acoustics.building import (rw_curve, rw, rw_c, rw_ctr, stc, stc_curve,
                                mass_law)

@pytest.fixture
def tl():
    return np.array([10, 11, 12, 13, 15, 18, 20, 20, 25, 22, 18, 15,
                            20, 25, 28, 30])


def test_rw_curve(tl):
    calculated = rw_curve(tl)
    real = np.array([3, 6, 9, 12, 15, 18, 21, 22, 23, 24, 25, 26, 26,
                     26, 26, 26])
    assert_array_almost_equal(calculated, real)


def test_rw(tl):
    calculated = rw(tl)
    real = 22
    assert_almost_equal(calculated, real)


def test_rw_c(tl):
    calculated = rw_c(tl)
    real = 19.657473180
    assert_almost_equal(calculated, real)


def test_rw_ctr(tl):
    calculated = rw_ctr(tl)
    real = 18.036282792
    assert_almost_equal(calculated, real)


def test_stc(tl):
    tl = np.array([18, 18, 17, 17, 20, 21, 22, 23, 25, 26, 26, 25, 25, 26,
                   30, 31])
    calculated = stc(tl)
    real = 25
    assert_almost_equal(calculated, real)


def test_stc_curve(tl):
    tl = np.array([18, 18, 17, 17, 20, 21, 22, 23, 25, 26, 26, 25, 25, 26,
                   30, 31])
    calculated = stc_curve(tl)
    real = np.array([9, 12, 15, 18, 21, 24, 25, 26, 27, 28, 29, 29, 29, 29,
                     29, 29])
    assert_array_almost_equal(calculated, real)


@pytest.mark.parametrize("freq, vol_density, thickness, theta, c, rho0, \
                         expected", [
    (200, 2000, 0.005, 0, 343, 1.225, 23.514371494),
    (150, 8000, 0.002, 20, 345, 1.18, 24.827231087),
    (np.array([200, 150]), 8000, 0.002, 20, 345, 1.18,
     np.array([27.319749023, 24.827231087])),
])
def test_mass_law(freq, vol_density, thickness, theta, c, rho0, expected):
    calculated = mass_law(freq, vol_density, thickness, theta, c, rho0)
    assert_array_almost_equal(calculated, expected)

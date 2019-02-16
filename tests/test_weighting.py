import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from acoustics.weighting import z2a, a2z, z2c, c2z, a2c, c2a


def setup_module(weighting):
    weighting.levels = 100 * np.ones(33)
    weighting.lowest = 12.5
    weighting.highest = 20000


def test_z2a():
    generated = z2a(levels, lowest, highest)
    real = np.array([
        36.6, 43.3, 49.5, 55.3, 60.6, 65.4, 69.8, 73.8, 77.5, 80.9, 83.9, 86.6, 89.1, 91.4, 93.4, 95.2, 96.8, 98.1,
        99.2, 100, 100.6, 101, 101.2, 101.3, 101.2, 101, 100.5, 99.9, 98.9, 97.5, 95.7, 93.4, 90.7
    ])
    assert_array_equal(real, generated)


def test_a2z():
    generated = a2z(levels, lowest, highest)
    real = np.array([
        163.4, 156.7, 150.5, 144.7, 139.4, 134.6, 130.2, 126.2, 122.5, 119.1, 116.1, 113.4, 110.9, 108.6, 106.6, 104.8,
        103.2, 101.9, 100.8, 100, 99.4, 99, 98.8, 98.7, 98.8, 99, 99.5, 100.1, 101.1, 102.5, 104.3, 106.6, 109.3
    ])
    assert_array_equal(real, generated)


def test_z2c():
    generated = z2c(levels, lowest, highest)
    real = np.array([
        88.8, 91.5, 93.8, 95.6, 97, 98, 98.7, 99.2, 99.5, 99.7, 99.8, 99.9, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        99.9, 99.8, 99.7, 99.5, 99.2, 98.7, 98, 97, 95.6, 93.8, 91.5, 88.8
    ])
    assert_array_equal(real, generated)


def test_c2z():
    generated = c2z(levels, lowest, highest)
    real = np.array([
        111.2, 108.5, 106.2, 104.4, 103, 102, 101.3, 100.8, 100.5, 100.3, 100.2, 100.1, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100.1, 100.2, 100.3, 100.5, 100.8, 101.3, 102, 103, 104.4, 106.2, 108.5, 111.2
    ])
    assert_array_equal(real, generated)


def test_a2c():
    generated = a2c(levels, lowest, highest)
    real = np.array([
        152.2, 148.2, 144.3, 140.3, 136.4, 132.6, 128.9, 125.4, 122, 118.8, 115.9, 113.3, 110.9, 108.6, 106.6, 104.8,
        103.2, 101.9, 100.8, 100, 99.4, 98.9, 98.6, 98.4, 98.3, 98.2, 98.2, 98.1, 98.1, 98.1, 98.1, 98.1, 98.1
    ])
    assert_array_almost_equal(real, generated)


def test_c2a():
    generated = c2a(levels, lowest, highest)
    real = np.array([
        47.8, 51.8, 55.7, 59.7, 63.6, 67.4, 71.1, 74.6, 78, 81.2, 84.1, 86.7, 89.1, 91.4, 93.4, 95.2, 96.8, 98.1, 99.2,
        100, 100.6, 101.1, 101.4, 101.6, 101.7, 101.8, 101.8, 101.9, 101.9, 101.9, 101.9, 101.9, 101.9
    ])
    assert_array_almost_equal(real, generated)


def teardown_module(weighting):
    pass

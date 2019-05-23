import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from acoustics.weighting import z2a, a2z, z2c, c2z, a2c, c2a, noaa_lf, noaa_mf, noaa_hf, noaa_pw, noaa_ow


def setup_module(weighting):
    weighting.levels = 100 * np.ones(43)
    weighting.lowest = 12.5
    weighting.highest = 200000


def test_z2a():
    generated = z2a(levels, lowest, highest)
    real = np.array([
        36.6, 43.3, 49.5, 55.3, 60.6, 65.4, 69.8, 73.8, 77.5, 80.9, 83.9, 86.6, 89.1, 91.4, 93.4, 95.2, 96.8, 98.1,
        99.2, 100., 100.6, 101., 101.2, 101.3, 101.2, 101., 100.5, 99.9, 98.9, 97.5, 95.7, 93.4, 90.7, 87.7, 84.3,
        80.6, 77., 73.2, 69.1, 65.3, 61.5, 57.2, 53.4
        ])
    assert_array_equal(real, generated)


def test_a2z():
    generated = a2z(levels, lowest, highest)
    real = np.array([
        163.4, 156.7, 150.5, 144.7, 139.4, 134.6, 130.2, 126.2, 122.5, 119.1, 116.1, 113.4, 110.9, 108.6, 106.6, 104.8,
        103.2, 101.9, 100.8, 100., 99.4, 99., 98.8, 98.7, 98.8, 99., 99.5, 100.1, 101.1, 102.5, 104.3, 106.6, 109.3,
        112.3, 115.7, 119.4, 123., 126.8, 130.9, 134.7, 138.5, 142.8, 146.6])
    assert_array_equal(real, generated)


def test_z2c():
    generated = z2c(levels, lowest, highest)
    real = np.array([
        88.8, 91.5, 93.8, 95.6, 97., 98., 98.7, 99.2, 99.5, 99.7, 99.8, 99.9, 100., 100., 100., 100., 100., 100., 100.,
        100., 100., 99.9, 99.8, 99.7, 99.5, 99.2, 98.7, 98., 97., 95.6, 93.8, 91.5, 88.8, 85.7, 82.4, 78.7, 75.1, 71.2,
        67.2, 63.4, 59.6, 55.3, 51.4
    ])
    assert_array_equal(real, generated)


def test_c2z():
    generated = c2z(levels, lowest, highest)
    real = np.array([
        111.2, 108.5, 106.2, 104.4, 103., 102., 101.3, 100.8, 100.5,100.3, 100.2, 100.1, 100., 100., 100., 100., 100.,
        100., 100., 100., 100., 100.1, 100.2, 100.3, 100.5, 100.8, 101.3,102., 103., 104.4, 106.2, 108.5, 111.2, 114.3,
        117.6, 121.3, 124.9, 128.8, 132.8, 136.6, 140.4, 144.7, 148.6])
    assert_array_equal(real, generated)


def test_a2c():
    generated = a2c(levels, lowest, highest)
    real = np.array([
        152.2, 148.2, 144.3, 140.3, 136.4, 132.6, 128.9, 125.4, 122., 118.8, 115.9, 113.3, 110.9, 108.6, 106.6, 104.8,
        103.2, 101.9, 100.8, 100., 99.4, 98.9, 98.6, 98.4, 98.3, 98.2, 98.2, 98.1, 98.1, 98.1, 98.1, 98.1, 98.1, 98.,
        98.1, 98.1, 98.1, 98., 98.1, 98.1, 98.1, 98.1, 98.])
    assert_array_almost_equal(real, generated)


def test_c2a():
    generated = c2a(levels, lowest, highest)
    real = np.array([
        47.8, 51.8, 55.7, 59.7, 63.6, 67.4, 71.1, 74.6, 78., 81.2, 84.1, 86.7, 89.1, 91.4, 93.4, 95.2, 96.8, 98.1,
        99.2, 100., 100.6, 101.1, 101.4, 101.6, 101.7, 101.8, 101.8, 101.9, 101.9, 101.9, 101.9, 101.9, 101.9, 102.,
        101.9, 101.9, 101.9, 102., 101.9, 101.9, 101.9, 101.9, 102.
    ])
    assert_array_almost_equal(real, generated)


def test_noaa_lf():
    generated = noaa_lf(levels, lowest, highest)
    real = np.array([
        76.0, 78.2, 80.1, 82.0, 84.0, 86.0, 87.8, 89.7, 91.5, 93.1, 94.6, 96.0, 97.1, 98.0, 98.7, 99.2, 99.5, 99.7,
        99.9, 99.9, 100.0, 100.0, 100.0, 100.0, 99.9, 99.7, 99.5, 99.2, 98.7, 98.0, 97.0, 95.5, 93.7, 91.4, 88.7, 85.4,
        82.1, 78.6, 74.7, 71.0, 67.2, 63.0, 59.2
    ])
    assert_array_almost_equal(real, generated)


def test_noaa_mf():
    generated = noaa_mf(levels, lowest, highest)
    real = np.array([
        10.1, 13.5, 16.6, 19.7, 22.9, 26.2, 29.3, 32.6, 35.9, 39.0, 42.1, 45.5, 48.6, 51.7, 54.9, 58.2, 61.3, 64.5,
        67.8, 70.9, 73.9, 77.3, 80.3, 83.2, 86.1, 88.9, 91.4, 93.7, 95.6, 97.1, 98.3, 99.2, 99.7, 100.0, 100.0, 99.8,
        99.4, 98.6, 97.4, 95.9, 94.0, 91.3, 88.5
    ])
    assert_array_almost_equal(real, generated)


def test_noaa_hf():
    generated = noaa_hf(levels, lowest, highest)
    real = np.array([
        -6.0, -2.1, 1.3, 4.8, 8.4, 12.2, 15.7, 19.3, 23.0, 26.5, 30.0, 33.9, 37.3, 40.8, 44.4, 48.2, 51.7,
        55.3, 59.0, 62.5, 65.9, 69.7, 73.1, 76.5, 79.9, 83.4, 86.4, 89.4, 92.1, 94.3, 96.2, 97.8, 98.8,
        99.5, 99.9, 100.0, 99.9, 99.5, 98.7, 97.7, 96.2, 94.1, 91.7
    ])
    assert_array_almost_equal(real, generated)


def test_noaa_pw():
    generated = noaa_pw(levels, lowest, highest)
    real = np.array([
        57.1, 59.3, 61.2, 63.1, 65.1, 67.2, 69.2, 71.2, 73.2, 75.2, 77.1, 79.2, 81.1, 83.1, 85.0, 87.0, 88.9, 90.7,
        92.5, 94.1, 95.5, 96.9, 97.9, 98.7, 99.3, 99.7, 99.9, 100.0, 99.9, 99.7, 99.3, 98.5, 97.5, 96.1, 94.3, 91.9,
        89.2, 86.1, 82.6, 79.1, 75.5, 71.4, 67.6
    ])
    assert_array_almost_equal(real, generated)


def test_noaa_ow():
    generated = noaa_ow(levels, lowest, highest)
    real = np.array([
        25.6, 29.9, 33.8, 37.6, 41.6, 45.8, 49.6, 53.6, 57.8, 61.6, 65.4, 69.6, 73.4, 77.0, 80.7, 84.3, 87.5, 90.5,
        93.1, 95.1, 96.7, 98.0, 98.9, 99.4, 99.8, 100.0, 100.0, 99.9, 99.7, 99.3, 98.7, 97.6, 96.3, 94.6, 92.4, 89.6,
        86.7, 83.3, 79.6, 76.0, 72.3, 68.2, 64.4
    ])
    assert_array_almost_equal(real, generated)


def teardown_module(weighting):
    pass

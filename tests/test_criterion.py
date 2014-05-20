import numpy as np
from numpy.testing import assert_array_equal

import pytest

from acoustics.criterion import nc_curve, nc


@pytest.mark.parametrize("nc, expected", [
    (15, np.array([47, 36, 29, 22, 17, 14, 12, 11])),
    (20, np.array([51, 40, 33, 26, 22, 19, 17, 16])),
    (25, np.array([54, 44, 37, 31, 27, 24, 22, 21])),
    (30, np.array([57, 48, 41, 35, 31, 29, 28, 27])),
    (35, np.array([60, 52, 45, 40, 36, 34, 33, 32])),
    (40, np.array([64, 56, 50, 45, 41, 39, 38, 37])),
    (45, np.array([67, 60, 54, 49, 46, 44, 43, 42])),
    (50, np.array([71, 64, 58, 54, 51, 49, 48, 47])),
    (55, np.array([74, 67, 62, 58, 56, 54, 53, 52])),
    (60, np.array([77, 71, 67, 63, 61, 59, 58, 57])),
    (65, np.array([80, 75, 71, 68, 66, 64, 63, 62])),
    (70, np.array([83, 79, 75, 72, 71, 70, 69, 68])),
    (11, None),
    (79, None),
])
def test_nc_curve(nc, expected):
    curve = nc_curve(nc)
    assert_array_equal(curve, expected)


@pytest.mark.parametrize("levels, expected", [
    (np.array([64, 56, 50, 45, 41, 39, 38, 37]), 40),
    (np.array([65, 56, 50, 45, 41, 39, 38, 37]), 45),
    (np.zeros(8), 15),
    (np.array([82, 78, 74, 71, 70, 69, 68, 67]), 70),
    (np.array([84, 80, 76, 73, 72, 71, 70, 69]), '70+'),
])
def test_nc(levels, expected):
    calculated = nc(levels)
    assert calculated == expected

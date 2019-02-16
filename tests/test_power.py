import numpy as np
from numpy.testing import assert_almost_equal

import pytest

from acoustics.power import lw_iso3746


@pytest.mark.parametrize("background_noise, expected", [
    (79, 91.153934187),
    (83, 90.187405234),
    (88, 88.153934187),
])
def test_lw_iso3746(background_noise, expected):
    LpAi = np.array([90, 90, 90, 90])
    LpAiB = background_noise * np.ones(4)
    S = 10
    alpha = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    surfaces = np.array([10, 10, 10, 10, 10, 10])
    calculated = lw_iso3746(LpAi, LpAiB, S, alpha, surfaces)
    assert_almost_equal(calculated, expected)

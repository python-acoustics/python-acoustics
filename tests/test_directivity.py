from acoustics.directivity import *
import numpy as np
import pytest


@pytest.mark.parametrize("given, expected, uncertainty", [
    (0.0, 1.0, 0.0),
    (1. / 2. * np.pi, 0.0, 0.0),
    (np.pi, +1.0, 0.0),
    (3. / 2. * np.pi, 0.0, 0.0),
    (2. * np.pi, +1.0, 0.0),
])
def test_figure_eight(given, expected, uncertainty):
    assert figure_eight(given) == pytest.approx(expected, uncertainty)

from __future__ import division

import numpy as np


def mean_alpha(alphas, surfaces):
    """
    Calculate mean of absorption coefficients.
    """
    return np.average(alphas, axis=0, weights=surfaces)


def nrc(alphas):
    """
    Calculate Noise Reduction Coefficient (NRC) from four absorption
    coefficient values (250, 500, 1000 and 2000 Hz).
    """
    alpha_axis = alphas.ndim - 1
    return np.mean(alphas, axis=alpha_axis)

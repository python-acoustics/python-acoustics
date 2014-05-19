from __future__ import division

import numpy as np


def esum(levels, axis=1):
    """
    Energetic summation.
    """
    levels = _is_1d(levels)
    return 10 * np.log10(np.sum(10**(levels/10), axis=axis))


def mean_tl(tl, surfaces):
    try:
        tau_axis = tl.ndim - 1
    except AttributeError:
        tau_axis = 0
    tau = 1 / (10**(tl/10))
    return 10 * np.log10(1 / np.average(tau, tau_axis, surfaces))


def wavelength(freq, c=343):
    """
    Wavelength for one or more frequencies (as ``NumPy array``).
    """
    return c/freq


def w(freq):
    """
    Angular frequency for one o more frequencies (as ``NumPy array``).
    """
    return 2 * np.pi * freq


def _is_1d(input):
    if type(input) is int or type(input) is float:
        return input
    elif input.ndim == 1:
        return np.array([input])
    elif input.ndim == 2 and input.shape[0] == 1:
        return input[0]
    else:
        return input

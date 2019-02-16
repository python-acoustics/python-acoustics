"""
Utils
=====

"""
import numpy as np
from acoustics.decibel import dbsum

SOUNDSPEED = 343.0
"""
Speed of sound in air.
"""

esum = dbsum


def mean_tl(tl, surfaces):
    """Mean tl."""
    try:
        tau_axis = tl.ndim - 1
    except AttributeError:
        tau_axis = 0
    tau = 1.0 / (10.0**(tl / 10.0))
    return 10.0 * np.log10(1.0 / np.average(tau, tau_axis, surfaces))


def wavelength(freq, c=SOUNDSPEED):
    """
    Wavelength for one or more frequencies (as ``NumPy array``).
    """
    return c / freq


def w(freq):
    """
    Angular frequency for one o more frequencies (as ``NumPy array``).
    """
    return 2.0 * np.pi * freq


def _is_1d(given):
    if isinstance(given, (int, float)):
        return given
    elif given.ndim == 1:
        return np.array([given])
    elif given.ndim == 2 and given.shape[0] == 1:
        return given[0]
    else:
        return given

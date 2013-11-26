from __future__ import division

import numpy as np


def _e10(x):
    return 10**(x/10)


def esum(x):
    levels = _is_1d(x)
    levels_sum = 10 * np.log10(np.sum(_e10(levels), axis=1))
    return levels_sum


def mean_TL(TL, surfaces):
    try:
        tau_axis = TL.ndim - 1
    except AttributeError:
        tau_axis = 0
    tau = 1 / (10**(TL/10))
    return 10 * np.log10(1 / np.average(tau, tau_axis, surfaces))


def _is_1d(input):
    if type(input) is int or type(input) is float:
        return input
    elif input.ndim == 1:
        return np.array([input])
    elif input.ndim == 2 and input.shape[0] == 1:
        return input[0]
    else:
        return input

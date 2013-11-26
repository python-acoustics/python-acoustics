from __future__ import division

import numpy as np


def wavelength(freq, c=343):
    '''
    Wavelength for one or more frequencies (as ``NumPy array``).
    '''
    return c/freq


def w(freq):
    '''
    Angular frequency for one o more frequencies (as ``NumPy array``).
    '''
    return 2 * np.pi * freq

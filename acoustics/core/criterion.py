from __future__ import division

import numpy as np


def nc_curve(nc):
    """Return an array containing the `nc` curve.

    Parameter:

    nc: `int` between 15 and 70 with step of 5. Valid values are: 15, 20, 25,
    ..., 60, 65 and 70. Invalid values (e.g. 23) returns `None`.
    """
    if nc == 15:
        curve = np.array([47, 36, 29, 22, 17, 14, 12, 11])
    elif nc == 20:
        curve = np.array([51, 40, 33, 26, 22, 19, 17, 16])
    elif nc == 25:
        curve = np.array([54, 44, 37, 31, 27, 24, 22, 21])
    elif nc == 30:
        curve = np.array([57, 48, 41, 35, 31, 29, 28, 27])
    elif nc == 35:
        curve = np.array([60, 52, 45, 40, 36, 34, 33, 32])
    elif nc == 40:
        curve = np.array([64, 56, 50, 45, 41, 39, 38, 37])
    elif nc == 45:
        curve = np.array([67, 60, 54, 49, 46, 44, 43, 42])
    elif nc == 50:
        curve = np.array([71, 64, 58, 54, 51, 49, 48, 47])
    elif nc == 55:
        curve = np.array([74, 67, 62, 58, 56, 54, 53, 52])
    elif nc == 60:
        curve = np.array([77, 71, 67, 63, 61, 59, 58, 57])
    elif nc == 65:
        curve = np.array([80, 75, 71, 68, 66, 64, 63, 62])
    elif nc == 70:
        curve = np.array([83, 79, 75, 72, 71, 70, 69, 68])
    else:
        curve = None
    return curve


def nc(levels):
    """
    It returns the NC curve of `levels`. If `levels` is upper than NC-70
    returns '70+'.

    Parameter:

    levels: 1-D NumPy array containing values between 63 Hz and 8 kHz in octave
    bands.
    """
    nc_range = np.arange(15, 71, 5)
    for nc_test in nc_range:
        curve = nc_curve(nc_test)
        if (levels <= curve).all() == True:
            break
        if nc_test == 70:
            nc_test = '70+'
            break
    return nc_test

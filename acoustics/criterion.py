"""
Criterion
=========

"""
import numpy as np

NC_CURVES = {
    15: np.array([47.0, 36.0, 29.0, 22.0, 17.0, 14.0, 12.0, 11.0]),
    20: np.array([51.0, 40.0, 33.0, 26.0, 22.0, 19.0, 17.0, 16.0]),
    25: np.array([54.0, 44.0, 37.0, 31.0, 27.0, 24.0, 22.0, 21.0]),
    30: np.array([57.0, 48.0, 41.0, 35.0, 31.0, 29.0, 28.0, 27.0]),
    35: np.array([60.0, 52.0, 45.0, 40.0, 36.0, 34.0, 33.0, 32.0]),
    40: np.array([64.0, 56.0, 50.0, 45.0, 41.0, 39.0, 38.0, 37.0]),
    45: np.array([67.0, 60.0, 54.0, 49.0, 46.0, 44.0, 43.0, 42.0]),
    50: np.array([71.0, 64.0, 58.0, 54.0, 51.0, 49.0, 48.0, 47.0]),
    55: np.array([74.0, 67.0, 62.0, 58.0, 56.0, 54.0, 53.0, 52.0]),
    60: np.array([77.0, 71.0, 67.0, 63.0, 61.0, 59.0, 58.0, 57.0]),
    65: np.array([80.0, 75.0, 71.0, 68.0, 66.0, 64.0, 63.0, 62.0]),
    70: np.array([83.0, 79.0, 75.0, 72.0, 71.0, 70.0, 69.0, 68.0])
}
"""
NC curves.
"""


def nc_curve(nc):
    """Return an array containing the `nc` curve.

    Parameter:

    nc: `int` between 15 and 70 with step of 5. Valid values are: 15, 20, 25,
    ..., 60, 65 and 70. Invalid values (e.g. 23) returns `None`.
    """
    return NC_CURVES.get(nc)


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
        if (levels <= curve).all():
            break
        if nc_test == 70:
            nc_test = '70+'
            break
    return nc_test  # pylint: disable=undefined-loop-variable


__all__ = ['NC_CURVES', 'nc_curve', 'nc']

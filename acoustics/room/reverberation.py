from __future__ import division

import numpy as np

from ..utils.utils import _is_1d


def t60_sabine(surfaces, alpha, volume, c=343):
    """
    Calculate reverberation time according to Sabine's formula:

    .. math:: T_{60} = \\frac{4 ln(10^6)}{c} \\frac{V}{S\\alpha}

    Where:

     - :math:`c`: speed of sound.
     - :math:`V`: Volume of the room.
     - :math:`S`: Surface of the room.
     - :math:`\\alpha`: Absorption coefficient of the room.

    Parameters:

    surfaces : ndarray
        NumPy array that contains different surfaces.

    alpha : ndarray
        Contains absorption coefficients of ``surfaces``.
        It could be one value or some values in different bands (1D and 2D
        array, respectively).

    volume : float
        Volume of the room.
    c : float
        Speed of sound (343 m/s by default).
    """
    mean_alpha = np.average(alpha, axis=0, weights=surfaces)
    S = np.sum(surfaces, axis=0)
    A = S * mean_alpha
    t60 = 4 * np.log(10**6) * volume / (c * A)
    return t60


def t60_eyring(surfaces, alpha, volume, c=343):
    mean_alpha = np.average(alpha, axis=0, weights=surfaces)
    S = np.sum(surfaces, axis=0)
    A = -S * np.log(1-mean_alpha)
    t60 = 4 * np.log(10**6) * volume / (c * A)
    return t60


def t60_millington(surfaces, alpha, volume, c=343):
    mean_alpha = np.average(alpha, axis=0, weights=surfaces)
    A = -np.sum(surfaces[:, np.newaxis] * np.log(1 - mean_alpha), axis=0)
    t60 = 4 * np.log(10**6) * volume / (c * A)
    return t60


def t60_fitzroy(surfaces, alpha, volume, c=343):
    Sx = np.sum(surfaces[0:2])
    Sy = np.sum(surfaces[2:4])
    Sz = np.sum(surfaces[4:6])
    St = np.sum(surfaces)
    alpha = _is_1d(alpha)
    a_x = np.average(alpha[:, 0:2], weights=surfaces[0:2], axis=1)
    a_y = np.average(alpha[:, 2:4], weights=surfaces[2:4], axis=1)
    a_z = np.average(alpha[:, 4:6], weights=surfaces[4:6], axis=1)
    factor = -(Sx / np.log(1-a_x) + Sy / np.log(1-a_y) + Sz / np.log(1-a_z))
    t60 = 4 * np.log(10**6) * volume * factor / (c * St**2)
    return t60


def t60_arau(Sx, Sy, Sz, alpha, volume, c=343):
    """
    Calculate reverberation time according to Arau's formula. [#arau]_

    ``Sx``: Sum of side walls.

    ``Sy``: Sum of other side walls.

    ``Sz``: Sum of room and floor surfaces.

    ``alpha``: Absorption coefficients for Sx, Sy and Sz, respectively.

    ``volume``: Volume of the room.

    ``c``: Speed of sound.

    .. [#arau] For more details, plase see
       http://www.arauacustica.com/files/publicaciones/pdf_esp_7.pdf
    """
    a_x = -np.log(1 - alpha[0])
    a_y = -np.log(1 - alpha[1])
    a_z = -np.log(1 - alpha[2])
    St = np.sum(np.array([Sx, Sy, Sz]))
    A = St * a_x**(Sx/St) * a_y**(Sy/St) * a_z**(Sz/St)
    t60 = 4 * np.log(10**6) * volume / (c * A)
    return t60

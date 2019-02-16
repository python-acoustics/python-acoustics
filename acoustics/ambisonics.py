"""
Ambisonics
==========

"""

import numpy as np
from scipy.misc import factorial


def acn(order=1):
    """Spherical harmonic degree `n` and order `m` for ambisonics order `order`.

    :param order: Ambisonics order.

    Yields tuples `(n, m)` where `n` is the degree and `m` the order.

    Follows ACN.

    === == == ======
    ACN n  m  letter
    === == == ======
    0   0  0  W
    --- -- -- ------
    1   1  -1 Y
    --- -- -- ------
    2   1  0  Z
    --- -- -- ------
    3   1  +1 X
    === == == ======

    """
    for n in range(order + 1):
        for m in range(-n, n + 1):
            yield (n, m)


def sn3d(m, n):
    """SN3D or Schmidt semi-normalisation

    :param m: order `n`
    :param n: degree `m`

    http://en.wikipedia.org/wiki/Ambisonic_data_exchange_formats#SN3D

    """
    m = np.atleast_1d(m)
    n = np.atleast_1d(n)

    d = np.logical_not(m.astype(bool))
    out = np.sqrt((2.0 - d) / (4.0 * np.pi) * factorial(n - np.abs(m)) / factorial(n + np.abs(m)))
    return out


def n3d(m, n):
    """N3D or full three-D normalisation

    :param m: order `n`
    :param n: degree `m`

    http://en.wikipedia.org/wiki/Ambisonic_data_exchange_formats#N3D

    """
    n = np.atleast_1d(n)
    return sn3d(m, n) * np.sqrt(2 * n + 1)


__all__ = ['acn', 'sn3d', 'n3d']

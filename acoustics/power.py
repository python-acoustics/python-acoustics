"""
Power
=====

"""
import numpy as np


def lw_iso3746(LpAi, LpAiB, S, alpha, surfaces):
    """
    Calculate sound power level according to ISO 3746:2010.

    :param LpAi: Sound pressure levels of the source :math:`L_{pAi}`.
    :type LpAi: :class:`np.ndarray`
    :param LpAiB: Background noise sound pressure levels :math:`L_{pAiB}`.
    :type LpAiB: :class:`np.ndarray`
    :param S: Area in square meters of the measurement surface :math:`S`.
    :type S: :class:`float`
    :param alpha: Absorption coefficients of the room :math:`\\alpha`.
    :type alpha: :class:`np.ndarray`
    :param surfaces: Room surfaces.
    :type surfaces: :class:`np.ndarray`

    :returns: Sound power level :math:`L_{w}`.
    """
    LpA = 10.0 * np.log10(np.sum(10.0**(0.1 * LpAi)) / LpAi.size)
    LpAB = 10.0 * np.log10(np.sum(10.0**(0.1 * LpAiB)) / LpAiB.size)
    deltaLpA = LpA - LpAB

    if deltaLpA > 10.0:
        k_1a = 0.0
    elif 3.0 <= deltaLpA <= 10.0:
        k_1a = -10.0 * np.log10(1.0 - 10.0**(-0.1 * deltaLpA))
    else:
        # This should alert to user because poor condition of the measurement.
        k_1a = 3.0

    S0 = 1.0
    Sv = np.sum(surfaces)
    alpha_mean = np.average(alpha, axis=0, weights=surfaces)
    A = alpha_mean * Sv

    k_2a = 10.0 * np.log10(1.0 + 4.0 * S / A)

    LpA_mean = LpA - k_1a - k_2a
    L_WA = LpA_mean + 10.0 * np.log10(S / S0)
    return L_WA

from __future__ import division

import numpy as np


def lw_iso3746(LpAi, LpAiB, S, alpha, surfaces):
    """
    Calculate sound power level according to ISO 3746:2010.
    
    :param LpAi: Sound pressure levels of the source.
    :type LpAi: :class:`np.ndarray`
    :param LpAiB: Background noise sound pressure levels.
    :type LpAiB: :class:`np.ndarray`
    :param S: Area in square meters of the measurement surface.
    :type S: :class:`float`
    :param alpha: Absorption coefficients of the room. 
    :type alpha: :class:`np.ndarray`
    :param surfaces: Room surfaces.
    :type :class:`np.ndarray`

    """
    LpA = 10 * np.log10(np.sum(10**(0.1*LpAi))/LpAi.size)
    LpAB = 10 * np.log10(np.sum(10**(0.1*LpAiB))/LpAiB.size)
    deltaLpA = LpA - LpAB

    if deltaLpA > 10:
        k_1a = 0
    elif 3 <= deltaLpA <= 10:
        k_1a = -10 * np.log10(1 - 10**(-0.1*deltaLpA))
    else:
        # This should alert to user because poor condition of the measurement.
        k_1a = 3

    S0 = 1
    Sv = np.sum(surfaces)
    alpha_mean = np.average(alpha, axis=0, weights=surfaces)
    A = alpha_mean * Sv

    k_2a = 10 * np.log10(1 + 4 * S / A)

    LpA_mean = LpA - k_1a - k_2a
    L_WA = LpA_mean + 10 * np.log10(S / S0)
    return L_WA

"""
Building
========

The building module contains functions related to building acoustics.

"""
import numpy as np


def rw_curve(tl):
    """
    Calculate the curve of :math:`Rw` from a NumPy array `tl` with third
    octave data between 100 Hz and 3.15 kHz.

    :param tl: Transmission Loss

    """
    ref_curve = np.array([0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 23, 23, 23, 23, 23])
    residuals = 0
    while residuals > -32:
        ref_curve += 1
        diff = tl - ref_curve
        residuals = np.sum(np.clip(diff, np.min(diff), 0))
    ref_curve -= 1
    return ref_curve


def rw(tl):
    """
    Calculate :math:`R_W` from a NumPy array `tl` with third octave data
    between 100 Hz and 3.15 kHz.

    :param tl: Transmission Loss
    """
    return rw_curve(tl)[7]


def rw_c(tl):
    """
    Calculate :math:`R_W + C` from a NumPy array `tl` with third octave data
    between 100 Hz and 3.15 kHz.

    :param tl: Transmission Loss
    """
    k = np.array([-29, -26, -23, -21, -19, -17, -15, -13, -12, -11, -10, -9, -9, -9, -9, -9])
    a = -10 * np.log10(np.sum(10**((k - tl) / 10)))
    return a


def rw_ctr(tl):
    """
    Calculate :math:`R_W + C_{tr}` from a NumPy array `tl` with third octave
    data between 100 Hz and 3.15 kHz.

    :param tl: Transmission Loss
    """
    k_tr = np.array([-20, -20, -18, -16, -15, -14, -13, -12, -11, -9, -8, -9, -10, -11, -13, -15])
    a_tr = -10 * np.log10(np.sum(10**((k_tr - tl) / 10)))
    return a_tr


def stc_curve(tl):
    """
    Calculate the Sound Transmission Class (STC) curve from a NumPy array `tl`
    with third octave data between 125 Hz and 4 kHz.

    :param tl: Transmission Loss
    """
    ref_curve = np.array([0, 3, 6, 9, 12, 15, 16, 17, 18, 19, 20, 20, 20, 20, 20, 20])
    top_curve = ref_curve
    res_sum = 0
    while True:
        diff = tl - top_curve
        residuals = np.clip(diff, np.min(diff), 0)
        res_sum = np.sum(residuals)
        if res_sum < -32:
            if np.any(residuals > -8):
                top_curve -= 1
                break
        top_curve += 1
    return top_curve


def stc(tl):
    """
    Calculate the Sound Transmission Class (STC) from a NumPy array `tl` with
    third octave data between 125 Hz and 4 kHz.

    :param tl: Transmission Loss
    """
    return stc_curve(tl)[6]


def mass_law(freq, vol_density, thickness, theta=0, c=343, rho0=1.225):
    """ Calculate transmission loss according to mass law.

    :param freq: Frequency of interest in Hz.
    :type freq: `float` or `NumPy array`
    :param vol_density: Volumetric density of material in [kg/m^3].
    :type vol_density: `float`
    :param thickness: Thickness of wall.
    :type thickness: `float`
    :param theta: Angle of incidence in degrees. Default value is `0` (normal incidence).
    :type theta: `float`
    :param c: Speed of sound in [m/s].
    :type c: `float`
    :param rho0: Density of air in kg/m^3.
    :type rho0: `float`
    """
    rad_freq = 2.0 * np.pi * freq
    surface_density = vol_density * thickness
    theta_rad = np.deg2rad(theta)
    a = rad_freq * surface_density * np.cos(theta_rad) / (2 * rho0 * c)
    tl_theta = 10 * np.log10(1 + a**2)
    return tl_theta


__all__ = ['rw_curve', 'rw', 'rw_c', 'rw_ctr', 'stc_curve', 'stc', 'mass_law']

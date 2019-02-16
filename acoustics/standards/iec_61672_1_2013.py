"""
IEC 61672-1:2013
================

IEC 61672-1:2013 gives electroacoustical performance specifications for three
kinds of sound measuring instruments [IEC61672]_:

- time-weighting sound level meters that measure exponential-time-weighted, frequency-weighted sound levels;
- integrating-averaging sound level meters that measure time-averaged, frequency-weighted sound levels; and
- integrating sound level meters that measure frequency-weighted sound exposure levels.

.. [IEC61672] http://webstore.iec.ch/webstore/webstore.nsf/artnum/048669!opendocument

Weighting functions
*******************

.. autofunction:: weighting_function_a
.. autofunction:: weighting_function_c
.. autofunction:: weighting_function_z


Weighting systems
*****************

.. autofunction:: weighting_system_a
.. autofunction:: weighting_system_c
.. autofunction:: weighting_system_z



"""
import io
import os
import pkgutil
import numpy as np
import pandas as pd
from scipy.signal import zpk2tf
from scipy.signal import lfilter, bilinear
from .iso_tr_25417_2007 import REFERENCE_PRESSURE


WEIGHTING_DATA = pd.read_csv(
    io.BytesIO(pkgutil.get_data('acoustics', os.path.join('data', 'iec_61672_1_2013.csv'))), sep=',', index_col=0)
"""DataFrame with indices, nominal frequencies and weighting values.
"""

NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES = np.array(WEIGHTING_DATA.nominal)
"""Nominal 1/3-octave frequencies. See table 3.
"""

NOMINAL_OCTAVE_CENTER_FREQUENCIES = np.array(WEIGHTING_DATA.nominal)[2::3]
"""Nominal 1/1-octave frequencies. Based on table 3.
"""

REFERENCE_FREQUENCY = 1000.0
"""Reference frequency. See table 3.
"""

EXACT_THIRD_OCTAVE_CENTER_FREQUENCIES = REFERENCE_FREQUENCY * 10.0**(0.01 * (np.arange(10, 44) - 30))
"""Exact third-octave center frequencies. See table 3.
"""

WEIGHTING_A = np.array(WEIGHTING_DATA.A)
"""Frequency weighting A. See table 3.
"""

WEIGHTING_C = np.array(WEIGHTING_DATA.C)
"""Frequency weighting C. See table 3.
"""

WEIGHTING_Z = np.array(WEIGHTING_DATA.Z)
"""Frequency weighting Z. See table 3.
"""

WEIGHTING_VALUES = {'A': WEIGHTING_A, 'C': WEIGHTING_C, 'Z': WEIGHTING_Z}
"""Dictionary with weighting values 'A', 'C' and 'Z' weighting.
"""

FAST = 0.125
"""FAST time-constant.
"""

SLOW = 1.000
"""SLOW time-constant.
"""


def time_averaged_sound_level(pressure, sample_frequency, averaging_time, reference_pressure=REFERENCE_PRESSURE):
    """Time-averaged sound pressure level.

    :param pressure: Dynamic pressure.
    :param sample_frequency: Sample frequency.
    :param averaging_time: Averaging time.
    :param reference_pressure: Reference pressure.

    """
    levels = 10.0 * np.log10(average(pressure**2.0, sample_frequency, averaging_time) / reference_pressure**2.0)
    times = np.arange(levels.shape[-1]) * averaging_time
    return times, levels


def average(data, sample_frequency, averaging_time):
    """Average the sound pressure squared.

    :param data: Energetic quantity, e.g. :math:`p^2`.
    :param sample_frequency: Sample frequency.
    :param averaging_time: Averaging time.
    :returns:

    Time weighting is applied by applying a low-pass filter with one real pole at :math:`-1/\\tau`.

    .. note::

        Because :math:`f_s \\cdot t_i` is generally not an integer, samples are discarded.
        This results in a drift of samples for longer signals (e.g. 60 minutes at 44.1 kHz).

    """
    averaging_time = np.asarray(averaging_time)
    sample_frequency = np.asarray(sample_frequency)
    samples = data.shape[-1]
    n = np.floor(averaging_time * sample_frequency).astype(int)
    data = data[..., 0:n * (samples // n)]  # Drop the tail of the signal.
    newshape = list(data.shape[0:-1])
    newshape.extend([-1, n])
    data = data.reshape(newshape)
    #data = data.reshape((-1, n))
    return data.mean(axis=-1)


def time_weighted_sound_level(pressure, sample_frequency, integration_time, reference_pressure=REFERENCE_PRESSURE):
    """Time-weighted sound pressure level.

    :param pressure: Dynamic pressure.
    :param sample_frequency: Sample frequency.
    :param integration_time: Integration time.
    :param reference_pressure: Reference pressure.
    """
    levels = 10.0 * np.log10(integrate(pressure**2.0, sample_frequency, integration_time) / reference_pressure**2.0)
    times = np.arange(levels.shape[-1]) * integration_time
    return times, levels


def integrate(data, sample_frequency, integration_time):
    """Integrate the sound pressure squared using exponential integration.

    :param data: Energetic quantity, e.g. :math:`p^2`.
    :param sample_frequency: Sample frequency.
    :param integration_time: Integration time.
    :returns:

    Time weighting is applied by applying a low-pass filter with one real pole at :math:`-1/\\tau`.

    .. note::

        Because :math:`f_s \\cdot t_i` is generally not an integer, samples are discarded.
        This results in a drift of samples for longer signals (e.g. 60 minutes at 44.1 kHz).

    """
    integration_time = np.asarray(integration_time)
    sample_frequency = np.asarray(sample_frequency)
    samples = data.shape[-1]
    b, a = zpk2tf([1.0], [1.0, integration_time], [1.0])
    b, a = bilinear(b, a, fs=sample_frequency)
    #b, a = bilinear([1.0], [1.0, integration_time], fs=sample_frequency) # Bilinear: Analog to Digital filter.
    n = np.floor(integration_time * sample_frequency).astype(int)
    data = data[..., 0:n * (samples // n)]
    newshape = list(data.shape[0:-1])
    newshape.extend([-1, n])
    data = data.reshape(newshape)
    #data = data.reshape((-1, n)) # Divide in chunks over which to perform the integration.
    return lfilter(
        b, a,
        data)[..., n - 1] / integration_time  # Perform the integration. Select the final value of the integration.


def fast(data, fs):
    """Apply fast (F) time-weighting.

    :param data: Energetic quantity, e.g. :math:`p^2`.
    :param fs: Sample frequency.

    .. seealso:: :func:`integrate`

    """
    return integrate(data, fs, FAST)
    #return time_weighted_sound_level(data, fs, FAST)


def slow(data, fs):
    """Apply slow (S) time-weighting.

    :param data: Energetic quantity, e.g. :math:`p^2`.
    :param fs: Sample frequency.

    .. seealso:: :func:`integrate`

    """
    return integrate(data, fs, SLOW)
    #return time_weighted_sound_level(data, fs, SLOW)


def fast_level(data, fs):
    """Time-weighted (FAST) sound pressure level.

    :param data: Dynamic pressure.
    :param fs: Sample frequency.

    .. seealso:: :func:`time_weighted_sound_level`

    """
    return time_weighted_sound_level(data, fs, FAST)


def slow_level(data, fs):
    """Time-weighted (SLOW) sound pressure level.

    :param data: Dynamic pressure.
    :param fs: Sample frequency.

    .. seealso:: :func:`time_weighted_sound_level`

    """
    return time_weighted_sound_level(data, fs, SLOW)


#---- Annex E - Analytical expressions for frequency-weightings C, A, and Z.-#

_POLE_FREQUENCIES = {
    1: 20.60,
    2: 107.7,
    3: 737.9,
    4: 12194.0,
}
"""Approximate values for pole frequencies f_1, f_2, f_3 and f_4.

See section E.4.1 of the standard.
"""

_NORMALIZATION_CONSTANTS = {
    'A': -2.000,
    'C': -0.062,
}
"""Normalization constants :math:`C_{1000}` and :math:`A_{1000}`.

See section E.4.2 of the standard.
"""


def weighting_function_a(frequencies):
    r"""A-weighting function in decibel.

    :param frequencies: Vector of frequencies at which to evaluate the weighting.
    :returns: Vector with scaling factors.

    The weighting curve is

    .. math:: 20 \log_{10}{\frac{(f_4^2 * f^4)}{(f^2 + f_1^2) \sqrt{(f^2 + f_2^2)(f^2 + f_3^2)}(f^2 + f_4^2)}} - A_{1000}

    with :math:`A_{1000} = -2` dB.

    See equation E.6 of the standard.

    """
    f = np.asarray(frequencies)
    offset = _NORMALIZATION_CONSTANTS['A']
    f1, f2, f3, f4 = _POLE_FREQUENCIES.values()
    weighting = 20.0 * np.log10((f4**2.0 * f**4.0) / (
        (f**2.0 + f1**2.0) * np.sqrt(f**2.0 + f2**2.0) * np.sqrt(f**2.0 + f3**2.0) * (f**2.0 + f4**2.0))) - offset
    return weighting


def weighting_function_c(frequencies):
    r"""C-weighting function in decibel.

    :param frequencies: Vector of frequencies at which to evaluate the weighting.
    :returns: Vector with scaling factors.

    The weighting curve is

    .. math:: 20 \log_{10}{\frac{(f_4^2 f^2)}{(f^2+f_1^2)(f^2+f_4^2)}} - C_{1000}

    with :math:`C_{1000} = -0.062` dB

    See equation E.1 of the standard.

    """
    f = np.asarray(frequencies)
    offset = _NORMALIZATION_CONSTANTS['C']
    f1, _, _, f4 = _POLE_FREQUENCIES.values()
    weighting = 20.0 * np.log10((f4**2.0 * f**2.0) / ((f**2.0 + f1**2.0) * (f**2.0 + f4**2.0))) - offset
    return weighting


def weighting_function_z(frequencies):
    """Z-weighting function in decibel.

    :param frequencies: Vector of frequencies at which to evaluate the weighting.
    :returns: Vector with scaling factors.

    """
    frequencies = np.asarray(frequencies)
    return np.zeros_like(frequencies)


WEIGHTING_FUNCTIONS = {
    'A': weighting_function_a,
    'C': weighting_function_c,
    'Z': weighting_function_z,
}
"""Dictionary with available weighting functions 'A', 'C' and 'Z'.
"""


def weighting_system_a():
    """A-weighting filter represented as polynomial transfer function.

    :returns: Tuple of `num` and `den`.

    See equation E.6 of the standard.

    """
    f1 = _POLE_FREQUENCIES[1]
    f2 = _POLE_FREQUENCIES[2]
    f3 = _POLE_FREQUENCIES[3]
    f4 = _POLE_FREQUENCIES[4]
    offset = _NORMALIZATION_CONSTANTS['A']
    numerator = np.array([(2.0 * np.pi * f4)**2.0 * (10**(-offset / 20.0)), 0.0, 0.0, 0.0, 0.0])
    part1 = [1.0, 4.0 * np.pi * f4, (2.0 * np.pi * f4)**2.0]
    part2 = [1.0, 4.0 * np.pi * f1, (2.0 * np.pi * f1)**2.0]
    part3 = [1.0, 2.0 * np.pi * f3]
    part4 = [1.0, 2.0 * np.pi * f2]
    denomenator = np.convolve(np.convolve(np.convolve(part1, part2), part3), part4)
    return numerator, denomenator


def weighting_system_c():
    """C-weighting filter represented as polynomial transfer function.

    :returns: Tuple of `num` and `den`.

    See equation E.1 of the standard.

    """
    f1 = _POLE_FREQUENCIES[1]
    f4 = _POLE_FREQUENCIES[4]
    offset = _NORMALIZATION_CONSTANTS['C']
    numerator = np.array([(2.0 * np.pi * f4)**2.0 * (10**(-offset / 20.0)), 0.0, 0.0])
    part1 = [1.0, 4.0 * np.pi * f4, (2.0 * np.pi * f4)**2.0]
    part2 = [1.0, 4.0 * np.pi * f1, (2.0 * np.pi * f1)**2.0]
    denomenator = np.convolve(part1, part2)
    return numerator, denomenator


def weighting_system_z():
    """Z-weighting filter represented as polynomial transfer function.

    :returns: Tuple of `num` and `den`.

    Z-weighting is 0.0 dB for all frequencies and therefore corresponds to a
    multiplication of 1.

    """
    numerator = [1]
    denomenator = [1]
    return numerator, denomenator


WEIGHTING_SYSTEMS = {
    'A': weighting_system_a,
    'C': weighting_system_c,
    'Z': weighting_system_z,
}
"""Weighting systems.
"""

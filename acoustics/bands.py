"""
Bands
=====

"""
import numpy as np
#from acoustics.decibel import dbsum
import acoustics
from acoustics.standards.iec_61672_1_2013 import (NOMINAL_OCTAVE_CENTER_FREQUENCIES,
                                                  NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES)

OCTAVE_CENTER_FREQUENCIES = NOMINAL_OCTAVE_CENTER_FREQUENCIES
THIRD_OCTAVE_CENTER_FREQUENCIES = NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES


def octave(first, last):
    """
    Generate a Numpy array for central frequencies of octave bands.

    There are more information on how to calculate 'real' bands in
    http://blog.prosig.com/2006/02/17/standard-octave-bands/

    Parameters
    ----------
    first : scalar
        First octave centerfrequency.

    last : scalar
        Last octave centerfrequency.

    Returns
    -------
    octave_bands : array
        An array of centerfrequency octave bands.
    """
    #octave_bands = OCTAVE_CENTER_FREQUENCIES
    #low = np.where(octave_bands == first)[0]
    #high = np.where(octave_bands == last)[0]
    #return octave_bands[low: high+1]
    return acoustics.signal.OctaveBand(fstart=first, fstop=last, fraction=1).nominal


def octave_low(first, last):
    """Lower cornerfrequencies of octaves."""
    return octave(first, last) / np.sqrt(2.0)
    #return acoustics.signal.OctaveBand(fstart=first, fstop=last, fraction=1).lower


def octave_high(first, last):
    """Upper cornerfrequencies of octaves."""
    return octave(first, last) * np.sqrt(2.0)
    #return acoustics.signal.OctaveBand(fstart=first, fstop=last, fraction=1).upper


def third(first, last):
    """
    Generate a Numpy array for central frequencies of third octave bands.

    Parameters
    ----------
    first : scalar
       First third octave centerfrequency.

    last : scalar
        Last third octave centerfrequency.

    Returns
    -------
    octave_bands : array
        An array of centerfrequency third octave bands.
    """
    #third_oct_bands = THIRD_OCTAVE_CENTER_FREQUENCIES
    #low = np.where(third_oct_bands == first)[0]
    #high = np.where(third_oct_bands == last)[0]
    #return third_oct_bands[low: high+1]
    return acoustics.signal.OctaveBand(fstart=first, fstop=last, fraction=3).nominal


def third_low(first, last):
    """Lower cornerfrequencies of third-octaves."""
    return third(first, last) / 2.0**(1.0 / 6.0)
    #return acoustics.signal.OctaveBand(fstart=first, fstop=last, fraction=3).lower


def third_high(first, last):
    """Higher cornerfrequencies of third-octaves."""
    return third(first, last) * 2.0**(1.0 / 6.0)
    #return Octaveband(fstart=first, fstop=last, fraction=3).upper


def third2oct(levels, axis=None):
    """
    Calculate Octave levels from third octave levels.

    :param levels: Array containing third octave levels.
    :type: :class:`np.ndarray`
    :param axis: Axis over which to perform the summation.
    :type axis: :class:`int`

    :returns: Third octave levels
    :rtype: :class:`np.ndarray`

    .. note:: The number of elements along the summation axis should be a factor of 3.
    """

    levels = np.array(levels)
    axis = axis if axis is not None else levels.ndim - 1

    try:
        assert levels.shape[axis] % 3 == 0
    except AssertionError:
        raise ValueError("Wrong shape.")
    shape = list(levels.shape)
    shape[axis] = shape[axis] // 3
    shape.insert(axis + 1, 3)
    levels = np.reshape(levels, shape)
    return np.squeeze(acoustics.decibel.dbsum(levels, axis=axis + 1))


def _check_band_type(freqs):
    """Check if an array contains octave or third octave bands values sorted
    or unsorted.
    """
    octave_bands = octave(16, 16000)
    third_oct_bands = third(12.5, 20000)

    def _check_sort(freqs, bands):
        index = np.where(np.in1d(bands, freqs))[0]
        band_pos = index - index[0]
        return (band_pos == np.arange(band_pos.size)).all()

    if np.in1d(freqs, octave_bands).all():
        is_sorted = _check_sort(freqs, octave_bands)
        band_type = "octave" if is_sorted else "octave-unsorted"
    elif np.in1d(freqs, third_oct_bands).all():
        is_sorted = _check_sort(freqs, third_oct_bands)
        band_type = "third" if is_sorted else "third-unsorted"
    else:
        band_type = None

    return band_type

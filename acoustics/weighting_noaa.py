"""
Weighting_NOAA
==============

NOAA weightings as defined
in 'Technical Guidance for Assessing the Effects of Anthropogenic Sound on Marine Mammal Hearing'
(https://www.fisheries.noaa.gov/resource/document/technical-guidance-assessing-effects-anthropogenic-sound-marine-mammal)


"""
import numpy as np

from acoustics.bands import third

THIRD_OCTAVE_NOAA_LF_WEIGHTING = np.array([
    -24.0, -21.8, -19.9, -18.0, -16.0, -14.0, -12.2, -10.3, -8.5, -6.9, -5.4, -4.0, -2.9, -2.0, -1.3, -0.8, -0.5, -0.3,
    -0.1, -0.1, -0.0, 0.0, -0.0, -0.0, -0.1, -0.3, -0.5, -0.8, -1.3, -2.0, -3.0, -4.5, -6.3, -8.6, -11.3, -14.6,
    -17.9, -21.4, -25.3, -29.0, -32.8, -37.0, -40.8
])
"""
NOAA LF-weighting filter for 1/3-octave band center frequencies, as specified by :attr:`acoustics.bands.third(12.5,200000)`.
"""

THIRD_OCTAVE_NOAA_MF_WEIGHTING = np.array([
    -89.9, -86.5, -83.4, -80.3, -77.1, -73.8, -70.7, -67.4, -64.1, -61.0, -57.9, -54.5, -51.4, -48.3, -45.1, -41.8,
    -38.7, -35.5, -32.2, -29.1, -26.1, -22.7, -19.7, -16.8, -13.9, -11.1, -8.6, -6.3, -4.4, -2.9, -1.7, -0.8, -0.3,
    -0.0, -0.0, -0.2, -0.6, -1.4, -2.6, -4.1, -6.0, -8.7, -11.5
])
"""
NOAA MF-weighting filter for 1/3-octave band center frequencies, as specified by :attr:`acoustics.bands.third(12.5,200000)`.
"""

THIRD_OCTAVE_NOAA_HF_WEIGHTING = np.array([
    -106.0, -102.1, -98.7, -95.2, -91.6, -87.8, -84.3, -80.7, -77.0, -73.5, -70.0, -66.1, -62.7, -59.2, -55.6, -51.8,
    -48.3, -44.7, -41.0, -37.5, -34.1, -30.3, -26.9, -23.5, -20.1, -16.6, -13.6, -10.6, -7.9, -5.7, -3.8, -2.2, -1.2,
    -0.5, -0.1, 0.0, -0.1, -0.5, -1.3, -2.3, -3.8, -5.9, -8.3
])
"""
NOAA HF-weighting filter for 1/3-octave band center frequencies, as specified by :attr:`acoustics.bands.third(12.5,200000)`.
"""

THIRD_OCTAVE_NOAA_PW_WEIGHTING = np.array([
    -42.9, -40.7, -38.8, -36.9, -34.9, -32.8, -30.8, -28.8, -26.8, -24.8, -22.9, -20.8, -18.9, -16.9, -15.0, -13.0,
    -11.1, -9.3, -7.5, -5.9, -4.5, -3.1, -2.1, -1.3, -0.7, -0.3, -0.1, -0.0, -0.1, -0.3, -0.7, -1.5, -2.5, -3.9, -5.7,
    -8.1, -10.8, -13.9, -17.4, -20.9, -24.5, -28.6, -32.4
])
"""
NOAA PW-weighting filter for 1/3-octave band center frequencies, as specified by :attr:`acoustics.bands.third(12.5,200000)`.
"""

THIRD_OCTAVE_NOAA_OW_WEIGHTING = np.array([
    -74.4, -70.1, -66.2, -62.4, -58.4, -54.2, -50.4, -46.4, -42.2, -38.4, -34.6, -30.4, -26.6, -23.0, -19.3, -15.7,
    -12.5, -9.5, -6.9, -4.9, -3.3, -2.0, -1.1, -0.6, -0.2, -0.0, -0.0, -0.1, -0.3, -0.7, -1.3, -2.4, -3.7, -5.4, -7.6,
    -10.4, -13.3, -16.7, -20.4, -24.0, -27.7, -31.8, -35.6
])
"""
NOAA OW-weighting filter for 1/3-octave band center frequencies, as specified by :attr:`acoustics.bands.third(12.5,200000)`.
"""


def noaa_lf_weighting(first, last):
    """
    Select frequency weightings between ``first`` and ``last``
    centerfrequencies from NOAA LF-weighting.
    Possible values for these frequencies are third-octave frequencies
    between 12.5 Hz and 200,000 Hz (including them).

    Parameters
    ----------
    first : scalar
       First third-octave centerfrequency.

    last : scalar
        Last third-octave centerfrequency.

    Returns
    -------
    NumPy array with NOAA LF-weighting between ``first`` and ``last``
    centerfrequencies.
    """
    return _weighting("noaa_lf", first, last)


def noaa_mf_weighting(first, last):
    """
    Select frequency weightings between ``first`` and ``last``
    centerfrequencies from NOAA MF-weighting.
    Possible values for these frequencies are third-octave frequencies
    between 12.5 Hz and 200,000 Hz (including them).

    Parameters
    ----------
    first : scalar
       First third-octave centerfrequency.

    last : scalar
        Last third-octave centerfrequency.

    Returns
    -------
    NumPy array with NOAA MF-weighting between ``first`` and ``last``
    centerfrequencies.
    """
    return _weighting("noaa_mf", first, last)


def noaa_hf_weighting(first, last):
    """
    Select frequency weightings between ``first`` and ``last``
    centerfrequencies from NOAA HF-weighting.
    Possible values for these frequencies are third-octave frequencies
    between 12.5 Hz and 200,000 Hz (including them).

    Parameters
    ----------
    first : scalar
       First third-octave centerfrequency.

    last : scalar
        Last third-octave centerfrequency.

    Returns
    -------
    NumPy array with NOAA HF-weighting between ``first`` and ``last``
    centerfrequencies.
    """
    return _weighting("noaa_hf", first, last)


def noaa_pw_weighting(first, last):
    """
    Select frequency weightings between ``first`` and ``last``
    centerfrequencies from NOAA PW-weighting.
    Possible values for these frequencies are third-octave frequencies
    between 12.5 Hz and 200,000 Hz (including them).

    Parameters
    ----------
    first : scalar
       First third-octave centerfrequency.

    last : scalar
        Last third-octave centerfrequency.

    Returns
    -------
    NumPy array with NOAA PW-weighting between ``first`` and ``last``
    centerfrequencies.
    """
    return _weighting("noaa_pw", first, last)



def noaa_ow_weighting(first, last):
    """
    Select frequency weightings between ``first`` and ``last``
    centerfrequencies from NOAA OW-weighting.
    Possible values for these frequencies are third-octave frequencies
    between 12.5 Hz and 200,000 Hz (including them).

    Parameters
    ----------
    first : scalar
       First third-octave centerfrequency.

    last : scalar
        Last third-octave centerfrequency.

    Returns
    -------
    NumPy array with NOAA OW-weighting between ``first`` and ``last``
    centerfrequencies.
    """
    return _weighting("noaa_ow", first, last)


def _weighting(filter_type, first, last):
    third_oct_bands = third(12.5, 200000.0).tolist()
    low = third_oct_bands.index(first)
    high = third_oct_bands.index(last)

    if filter_type == "noaa_lf":
        freq_weightings = THIRD_OCTAVE_NOAA_LF_WEIGHTING

    elif filter_type == "noaa_mf":
        freq_weightings = THIRD_OCTAVE_NOAA_MF_WEIGHTING

    elif filter_type == "noaa_hf":
        freq_weightings = THIRD_OCTAVE_NOAA_HF_WEIGHTING

    elif filter_type == "noaa_pw":
        freq_weightings = THIRD_OCTAVE_NOAA_PW_WEIGHTING

    elif filter_type == "noaa_ow":
        freq_weightings = THIRD_OCTAVE_NOAA_OW_WEIGHTING

    return freq_weightings[low:high + 1]


def noaa_lf(levels, first, last):
    """Apply NOAA LF-weighting to unweighted signal.
    """
    return levels + noaa_lf_weighting(first, last)


def noaa_mf(levels, first, last):
    """Apply NOAA MF-weighting to unweighted signal.
    """
    return levels + noaa_mf_weighting(first, last)


def noaa_hf(levels, first, last):
    """Apply NOAA HF-weighting to unweighted signal.
    """
    return levels + noaa_hf_weighting(first, last)


def noaa_pw(levels, first, last):
    """Apply NOAA PW-weighting to unweighted signal.
    """
    return levels + noaa_pw_weighting(first, last)


def noaa_ow(levels, first, last):
    """Apply NOAA OW-weighting to unweighted signal.
    """
    return levels + noaa_ow_weighting(first, last)


__all__ = [
    'THIRD_OCTAVE_NOAA_LF_WEIGHTING',
    'THIRD_OCTAVE_NOAA_MF_WEIGHTING',
    'THIRD_OCTAVE_NOAA_HF_WEIGHTING',
    'THIRD_OCTAVE_NOAA_PW_WEIGHTING',
    'THIRD_OCTAVE_NOAA_OW_WEIGHTING',
    'noaa_lf_weighting',
    'noaa_mf_weighting',
    'noaa_hf_weighting',
    'noaa_pw_weighting',
    'noaa_ow_weighting',
    'noaa_lf',
    'noaa_mf',
    'noaa_hf',
    'noaa_pw',
    'noaa_ow',
]

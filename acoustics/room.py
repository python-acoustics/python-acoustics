from __future__ import division

import numpy as np

from scipy.io import wavfile
from scipy import stats

from acoustics.utils.utils import _is_1d
from acoustics.signal import butter_bandpass_filter
from acoustics.core.bands import (_check_band_type, octave_low, octave_high,
                                  third_low, third_high)


def mean_alpha(alphas, surfaces):
    """
    Calculate mean of absorption coefficients.
    
    :param alphas: Absorption coefficients
    :param surfaces: Surfaces
    """
    return np.average(alphas, axis=0, weights=surfaces)


def nrc(alphas):
    """
    Calculate Noise Reduction Coefficient (NRC) from four absorption
    coefficient values (250, 500, 1000 and 2000 Hz).
    
    :param alphas: Absorption coefficients
    
    """
    alpha_axis = alphas.ndim - 1
    return np.mean(alphas, axis=alpha_axis)


def t60_sabine(surfaces, alpha, volume, c=343):
    """
    Reverberation time according to Sabine:

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
    """
    Reverberation time according to Eyring.
    
    :param surfaces: Surfaces
    :param alpha: Mean absorption coefficient or by frequency bands
    :param volume: Volume
    :param c: Speed of sound
    """
    mean_alpha = np.average(alpha, axis=0, weights=surfaces)
    S = np.sum(surfaces, axis=0)
    A = -S * np.log(1-mean_alpha)
    t60 = 4 * np.log(10**6) * volume / (c * A)
    return t60


def t60_millington(surfaces, alpha, volume, c=343):
    """
    Reverberation time according to Millington.
    
    :param surfaces: Surfaces
    :param alpha: Mean absorption coefficient or by frequency bands
    :param volume: Volume
    :param c: Speed of sound
    """
    mean_alpha = np.average(alpha, axis=0, weights=surfaces)
    A = -np.sum(surfaces[:, np.newaxis] * np.log(1 - mean_alpha), axis=0)
    t60 = 4 * np.log(10**6) * volume / (c * A)
    return t60


def t60_fitzroy(surfaces, alpha, volume, c=343):
    """
    Reverberation time according to Fitzroy.
    
    :param surfaces: Surfaces
    :param alpha: Mean absorption coefficient or by frequency bands
    :param volume: Volume
    :param c: Speed of sound
    """
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
    Reverberation time according to Arau. [#arau]_

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


def t60_impulse(file_name, bands, rt='t30'):
    """Reverberation time from a WAV impulse response.

    :param file_name: name of the WAV file containing the impulse response.
    :param bands: Octave or third bands as NumPy array.
    :param rt: Reverberation time estimator. It accepts `'t30'`, `'t20'`,
    `'t10'` and `'edt'`.
    """
    fs, raw_signal = wavfile.read(file_name)
    band_type = _check_band_type(bands)

    if band_type is 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type is 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    rt = rt.lower()
    if rt == 't30':
        init = -5
        end = -35
        factor = 2
    elif rt == 't20':
        init = -5
        end = -25
        factor = 3
    elif rt == 't10':
        init = -5
        end = -15
        factor = 6
    elif rt == 'edt':
        init = 0
        end = -10
        factor = 6

    t60 = np.zeros(bands.size)

    for band in range(bands.size):
        # Filtering signal
        filtered_signal = butter_bandpass_filter(raw_signal, low[band],
                                                 high[band], fs, order=3)
        abs_signal = np.abs(filtered_signal) / np.max(np.abs(filtered_signal))

        # Schroeder integration
        sch = np.cumsum(abs_signal[::-1]**2)[::-1]
        sch_db = 10 * np.log10(sch / np.max(sch))
        
        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]
        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]
        x = np.arange(init_sample, end_sample + 1) / fs
        y = sch_db[init_sample: end_sample + 1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

        # Reverberation time (T30, T20, T10 or EDT)
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60[band] = factor * (db_regress_end - db_regress_init)
    return t60

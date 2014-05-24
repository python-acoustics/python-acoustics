"""
IEC 61672-1:2013
================

IEC 61672-1:2013 gives electroacoustical performance specifications for three 
kinds of sound measuring instruments [IEC61672]_:

- time-weighting sound level meters that measure exponential-time-weighted, frequency-weighted sound levels;
- integrating-averaging sound level meters that measure time-averaged, frequency-weighted sound levels; and
- integrating sound level meters that measure frequency-weighted sound exposure levels. 

.. [IEC61672] http://webstore.iec.ch/webstore/webstore.nsf/artnum/048669!opendocument

.. inheritance-diagram:: acoustics.standards.iec_61672_1_2013

"""
import numpy as np
from scipy.signal import lfilter, bilinear
from math import floor
from .iso_tr_25417_2007 import REFERENCE_PRESSURE

NOMINAL_FREQUENCIES = np.array([10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 
                                100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 
                                800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 
                                6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0
                                ])
"""
Nominal frequencies.
"""

WEIGHTING_A = np.array([-70.4, -63.4, -56.7, -50.5, -44.7, -39.4, -34.6, -30.2, -26.2, 
                        -22.5, -19.1, -16.1, -13.4, -10.9, -8.6, -6.6, -4.8, -3.2, -1.9,
                        -0.8, 0.0, +0.6, +1.0, +1.2, +1.3, +1.2, +1.0, +0.5, -0.1, -1.1, -2.5, -4.3, -6.6, -9.3
                        ])
"""
Frequency weighting A.
"""

WEIGHTING_C = np.array([-14.3, -11.2,-8.5, -6.2, -4.4, -3.0, -2.0, -1.3, -0.8, -0.5, -0.3, 
                        -0.2, -0.1, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,0,  0.0,  
                        -0.1, -0.2, -0.3, -0.5, -0.8, -1.3, -2.0, -3.0, -4.4, -6.2, -8.5, -11.2]) 
"""
Frequency weighting C.
"""

WEIGHTING_Z = np.array([0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 
                        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
"""
Frequency weighting Z.
"""

def time_averaged_sound_level(pressure, sample_frequency, averaging_time, reference_pressure=REFERENCE_PRESSURE):
    """
    Time-averaged sound pressure level.
    
    :param pressure: Dynamic pressure.
    :param sample_frequency: Sample frequency.
    :param averaging_time: Averaging time.
    :param reference_pressure: Reference pressure.
    
    """
    return 10.0 * np.log10( average(pressure**2.0, sample_frequency, averaging_time) / reference_pressure**2.0)


def average(data, sample_frequency, averaging_time):
    """
    Average the sound pressure squared.
    
    :param data: Energetic quantity, e.g. :math:`p^2`.
    :param sample_frequency: Sample frequency.
    :param averaging_time: Averaging time.
    :returns: 
    
    Time weighting is applied by applying a low-pass filter with one real pole at :math:`-1/\\tau`.
    
    .. note:: Because $f_s \cdot t_i$ is generally not an integer, samples are discarded. This results in a drift of samples for longer signals (e.g. 60 minutes at 44.1 kHz).
    
    """
    n = int(floor(averaging_time * sample_frequency))
    data = data[0:n*(len(data)//n)]
    data = data.reshape((-1, n))
    return data.mean(axis=1)

def time_weighted_sound_level(pressure, sample_frequency, integration_time, reference_pressure=REFERENCE_PRESSURE):
    """
    Time-weighted sound pressure level.
    
    :param pressure: Dynamic pressure.
    :param sample_frequency: Sample frequency.
    :param integration_time: Integration time.
    :param reference_pressure: Reference pressure.
    """
    return 10.0 * np.log10( integrate(pressure**2.0, sample_frequency, integration_time) / reference_pressure**2.0)

    
def integrate(data, sample_frequency, integration_time):
    """
    Integrate the sound pressure squared using exponential integration.
    
    :param data: Energetic quantity, e.g. :math:`p^2`.
    :param sample_frequency: Sample frequency.
    :param integration_time: Integration time.
    :returns: 
    
    Time weighting is applied by applying a low-pass filter with one real pole at :math:`-1/\\tau`.
    
    .. note:: Because $f_s \cdot t_i$ is generally not an integer, samples are discarded. This results in a drift of samples for longer signals (e.g. 60 minutes at 44.1 kHz).
    
    """
    b, a = bilinear([1.0], [1.0, integration_time], sample_frequency=sample_frequency) # Bilinear: Analog to Digital filter.
    n = int(floor(integration_time * sample_frequency))
    data = data[0:n*(len(data)//n)]
    data = data.reshape((-1, n)) # Divide in chunks over which to perform the integration.
    return lfilter(b, a, data)[:,n-1] # Perform the integration. Select the final value of the integration.


def fast(data, fs):
    """
    Apply fast (F) time-weighting.
    """
    return integrate(data, fs, 0.125)

def slow(data, fs):
    """
    Apply slow (S) time-weighting.
    """
    return integrate(data, fs, 1.0)

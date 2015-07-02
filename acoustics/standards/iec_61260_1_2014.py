"""
IEC 61260-1:2014
================

IEC 61260-1:2014 specifies performance requirements for analogue, sampled-data, 
and digital implementations of band-pass filters. 


Frequency functions
**************************************

.. autofunction:: acoustics.standards.iec_61260_1_2014.exact_center_frequency
.. autofunction:: acoustics.standards.iec_61260_1_2014.lower_frequency
.. autofunction:: acoustics.standards.iec_61260_1_2014.upper_frequency
.. autofunction:: acoustics.standards.iec_61260_1_2014.index_of_frequency


Nominal center frequencies
**************************

.. autoattribute:: acoustics.standards.iec_61260_1_2014.NOMINAL_OCTAVE_CENTER_FREQUENCIES
.. autoattribute:: acoustics.standards.iec_61260_1_2014.NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES

.. autoattribute:: acoustics.standards.iec_61260_1_2014.REFERENCE_FREQUENCY
.. autoattribute:: acoustics.standards.iec_61260_1_2014.OCTAVE_FREQUENCY_RATIO

"""




"""

The normalized frequency :math:`\\Omega` is obtained by dividing the frequency :math:`f` by the reference frequency :math:`f_m`.

.. math::\\Omega = f / f_m.

"""
import numpy as np

NOMINAL_OCTAVE_CENTER_FREQUENCIES = np.array([31.5, 63.0, 125.0, 250.0, 
                                              500.0, 1000.0, 2000.0, 
                                              4000.0, 8000.0, 16000.0])
"""Nominal octave center frequencies.
"""

NOMINAL_THIRD_OCTAVE_CENTER_FREQUENCIES = np.array([25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 
                                                    200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 
                                                    1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 
                                                    6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0
                                                    ])
"""Nominal third-octave center frequencies in the audio range.
"""



REFERENCE_FREQUENCY = 1000.0
"""Reference frequency.
"""

OCTAVE_FREQUENCY_RATIO = 10.0**(3.0/10.0)
"""Octave frequency ratio :math:`G`.

See equation 1.
"""

def exact_center_frequency(x, fraction=1, ref=REFERENCE_FREQUENCY, G=OCTAVE_FREQUENCY_RATIO):
    """
    Center frequencies :math:`f_m` for band indices :math:`x`. See equation 2 and 3.
    
    :param x: Band index :math:`x`.
    :param ref: Reference center frequency :math:`f_r`.
    :param fraction: Bandwidth designator :math`b`. For example, for 1/3-octave filter b=3.
    :param G: Octave frequency ratio :math:`G`.
    
    The center frequencies are given by
    
    .. math:: f_m = f_r G^{x/b}
    
    In case the bandwidth designator :math:`b` is an even number, the center frequencies are given by
    
    .. math:: f_m = f_r G^{(2x+1)/2b}
    
    See equation 2 and 3 of the standard.
    """
    if fraction%2==0.0:
        return ref * G**((x+1) / (2.0*fraction))
    else:
        return ref * G**(x / fraction)


def lower_frequency(center, fraction=1, G=OCTAVE_FREQUENCY_RATIO):
    """
    Lower band-edge frequencies. See equation 4.
    
    :param center: Center frequencies :math:`f_m`.
    :param fraction: Bandwidth designator :math:`b`.
    :param G: Octave frequency ratio :math:`G`.
    
    The lower band-edge frequencies are given by
    
    .. math:: f_1 = f_m G^{-1/2b}
    
    See equation 4 of the standard.
    
    """
    return center * G**(-1.0 / (2.0*fraction))


def upper_frequency(center, fraction=1, G=OCTAVE_FREQUENCY_RATIO):
    """
    Upper band-edge frequencies. See equation 5.
    
    :param center: Center frequencies :math:`f_m`.
    :param fraction: Bandwidth designator :math:`b`.
    :param G: Octave frequency ratio :math:`G`.
    
    The upper band-edge frequencies are given by
    
    .. math:: f_2 = f_m G^{+1/2b}
    
    See equation 5 of the standard.
    
    """
    return center * G**(+1.0 / (2.0*fraction))


def index_of_frequency(frequency, fraction=1, ref=REFERENCE_FREQUENCY, G=OCTAVE_FREQUENCY_RATIO):
    """Determine the band index `x` from a given frequency.
    
    :param frequency: Frequencies :math:`f`.
    :param fraction: Bandwidth designator :math:`b`.
    :param ref: Reference frequency.
    :param G: Octave frequency ratio :math:`G`.
    
    The index of the center frequency is given by
    
    .. math:: x = round{b \\frac{\log{f/f_{ref} }}{\log{G} }}        
    
    .. note:: This equation is not part of the standard. However, it follows from :func:`exact_center_frequency`.
    
    """
    return np.round(fraction * np.log(frequency/ref) / np.log(G)).astype('int16')
    

#def nominal_center_frequency(x, fraction=1, G=OCTAVE_FREQUENCY_RATIO):
    #"""Nominal center frequencies.
    #"""
    #pass
    


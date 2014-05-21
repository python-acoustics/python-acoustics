"""
Module for working with octaves.

The following is an example on how to use :class:`acoustics.octave.Octave`.

.. literalinclude:: ../examples/octave.py

"""
from __future__ import division

import numpy as np

REFERENCE = 1000.0
"""
Reference frequency.
"""

def band_of_frequency(f, order=1, ref=REFERENCE):
    """
    Calculate the band ``n`` from a given center frequency.

    :param f: Frequency :math:`f`.
    :param order: Band order.
    :param ref: Reference center frequency :math:`f_0`.
    """
    return np.round( ( np.log2(f/ref) - 1.0/order ) * order)


def frequency_of_band(n, order=1, ref=REFERENCE):
    """
    Calculate center frequency of band ``n``.
    
    :param n: band ``n`.
    :param order: Order of octave.
    :param ref: Reference center frequency.
    """
    return ref * 10.0**(3.0/order/10.0) * 2.0**(n/order)


def upper_frequency(center, order=1):
    """
    Upper frequency of frequency band given a center frequency and order.
    
    :param centr: Center frequencies.
    :param order: Fraction of octave.
    
    .. math:: f_u = f_c \cdot 2^{\\frac{+1}{2N}}
    
    """
    return center * 2.0**(+1.0/(2.0*order))
    
    
def lower_frequency(center, order=1):
    """
    Lower frequency of frequency band given a center frequency and order.
    
    :param centr: Center frequencies.
    :param order: Fraction of octave.
    
    .. math:: f_l = f_c \cdot 2^{\\frac{-1}{2N}}
    
    """
    return center * 2.0**(-1.0/(2.0*order))    


class Octave(object):
    """
    Class to calculate octave center frequencies.
    """

    def __init__(self, order=1, interval=None, fmin=None, fmax=None, unique=False, reference=REFERENCE):
        
        self.reference = reference
        """
        Reference center frequency :math:`f_{c,0}`.
        """
        
        self.order = order
        """
        Fraction of octave.
        """
        
        if (interval is not None) and (fmin is not None or fmax is not None):
            raise AttributeError
        
        self._interval = interval
        """Interval"""
        
        self._fmin = fmin
        """Minimum frequency of a range."""
        
        self._fmax = fmax
        """Maximum frequency of a range."""
        
        self.unique = unique
        """Whether or not to calculate the requested values for every value of ``interval``."""
        
        
    @property
    def fmin(self):
        """Minimum frequency of an interval."""
        if self._fmin is not None:
            return self._fmin
        elif self._interval is not None:
            return self.interval.min()
    
    @fmin.setter
    def fmin(self, x):
        if self.interval is not None:
            pass    # Warning, remove interval first.
        else:
            self._fmin = x
    
    @property
    def fmax(self):
        """Maximum frequency of an interval."""
        if self._fmax is not None:
            return self._fmax
        elif self._interval is not None:
            return self.interval.max()
    
    @fmax.setter
    def fmax(self, x):
        if self.interval is not None:
            pass
        else:
            self._fmax = x

    @property
    def interval(self):
        """Interval."""
        return self._interval
    
    @interval.setter
    def interval(self, x):
        if self._fmin or self._fmax:
            pass
        else:
            self._interval = x if isinstance(x, np.ndarray) else np.array(x)
        
    def _n(self, f):
        """
        Calculate the band ``n`` from a given frequency.
        
        :param f: Frequency
        
        See also :func:`band_of_frequency`.
        """
        return band_of_frequency(f, order=self.order, ref=self.reference)
        
    def _fc(self, n):
        """
        Calculate center frequency of band ``n``.
        
        :param n: band ``n`.
        
        See also :func:`frequency_of_band`.
        """
        return frequency_of_band(n, order=self.order, ref=self.reference)
        
    def n(self):
        """
        Return band ``n`` for a given frequency.
        """
        if self.interval is not None and self.unique:
            return self._n(self.interval)
        else:
            return np.arange(self._n(self.fmin), self._n(self.fmax)+1)
            
    def center(self):
        """
        Return center frequencies :math:`f_c`.
        
        .. math::  f_c = f_{ref} \cdot 2^{n/N} \\cdot 10^{\\frac{3}{10N}}
        
        """
        n = self.n()
        return self._fc(n)
     
    def bandwidth(self):
        """
        Bandwidth of bands.
        
        .. math:: B = f_u - f_l
        
        """
        return self.upper() - self.lower()
    
    def lower(self):
        """
        Lower frequency limits of bands.
        
        .. math:: f_l = f_c \cdot 2^{\\frac{-1}{2N}}
        
        See also :func:`lower_frequency`.
        """
        return lower_frequency(self.center, self.order)
        
    def upper(self):
        """
        Upper frequency limits of bands.
        
        .. math:: f_u = f_c \cdot 2^{\\frac{+1}{2N}}
        
        See also :func:`upper_frequency`.
        """
        return upper_frequency(self.center, self.order)
        
    
###def center_frequency_octave(frequencies, order=1):
    ###"""
    ###Calculate the center frequencies :math:`f_c` of the octaves that (partially) cover ``frequencies``.
    
    ###:param frequencies: An iterable containing frequencies.
    ###:param order: An integer indicating the octave order. E.g., for 1/3-octaves use ``order=3``
    
    ###Center frequencies are calculated using:
    
    ###.. math::  f_c = 1000 \cdot 2^{n/N} \\cdot 10^{\\frac{3}{10N}}
    
    ###"""
    ###n = lambda fc, N: np.log2(fc/1000.0) - 1.0/N    # To calculate the nth octave from a given frequency.
    
    ###n_min = np.floor(n(np.min(frequencies), order))
    ###n_max = np.ceil(n(np.max(frequencies), order))
    ###n = np.arange(n_min, n_max+1)
    ###fc = 1000.0 * 10.0**(3.0/order/10.0) * 2.0**(n)
    ###return fc
